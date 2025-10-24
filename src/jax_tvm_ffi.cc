// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <dlpack/dlpack.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <xla/ffi/api/ffi.h>

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>

namespace jax_tvm_ffi {

// decode action kind
enum class DecodeKind {
  // place all the arguments here
  kArgs,
  // place all the returns here
  kRets,
  // place one attribute value here
  kAttrValue,
  // place the CUDA/platform stream as int64_t here
  kContextStream
};

// Decode Item used to decode a call frame into the call stack
struct DecodeItem {
  // The kind of the decoding item
  DecodeKind kind;
  // Sorted attribute index when decoding attributes
  // XLA passes attributes in sorted order
  // expected attribute index, which can be found by sorting attributes by name
  // and the expected index in the sorted list
  size_t sorted_attr_index{0};
};

// specification for the decoding item
struct DecodeSpec {
  /*! \brief The decoding items */
  std::vector<DecodeItem> items;
  /*! \brief sorted attribute keys */
  std::vector<std::pair<std::string, size_t>> sorted_attr_keys;
};

/*!
 * \brief Parse the arg spec string into a DecodeItem sequence
 * \param arg_spec The arg spec array of strings
 * \return The DecodeSpec used in function handler
 */
DecodeSpec ParseArgSpec(tvm::ffi::Array<tvm::ffi::String> arg_spec) {
  std::vector<DecodeItem> items;
  items.reserve(arg_spec.size());
  std::vector<std::pair<std::string, size_t>> attr_keys;

  for (size_t i = 0; i < static_cast<size_t>(arg_spec.size()); ++i) {
    std::string spec_item = arg_spec[i];
    if (spec_item == "args") {
      items.push_back(DecodeItem{DecodeKind::kArgs, 0});
    } else if (spec_item == "rets") {
      items.push_back(DecodeItem{DecodeKind::kRets, 0});
    } else if (spec_item == "ctx.stream") {
      items.push_back(DecodeItem{DecodeKind::kContextStream, 0});
    } else if (spec_item.compare(0, 6, "attrs.") == 0) {
      std::string attr_key = spec_item.substr(6);
      items.push_back(DecodeItem{DecodeKind::kAttrValue, 0});
      attr_keys.push_back(std::make_pair(attr_key, i));
    } else {
      TVM_FFI_THROW(RuntimeError) << "Invalid arg spec: " << spec_item
                                  << ". Expected 'args', 'rets', 'ctx.stream', or 'attrs.<key>'";
    }
  }
  // sort the attributes by their names
  std::sort(attr_keys.begin(), attr_keys.end(),
            [](const std::pair<std::string, size_t>& a, const std::pair<std::string, size_t>& b) {
              return a.first < b.first;
            });
  for (size_t i = 0; i < attr_keys.size(); ++i) {
    items[attr_keys[i].second].sorted_attr_index = i;
  }
  return DecodeSpec{items, attr_keys};
}

/*!
 * \brief RAII call context that manages the states and stack for the call.
 */
class CallContext {
 public:
  /*! \brief stack for the call context */
  class Stack {
   public:
    // default constructor
    Stack() = default;
    /*! \brief The workspace for the packed arguments */
    std::vector<tvm::ffi::AnyView> packed_args;
    /*! \brief The workspace to hold temp attributes */
    std::vector<tvm::ffi::Any> temp_attrs;
    /*! \brief Reserve the workspace for the temp DLTensors */
    TVM_FFI_INLINE void ReserveTempDLTensors(size_t num_args, size_t num_rets) {
      temp_dltensors_.resize(num_args + num_rets);
      dltensor_total_ndims_ = 0;
      dltensor_fp4_total_ndims_ = 0;
      dltensor_alloc_count_ = 0;
    }

    /*!
     * \brief Allocate a temp DLTensor
     * \param ndim The number of dimensions of the DLTensor
     * \param need_fp4_packing Whether this is an FP4 tensor that needs shape transformation
     * \return The allocated DLTensor
     */
    TVM_FFI_INLINE DLTensor* AllocTempDLTensor(size_t ndim, bool need_fp4_packing = false) {
      // invariance, this should never happen as we reserve the workspace in advance
      TVM_FFI_ICHECK_LT(dltensor_alloc_count_, temp_dltensors_.size());
      dltensor_total_ndims_ += ndim;
      if (need_fp4_packing) {
        dltensor_fp4_total_ndims_ += ndim;
      }
      return &temp_dltensors_[dltensor_alloc_count_++];
    }

    TVM_FFI_INLINE void FillStridesForTempDLTensors() {
      // invariance, this should never happen as we reserve the workspace in advance
      TVM_FFI_ICHECK_LE(dltensor_alloc_count_, temp_dltensors_.size());
      temp_strides_.resize(dltensor_total_ndims_);
      size_t strides_begin = 0;
      for (size_t i = 0; i < dltensor_alloc_count_; ++i) {
        temp_dltensors_[i].strides = temp_strides_.data() + strides_begin;
        tvm::ffi::details::FillStridesFromShape(
            tvm::ffi::ShapeView(temp_dltensors_[i].shape, temp_dltensors_[i].ndim),
            temp_dltensors_[i].strides);
        strides_begin += temp_dltensors_[i].ndim;
      }
    }

    /*!
     * \brief Fill shapes for FP4 tensors that need shape transformation
     * \note This must be called before FillStridesForTempDLTensors since strides
     *       computation depends on the (potentially modified) shape.
     */
    TVM_FFI_INLINE void FillShapesForFP4Tensors() {
      if (dltensor_fp4_total_ndims_ == 0) return;  // No FP4 tensors, nothing to do

      temp_shapes_.resize(dltensor_fp4_total_ndims_);
      size_t shape_begin = 0;

      for (size_t i = 0; i < dltensor_alloc_count_; ++i) {
        // Check if this is an FP4 tensor (4-bit float with 2 elements per byte)
        if (temp_dltensors_[i].dtype.code == kDLFloat4_e2m1fn) {
          const int64_t* orig_shape = temp_dltensors_[i].shape;
          const int64_t ndim = temp_dltensors_[i].ndim;
          std::memcpy(&temp_shapes_[shape_begin], orig_shape, sizeof(int64_t) * ndim);
          if (ndim > 0) {
            temp_shapes_[shape_begin + ndim - 1] = (orig_shape[ndim - 1] + 1) / 2;
          }
          temp_dltensors_[i].shape = temp_shapes_.data() + shape_begin;
          shape_begin += ndim;
        }
      }
    }
    /*!
     * \brief Allocate a temp owned tensor
     * \param dltensor The DLTensor to allocate the tensor from
     * \return The allocated owned tensor
     */
    TVM_FFI_INLINE tvm::ffi::Tensor AllocTempOwnedTensor(DLTensor* dltensor) {
      // note: this is really a leaky malloc, but we will detect it in DetectedLeakedTempTensors
      struct TempNDAlloc {
        DLTensor* temp;
        void AllocData(DLTensor* tensor) { tensor->data = temp->data; }
        void FreeData(DLTensor* tensor) {}
      };
      tvm::ffi::Tensor tensor = tvm::ffi::Tensor::FromNDAlloc(
          TempNDAlloc{dltensor}, tvm::ffi::ShapeView(dltensor->shape, dltensor->ndim),
          dltensor->dtype, dltensor->device);
      temp_owned_tensors_.emplace_back(tensor);
      return tensor;
    }

    /*!
     * \brief Detect leaked temp tensors
     *
     * XLA runtime do not allow Buffer to leave longer than the execution context.
     * We create temp tensors to enable python callback still being able to exchange
     * via DLPack and run some computation, however, the callback should not retain
     * the temp tensors beyond the execution context.
     *
     * This function will check the use_count of the temp tensors, if it is greater than 1,
     * it means the temp tensor is leaked.
     */
    TVM_FFI_INLINE size_t DetectedLeakedTempTensors() {
      size_t leaked_count = 0;
      for (size_t i = 0; i < temp_owned_tensors_.size(); ++i) {
        if (temp_owned_tensors_[i].use_count() != 1) {
          leaked_count++;
        }
      }
      return leaked_count;
    }

   private:
    /*!
     * \brief The workspace for the temp DLTensors
     * \note This workspace must be set in advanced and not changing during
     *       the call processing, otherwise address will be invalidated.
     */
    std::vector<DLTensor> temp_dltensors_;
    /*! \brief The workspace for the temp owned tensors */
    std::vector<tvm::ffi::Tensor> temp_owned_tensors_;
    /*!
     * \brief The workspace for the strides
     * \note This is used to fill in strides for DLTensors for compact
     *       It will be set after DLTensors are decoded.
     */
    std::vector<int64_t> temp_strides_;
    /*! \brief The workspace for transformed shapes (e.g., FP4 packing)
     * \note This is used to store transformed shapes when dtype translation is needed.
     */
    std::vector<int64_t> temp_shapes_;
    /*! \brief The number of DLTensors */
    size_t dltensor_alloc_count_ = 0;
    /*! \brief Total number of ndims needed for the DLTensors */
    size_t dltensor_total_ndims_ = 0;
    /*! \brief Total number of ndims needed for FP4 tensors only */
    size_t dltensor_fp4_total_ndims_ = 0;

    friend class CallContext;
  };
  /*! \brief stack for the call context */
  Stack* stack;
  /*! \brief Detected device, if any */
  DLDevice device;
  /*! \brief Detected stream, if any */
  void* stream = nullptr;

  CallContext() {
    stack = ThreadLocalStack();
    if (XLA_FFI_PREDICT_FALSE(stack->packed_args.size() > 0)) {
      temp_stack_ = std::make_unique<Stack>();
      stack = temp_stack_.get();
    }
  }

  // RAII exit, clear the stack
  ~CallContext() {
    stack->packed_args.clear();
    stack->temp_owned_tensors_.clear();
    stack->temp_dltensors_.clear();
    stack->temp_strides_.clear();
    stack->temp_shapes_.clear();
  }

 private:
  // temporary stack for the call context
  // only when thread local stack is already in use
  std::unique_ptr<Stack> temp_stack_;
  // by default we use thread local stack
  // to avoid repeated allocation/deallocation of stack
  Stack* ThreadLocalStack() {
    // reserve a reasonable size TLS stack
    static thread_local Stack inst;
    return &inst;
  }
};

TVM_FFI_INLINE std::optional<DLDataType> DecodeDataType(XLA_FFI_DataType dtype) {
  switch (dtype) {
    case XLA_FFI_DataType_PRED: {
      return DLDataType{kDLBool, 8, 1};
    }
    case XLA_FFI_DataType_S8: {
      return DLDataType{kDLInt, 8, 1};
    }
    case XLA_FFI_DataType_S16: {
      return DLDataType{kDLInt, 16, 1};
    }
    case XLA_FFI_DataType_S32: {
      return DLDataType{kDLInt, 32, 1};
    }
    case XLA_FFI_DataType_S64: {
      return DLDataType{kDLInt, 64, 1};
    }
    case XLA_FFI_DataType_U8: {
      return DLDataType{kDLUInt, 8, 1};
    }
    case XLA_FFI_DataType_U16: {
      return DLDataType{kDLUInt, 16, 1};
    }
    case XLA_FFI_DataType_U32: {
      return DLDataType{kDLUInt, 32, 1};
    }
    case XLA_FFI_DataType_U64: {
      return DLDataType{kDLUInt, 64, 1};
    }
    case XLA_FFI_DataType_F16: {
      return DLDataType{kDLFloat, 16, 1};
    }
    case XLA_FFI_DataType_F32: {
      return DLDataType{kDLFloat, 32, 1};
    }
    case XLA_FFI_DataType_BF16: {
      return DLDataType{kDLBfloat, 16, 1};
    }
    case XLA_FFI_DataType_F8E5M2: {
      return DLDataType{kDLFloat8_e5m2, 8, 1};
    }
    case XLA_FFI_DataType_F8E4M3: {
      return DLDataType{kDLFloat8_e4m3, 8, 1};
    }
    case XLA_FFI_DataType_F8E4M3FN: {
      return DLDataType{kDLFloat8_e4m3fn, 8, 1};
    }
    case XLA_FFI_DataType_F8E4M3B11FNUZ: {
      return DLDataType{kDLFloat8_e4m3b11fnuz, 8, 1};
    }
    case XLA_FFI_DataType_F8E5M2FNUZ: {
      return DLDataType{kDLFloat8_e5m2fnuz, 8, 1};
    }
    case XLA_FFI_DataType_F8E4M3FNUZ: {
      return DLDataType{kDLFloat8_e4m3fnuz, 8, 1};
    }
    case XLA_FFI_DataType_F8E8M0FNU: {
      return DLDataType{kDLFloat8_e8m0fnu, 8, 1};
    }
    case XLA_FFI_DataType_F4E2M1FN: {
      return DLDataType{kDLFloat4_e2m1fn, 4, 2};
    }
    default: {
      return std::nullopt;
    }
  }
}

TVM_FFI_INLINE std::optional<tvm::ffi::Any> DecodeAttrScalar(XLA_FFI_Scalar* scalar) {
  switch (scalar->dtype) {
    case XLA_FFI_DataType_PRED: {
      return tvm::ffi::Any(static_cast<bool*>(scalar->value)[0]);
    }
    case XLA_FFI_DataType_S8: {
      return tvm::ffi::Any(static_cast<int8_t*>(scalar->value)[0]);
    }
    case XLA_FFI_DataType_S16: {
      return tvm::ffi::Any(static_cast<int16_t*>(scalar->value)[0]);
    }
    case XLA_FFI_DataType_S32: {
      return tvm::ffi::Any(static_cast<int32_t*>(scalar->value)[0]);
    }
    case XLA_FFI_DataType_S64: {
      return tvm::ffi::Any(static_cast<int64_t*>(scalar->value)[0]);
    }
    case XLA_FFI_DataType_U8: {
      return tvm::ffi::Any(static_cast<uint8_t*>(scalar->value)[0]);
    }
    case XLA_FFI_DataType_U16: {
      return tvm::ffi::Any(static_cast<uint16_t*>(scalar->value)[0]);
    }
    case XLA_FFI_DataType_U32: {
      return tvm::ffi::Any(static_cast<uint32_t*>(scalar->value)[0]);
    }
    case XLA_FFI_DataType_U64: {
      return tvm::ffi::Any(static_cast<uint64_t*>(scalar->value)[0]);
    }
    case XLA_FFI_DataType_F32: {
      return tvm::ffi::Any(static_cast<float*>(scalar->value)[0]);
    }
    case XLA_FFI_DataType_F64: {
      return tvm::ffi::Any(static_cast<double*>(scalar->value)[0]);
    }
    default: {
      return std::nullopt;
    }
  }
}

TVM_FFI_INLINE std::optional<tvm::ffi::Any> DecodeAttrArray(XLA_FFI_Array* array) {
  switch (array->dtype) {
    case XLA_FFI_DataType_PRED: {
      bool* data = static_cast<bool*>(array->data);
      return tvm::ffi::Array<bool>(data, data + array->size);
    }
    case XLA_FFI_DataType_S8: {
      int8_t* data = static_cast<int8_t*>(array->data);
      return tvm::ffi::Array<int8_t>(data, data + array->size);
    }
    case XLA_FFI_DataType_S16: {
      int16_t* data = static_cast<int16_t*>(array->data);
      return tvm::ffi::Array<int16_t>(data, data + array->size);
    }
    case XLA_FFI_DataType_S32: {
      int32_t* data = static_cast<int32_t*>(array->data);
      return tvm::ffi::Array<int32_t>(data, data + array->size);
    }
    case XLA_FFI_DataType_S64: {
      int64_t* data = static_cast<int64_t*>(array->data);
      return tvm::ffi::Array<int64_t>(data, data + array->size);
    }
    case XLA_FFI_DataType_U8: {
      uint8_t* data = static_cast<uint8_t*>(array->data);
      return tvm::ffi::Array<uint8_t>(data, data + array->size);
    }
    case XLA_FFI_DataType_U16: {
      uint16_t* data = static_cast<uint16_t*>(array->data);
      return tvm::ffi::Array<uint16_t>(data, data + array->size);
    }
    case XLA_FFI_DataType_U32: {
      uint32_t* data = static_cast<uint32_t*>(array->data);
      return tvm::ffi::Array<uint32_t>(data, data + array->size);
    }
    case XLA_FFI_DataType_U64: {
      uint64_t* data = static_cast<uint64_t*>(array->data);
      return tvm::ffi::Array<uint64_t>(data, data + array->size);
    }
    case XLA_FFI_DataType_F32: {
      float* data = static_cast<float*>(array->data);
      return tvm::ffi::Array<float>(data, data + array->size);
    }
    case XLA_FFI_DataType_F64: {
      double* data = static_cast<double*>(array->data);
      return tvm::ffi::Array<double>(data, data + array->size);
    }
    default: {
      return std::nullopt;
    }
  }
}

// subclass of xla::ffi::Ffi
// so we can reuse existing helper functions defined.
// we do not use the Handler class directly since we need to operate
// on lower-level call frame arguments.
class JAXTVMFFIHandler : public xla::ffi::Ffi {
 public:
  JAXTVMFFIHandler(tvm::ffi::Function func, DecodeSpec decode_spec, int device_type, int traits,
                   bool pass_owned_tensor)
      : func_(func),
        decode_spec_(decode_spec),
        device_type_(device_type),
        traits_(traits),
        pass_owned_tensor_(pass_owned_tensor) {}

  XLA_FFI_Error* Call(XLA_FFI_CallFrame* call_frame) const final {
    // If passed a call frame with the metadata extension, just return the
    if (XLA_FFI_PREDICT_FALSE(call_frame->extension_start != nullptr &&
                              call_frame->extension_start->type == XLA_FFI_Extension_Metadata)) {
      return PopulateMetadata(call_frame->api, reinterpret_cast<XLA_FFI_Metadata_Extension*>(
                                                   call_frame->extension_start));
    }
    // create call context
    CallContext call_ctx;
    // set device type
    call_ctx.device.device_type = static_cast<DLDeviceType>(device_type_);
    // decode device ordinal
    if (XLA_FFI_Error* err = DecodeDeviceIndex(call_frame, &call_ctx);  //
        XLA_FFI_PREDICT_FALSE(err)) {
      return err;
    }
    // decode stream
    if (XLA_FFI_Error* err = DecodePlatformStream(call_frame, &call_ctx);  //
        XLA_FFI_PREDICT_FALSE(err)) {
      return err;
    }
    // reserve the maximum number of DLTensor needed to holds all args and returns
    // we will not change the size of temp_dltensors_ during the call processing
    call_ctx.stack->ReserveTempDLTensors(call_frame->args.size, call_frame->rets.size);
    // populate the call stack
    for (const auto& item : decode_spec_.items) {
      switch (item.kind) {
        case DecodeKind::kArgs: {
          if (XLA_FFI_Error* err = DecodeArgs(call_frame, &call_ctx);
              XLA_FFI_PREDICT_FALSE(err)) {  //
            return err;
          }
          break;
        }
        case DecodeKind::kRets: {
          if (XLA_FFI_Error* err = DecodeRets(call_frame, &call_ctx);  //
              XLA_FFI_PREDICT_FALSE(err)) {
            return err;
          }
          break;
        }
        case DecodeKind::kAttrValue: {
          if (XLA_FFI_Error* err = DecodeAttr(call_frame, &call_ctx, item);  //
              XLA_FFI_PREDICT_FALSE(err)) {
            return err;
          }
          break;
        }
        case DecodeKind::kContextStream: {
          if (XLA_FFI_Error* err = DecodeContextStream(&call_ctx);  //
              XLA_FFI_PREDICT_FALSE(err)) {
            return err;
          }
          break;
        }
        default: {
          return MakeError(call_frame->api, XLA_FFI_Error_Code_INTERNAL, "Invalid decode kind");
        }
      }
    }
    // fill in shapes for FP4 tensors
    call_ctx.stack->FillShapesForFP4Tensors();
    // fill in strides for the temp DLTensors
    call_ctx.stack->FillStridesForTempDLTensors();
    // now run the invocation
    // use C safe call so that we don't have to catch an exception
    void* prev_stream = nullptr;
    if (call_ctx.stream != nullptr) {
      int ret_code = TVMFFIEnvSetStream(call_ctx.device.device_type, call_ctx.device.device_id,
                                        call_ctx.stream, &prev_stream);
      if (XLA_FFI_PREDICT_FALSE(ret_code != 0)) {
        return MoveFromSafeCallRaisedToXLAError(call_frame, XLA_FFI_Error_Code_INTERNAL);
      }
    }
    // run the call
    tvm::ffi::Any result;
    const tvm::ffi::FunctionObj* func_obj = static_cast<const tvm::ffi::FunctionObj*>(func_.get());
    int call_code =
        func_obj->safe_call(const_cast<tvm::ffi::FunctionObj*>(func_obj),
                            reinterpret_cast<const TVMFFIAny*>(call_ctx.stack->packed_args.data()),
                            static_cast<int32_t>(call_ctx.stack->packed_args.size()),
                            reinterpret_cast<TVMFFIAny*>(&result));
    // restore the stream regardless of the result of the call
    if (prev_stream != nullptr && prev_stream != call_ctx.stream) {
      int ret_code = TVMFFIEnvSetStream(call_ctx.device.device_type, call_ctx.device.device_id,
                                        prev_stream, nullptr);
      if (XLA_FFI_PREDICT_FALSE(ret_code != 0)) {
        return MoveFromSafeCallRaisedToXLAError(call_frame, XLA_FFI_Error_Code_INTERNAL);
      }
    }
    if (XLA_FFI_PREDICT_FALSE(call_code != 0)) {
      return MoveFromSafeCallRaisedToXLAError(call_frame, XLA_FFI_Error_Code_INTERNAL);
    }
    if (pass_owned_tensor_ &&
        XLA_FFI_PREDICT_FALSE(call_ctx.stack->DetectedLeakedTempTensors() != 0)) {
      return MakeError(call_frame->api, XLA_FFI_Error_Code_INTERNAL,
                       "Leaked temp owned tensors, cannot retain ffi::Tensor in the function");
    }
    return Success();
  }

 private:
  XLA_FFI_Error* MoveFromSafeCallRaisedToXLAError(const XLA_FFI_CallFrame* call_frame,
                                                  XLA_FFI_Error_Code code) const {
    tvm::ffi::Error err = tvm::ffi::details::MoveFromSafeCallRaised();
    return MakeError(call_frame->api, code, err.what());
  }

  TVM_FFI_INLINE XLA_FFI_Error* DecodeDeviceIndex(const XLA_FFI_CallFrame* call_frame,
                                                  CallContext* call_ctx) const {
    if (device_type_ == kDLCPU) {
      call_ctx->device.device_id = 0;
      return Success();
    }
    XLA_FFI_DeviceOrdinal_Get_Args args;
    args.struct_size = XLA_FFI_DeviceOrdinal_Get_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.ctx = call_frame->ctx;
    args.device_ordinal = 0;
    if (XLA_FFI_Error* err = call_frame->api->XLA_FFI_DeviceOrdinal_Get(&args);  //
        XLA_FFI_PREDICT_FALSE(err)) {
      std::ostringstream msg;
      msg << "Failed to get device ordinal: "
          << xla::ffi::internal::GetErrorMessage(call_frame->api, err);
      return MakeError(call_frame->api, XLA_FFI_Error_Code_INTERNAL, msg.str());
    }
    call_ctx->device.device_id = args.device_ordinal;
    return Success();
  }

  TVM_FFI_INLINE XLA_FFI_Error* DecodePlatformStream(const XLA_FFI_CallFrame* call_frame,
                                                     CallContext* call_ctx) const {
    if (device_type_ == kDLCPU) {
      call_ctx->stream = nullptr;
      return Success();
    }
    XLA_FFI_Stream_Get_Args args;
    args.struct_size = XLA_FFI_Stream_Get_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.ctx = call_frame->ctx;
    args.stream = nullptr;

    if (XLA_FFI_Error* error = call_frame->api->XLA_FFI_Stream_Get(&args);  //
        XLA_FFI_PREDICT_FALSE(error)) {
      std::ostringstream msg;
      msg << "Failed to get platform stream: "
          << xla::ffi::internal::GetErrorMessage(call_frame->api, error);
      return MakeError(call_frame->api, XLA_FFI_Error_Code_INTERNAL, msg.str());
    }
    call_ctx->stream = args.stream;
    return Success();
  }

  TVM_FFI_INLINE XLA_FFI_Error* DecodeBuffer(const XLA_FFI_CallFrame* call_frame,
                                             CallContext* call_ctx, XLA_FFI_Buffer* buffer,
                                             DLTensor** target) const {
    DLTensor* dltensor = call_ctx->stack->AllocTempDLTensor(
        buffer->rank, (buffer->dtype == XLA_FFI_DataType_F4E2M1FN));
    dltensor->data = buffer->data;
    dltensor->device = call_ctx->device;
    if (auto dtype = DecodeDataType(buffer->dtype); XLA_FFI_PREDICT_TRUE(dtype)) {
      dltensor->dtype = *dtype;
    } else {
      std::ostringstream msg;
      msg << "Unsupported XLA data type " << buffer->dtype;
      return InvalidArgument(call_frame->api, msg.str());
    }
    dltensor->ndim = buffer->rank;
    dltensor->shape = buffer->dims;
    dltensor->strides = nullptr;
    dltensor->byte_offset = 0;
    *target = dltensor;
    return Success();
  }

  XLA_FFI_Error* DecodeArgs(const XLA_FFI_CallFrame* call_frame, CallContext* call_ctx) const {
    for (int64_t i = 0; i < call_frame->args.size; ++i) {
      // currently only buffer is supported
      if (XLA_FFI_PREDICT_FALSE(call_frame->args.types[i] != XLA_FFI_ArgType_BUFFER)) {
        return InvalidArgument(call_frame->api, "Only support AnyBuffer argument type");
      }
      XLA_FFI_Buffer* buffer = static_cast<XLA_FFI_Buffer*>(call_frame->args.args[i]);
      DLTensor* dltensor;
      if (XLA_FFI_Error* err = DecodeBuffer(call_frame, call_ctx, buffer, &dltensor);  //
          XLA_FFI_PREDICT_FALSE(err)) {
        return err;
      }
      if (!pass_owned_tensor_) {
        call_ctx->stack->packed_args.emplace_back(dltensor);
      } else {
        call_ctx->stack->packed_args.emplace_back(call_ctx->stack->AllocTempOwnedTensor(dltensor));
      }
    }
    return Success();
  }

  XLA_FFI_Error* DecodeRets(const XLA_FFI_CallFrame* call_frame, CallContext* call_ctx) const {
    for (int64_t i = 0; i < call_frame->rets.size; ++i) {
      if (XLA_FFI_PREDICT_FALSE(call_frame->rets.types[i] != XLA_FFI_RetType_BUFFER)) {
        return InvalidArgument(call_frame->api, "Only support AnyBuffer return type");
      }
      XLA_FFI_Buffer* buffer = static_cast<XLA_FFI_Buffer*>(call_frame->rets.rets[i]);
      DLTensor* dltensor;
      if (XLA_FFI_Error* err = DecodeBuffer(call_frame, call_ctx, buffer, &dltensor);  //
          XLA_FFI_PREDICT_FALSE(err)) {
        return err;
      }
      if (!pass_owned_tensor_) {
        call_ctx->stack->packed_args.emplace_back(dltensor);
      } else {
        call_ctx->stack->packed_args.emplace_back(call_ctx->stack->AllocTempOwnedTensor(dltensor));
      }
    }
    return Success();
  }

  XLA_FFI_Error* DecodeContextStream(CallContext* call_ctx) const {
    // Convert void* stream pointer to int64_t and add to packed args.
    // This allows functions to explicitly receive the CUDA/platform stream if they include
    // "ctx.stream" in their arg_spec.
    int64_t stream_as_int64 = reinterpret_cast<int64_t>(call_ctx->stream);
    call_ctx->stack->packed_args.emplace_back(stream_as_int64);
    return Success();
  }

  TVM_FFI_INLINE XLA_FFI_Error* DecodeAttrValue(const XLA_FFI_CallFrame* call_frame,
                                                CallContext* call_ctx, XLA_FFI_AttrType attr_type,
                                                void* attr_value, tvm::ffi::Any* result) const {
    switch (attr_type) {
      case XLA_FFI_AttrType_SCALAR: {
        XLA_FFI_Scalar* scalar = static_cast<XLA_FFI_Scalar*>(attr_value);
        if (auto opt_res = DecodeAttrScalar(scalar); XLA_FFI_PREDICT_TRUE(opt_res)) {
          // keep as temp attribute
          call_ctx->stack->temp_attrs.emplace_back(*opt_res);
          *result = *std::move(opt_res);
        } else {
          std::ostringstream msg;
          msg << "Unsupported scalar attribute dtype: " << scalar->dtype;
          return InvalidArgument(call_frame->api, msg.str());
        }
        return Success();
      }
      case XLA_FFI_AttrType_STRING: {
        XLA_FFI_ByteSpan* str = static_cast<XLA_FFI_ByteSpan*>(attr_value);
        *result = tvm::ffi::String(str->ptr, str->len);
        return Success();
      }
      case XLA_FFI_AttrType_ARRAY: {
        XLA_FFI_Array* array = static_cast<XLA_FFI_Array*>(attr_value);
        if (auto opt_res = DecodeAttrArray(array); XLA_FFI_PREDICT_TRUE(opt_res)) {
          // keep as temp attribute for liveness
          call_ctx->stack->temp_attrs.emplace_back(*opt_res);
          *result = *std::move(opt_res);
        } else {
          std::ostringstream msg;
          msg << "Unsupported array attribute dtype: " << array->dtype;
          return InvalidArgument(call_frame->api, msg.str());
        }
        return Success();
      }
      default: {
        // TODO(jax-tvm-ffi team): consider support array attirbute type
        std::ostringstream msg;
        msg << "Unsupported attribute type: " << attr_type;
        return InvalidArgument(call_frame->api, msg.str());
      }
    }
  }

  TVM_FFI_INLINE XLA_FFI_Error* DecodeAttr(const XLA_FFI_CallFrame* call_frame,
                                           CallContext* call_ctx,
                                           const DecodeItem& decode_item) const {
    // get the attribute name
    XLA_FFI_AttrType attr_type = call_frame->attrs.types[decode_item.sorted_attr_index];
    XLA_FFI_ByteSpan* attr_name = call_frame->attrs.names[decode_item.sorted_attr_index];
    void* attr_value = call_frame->attrs.attrs[decode_item.sorted_attr_index];
    // inline string comparison
    auto str_eq = [](const XLA_FFI_ByteSpan* a, const std::string& b) {
      if (XLA_FFI_PREDICT_FALSE(a->len != b.size())) {
        return false;
      }
      for (size_t i = 0; i < a->len; ++i) {
        if (XLA_FFI_PREDICT_FALSE(a->ptr[i] != b[i])) {
          return false;
        }
      }
      return true;
    };
    const std::string& expected_attr_name =
        decode_spec_.sorted_attr_keys[decode_item.sorted_attr_index].first;
    if (XLA_FFI_PREDICT_FALSE(!str_eq(attr_name, expected_attr_name))) {
      std::ostringstream msg;
      msg << "Attribute mismatch mismatch, "
          << "expected `" << expected_attr_name << "`, actual `"
          << std::string_view(attr_name->ptr, attr_name->len) << "`";
      return InvalidArgument(call_frame->api, msg.str());
    }
    tvm::ffi::Any attr_value_any;
    if (XLA_FFI_Error* err =
            DecodeAttrValue(call_frame, call_ctx, attr_type, attr_value, &attr_value_any);  //
        XLA_FFI_PREDICT_FALSE(err)) {
      return err;
    }
    call_ctx->stack->packed_args.emplace_back(std::move(attr_value_any));
    return Success();
  }

  // populate metadata extension to the call frame
  XLA_FFI_Error* PopulateMetadata(const XLA_FFI_Api* api,
                                  XLA_FFI_Metadata_Extension* extension) const {
    if (XLA_FFI_Error* err = StructSizeIsGreaterOrEqual(api, "XLA_FFI_Metadata_Extension",
                                                        XLA_FFI_Metadata_Extension_STRUCT_SIZE,
                                                        extension->extension_base.struct_size)) {
      return err;
    }

    if (XLA_FFI_Error* err =
            StructSizeIsGreaterOrEqual(api, "XLA_FFI_Metadata", XLA_FFI_Metadata_STRUCT_SIZE,
                                       extension->metadata->struct_size)) {
      return err;
    }

    extension->metadata->api_version = XLA_FFI_Api_Version{
        XLA_FFI_Api_Version_STRUCT_SIZE,
        /*extension_start=*/nullptr,
        XLA_FFI_API_MAJOR,
        XLA_FFI_API_MINOR,
    };
    extension->metadata->traits = static_cast<XLA_FFI_Handler_Traits>(traits_);
    return Success();
  }
  // upstream have a typo with the name Sucess
  static XLA_FFI_Error* Success() { return nullptr; }
  // underlying function
  tvm::ffi::Function func_;
  DecodeSpec decode_spec_;
  int device_type_;
  int traits_;
  bool pass_owned_tensor_;
};

//-------------------------------------------------------------------
// global registry of handlers
class JAXTVMFFIRegistry {
 public:
  static void* Register(tvm::ffi::Function func, tvm::ffi::Array<tvm::ffi::String> arg_spec,
                        int device_type, int traits, bool pass_owned_tensor) {
    return Global()->RegisterInternal(func, arg_spec, device_type, traits, pass_owned_tensor);
  }

  static size_t RegisteredCount() { return Global()->registered_count_; }

 private:
  // size of the trampoline table
  static constexpr int kTrampolineTableSize = 1024;
  // current number of handlers allocated
  size_t registered_count_ = 0;
  // handler table to dispatch to
  std::array<std::unique_ptr<JAXTVMFFIHandler>, kTrampolineTableSize> handler_table_;
  // global static trampoline table pre-populated
  std::array<XLA_FFI_Handler*, kTrampolineTableSize> trampoline_table_ =
      MakeTrampolineTable(std::make_index_sequence<kTrampolineTableSize>{});

  // global instance
  static JAXTVMFFIRegistry* Global() {
    static JAXTVMFFIRegistry* inst = new JAXTVMFFIRegistry();
    return inst;
  }

  // internal register function
  void* RegisterInternal(tvm::ffi::Function func, tvm::ffi::Array<tvm::ffi::String> arg_spec,
                         int device_type, int traits, bool pass_owned_tensor) {
    if (registered_count_ >= kTrampolineTableSize) {
      TVM_FFI_THROW(RuntimeError) << "JAXTVMFFIRegistry: cannot register more than "
                                  << kTrampolineTableSize << " handlers";
    }
    handler_table_[registered_count_++] = std::make_unique<JAXTVMFFIHandler>(
        func, ParseArgSpec(arg_spec), device_type, traits, pass_owned_tensor);
    return reinterpret_cast<void*>(trampoline_table_[registered_count_ - 1]);
  }

  // must not inline the entry to minimize the trampoline function size
  TVM_FFI_NO_INLINE static XLA_FFI_Error* Entry(int index, XLA_FFI_CallFrame* call_frame) {
    return Global()->handler_table_[index]->Call(call_frame);
  }
  //-------------------------------------------------------------------
  // Trampoline table trick:
  // trampoline functions that dispatches based on index
  //
  // We use this design to work around the limitation that xla ffi handler right now
  // can only take in raw function pointer without extra user data
  //
  // Each function pointer is a trampoline that dispatches based on index
  // so we can dispatch to a tvm::ffi::Function closure based on the index.
  template <int index>
  static XLA_FFI_Error* Trampoline(XLA_FFI_CallFrame* call_frame) {
    return JAXTVMFFIRegistry::Entry(index, call_frame);
  }
  // helper to geenrate the trampoline table
  template <size_t... Indices>
  static std::array<XLA_FFI_Handler*, sizeof...(Indices)> MakeTrampolineTable(
      std::index_sequence<Indices...>) {
    // This uses a parameter pack expansion to create an array of function pointers,
    // one for each index in the sequence.
    return std::array<XLA_FFI_Handler*, sizeof...(Indices)>{&Trampoline<Indices>...};
  }
};

// callback to register a handler to the global registry
TVM_FFI_DLL_EXPORT_TYPED_FUNC(register_tvm_ffi_handler, JAXTVMFFIRegistry::Register);
// get the number of handlers registered
TVM_FFI_DLL_EXPORT_TYPED_FUNC(registered_count, JAXTVMFFIRegistry::RegisteredCount);

}  // namespace jax_tvm_ffi
