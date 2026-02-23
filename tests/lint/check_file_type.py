# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helper tool to check file types that are allowed to checkin."""

import subprocess
import sys
from pathlib import Path

# List of file types we allow
ALLOW_EXTENSION = {
    # source code
    "cc",
    "c",
    "h",
    "s",
    "rs",
    "m",
    "mm",
    "g4",
    "gradle",
    "js",
    "cjs",
    "mjs",
    "tcl",
    "scala",
    "java",
    "go",
    "ts",
    "sh",
    "py",
    "pyi",
    "pxi",
    "pyd",
    "pyx",
    "cu",
    "cuh",
    "bat",
    # configurations
    "mk",
    "in",
    "cmake",
    "xml",
    "toml",
    "yml",
    "yaml",
    "json",
    "cfg",
    # docs
    "txt",
    "md",
    "rst",
    "css",
    # sgx
    "edl",
    "lds",
    # ios
    "pbxproj",
    "plist",
    "xcworkspacedata",
    "storyboard",
    "xcscheme",
    # hw/chisel
    "sbt",
    "properties",
    "v",
    "sdc",
    # generated parser
    "interp",
    "tokens",
    # interface definition
    "idl",
    # opencl file
    "cl",
    # zephyr config file
    "conf",
    # arduino sketch file
    "ino",
    # linker scripts
    "ld",
    # Jinja2 templates
    "j2",
    # Jenkinsfiles
    "groovy",
    # Python-parseable config files
    "ini",
}

# List of file names allowed
ALLOW_FILE_NAME = {
    ".gitignore",
    ".eslintignore",
    ".gitattributes",
    "README",
    "Makefile",
    "Doxyfile",
    "pylintrc",
    ".clang-format",
    ".gitmodules",
    "CODEOWNERSHIP",
    "Dockerfile",
    "py.typed",
}

# List of specific files allowed in relpath to <proj_root>
ALLOW_SPECIFIC_FILE = {"LICENSE", "NOTICE", "KEYS", "DISCLAIMER"}


def filename_allowed(name: str) -> bool:
    """Check if name is allowed by the current policy.

    Paramaters
    ----------
    name : str
        Input name

    Returns
    -------
    allowed : bool
        Whether the filename is allowed.

    """
    arr = name.rsplit(".", 1)
    if arr[-1] in ALLOW_EXTENSION:
        return True

    if Path(name).name in ALLOW_FILE_NAME:
        return True

    if name.startswith("3rdparty"):
        return True

    if name in ALLOW_SPECIFIC_FILE:
        return True

    return False


def copyright_line(line: str) -> bool:
    # SPDX headers include copyright, so exclude them from this check
    if line.find("SPDX-FileCopyrightText") != -1:
        return False
    # Following two items are intentionally break apart
    # so that the copyright detector won't detect the file itself.
    if line.find("Copyright " + "(c)") != -1:
        return True
    # break pattern into two lines to avoid false-negative check
    spattern1 = "Copyright"
    if line.find(spattern1) != -1 and line.find("by") != -1:
        return True
    return False


def check_spdx_header(fname: str) -> tuple[bool, str]:
    """Check if file has proper SPDX header.

    Returns:
        (bool, str): (passes_check, error_message)
    """
    # Skip binary files, directories, and special files
    skip_files = {
        ".gitignore",
        ".gitattributes",
        ".clang-format",
        ".clang-tidy",
        "LICENSE",
        "NOTICE",
        "README.md",
        "CONTRIBUTING.md",
    }
    if (
        fname.endswith((".png", ".whl", ".txt"))
        or not Path(fname).is_file()
        or Path(fname).name in skip_files
    ):
        return (True, "")

    has_spdx_copyright = False
    has_spdx_license = False
    has_separate_copyright = False

    try:
        for line in Path(fname).open():
            if line.find("SPDX-FileCopyrightText") != -1:
                has_spdx_copyright = True
            if line.find("SPDX-License-Identifier") != -1:
                has_spdx_license = True
            if copyright_line(line):
                has_separate_copyright = True
    except UnicodeDecodeError:
        # Skip binary files
        return (True, "")

    # Check for issues
    if not has_spdx_copyright or not has_spdx_license:
        return (False, "missing SPDX header")
    if has_separate_copyright:
        return (False, "has SPDX header AND separate copyright line")

    return (True, "")


def main() -> None:
    cmd = ["git", "ls-files"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out, _) = proc.communicate()
    res = out.decode("utf-8")
    assert proc.returncode == 0, f"{' '.join(cmd)} errored: {res}"
    flist = res.split()
    error_list = []

    for fname in flist:
        if not filename_allowed(fname):
            error_list.append(fname)

    if error_list:
        report = "------File type check report----\n"
        report += "\n".join(error_list)
        report += f"\nFound {len(error_list)} files that are not allowed\n"
        report += (
            "We do not check in binary files into the repo.\n"
            "If necessary, please discuss with committers and"
            "modify tests/lint/check_file_type.py to enable the file you need.\n"
        )
        sys.stderr.write(report)
        sys.stderr.flush()
        sys.exit(-1)

    spdx_issues = {}

    for fname in res.split():
        passes, error_msg = check_spdx_header(fname)
        if not passes:
            spdx_issues[fname] = error_msg

    if spdx_issues:
        report = "------SPDX Header Check Report----\n"
        report += f"Found {len(spdx_issues)} files with header issues:\n\n"
        for fname, error in spdx_issues.items():
            report += f"  {fname}: {error}\n"
        report += "\n"
        report += "--- All source files must have SPDX headers:\n"
        report += "---   # SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n"
        report += "---   # SPDX-License-Identifier: Apache-2.0\n"
        report += "---\n"
        report += "--- Files with SPDX headers should not have separate copyright lines.\n"
        report += "--- SPDX-FileCopyrightText already includes copyright information.\n"
        sys.stderr.write(report)
        sys.stderr.flush()
        sys.exit(-1)

    print("check_file_type.py: all checks passed..")


if __name__ == "__main__":
    main()
