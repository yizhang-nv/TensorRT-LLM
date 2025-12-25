"""Setup script for compiling kv_cache_manager_v2 with mypyc.

Usage (from project root):
    python tensorrt_llm/runtime/kv_cache_manager_v2/setup_mypyc.py build_ext --inplace

Or use the build script:
    ./tensorrt_llm/runtime/kv_cache_manager_v2/build_mypyc.sh
"""

import os
import sys

from mypyc.build import mypycify
from setuptools import setup

# Set environment variables BEFORE importing mypyc
os.environ["MYPY_FORCE_COLOR"] = "0"

# Write a strict mypy config that won't check external files
mypy_config_path = os.path.abspath("mypy_mypyc_build.ini")
with open(mypy_config_path, "w") as f:
    f.write("""[mypy]
# Critical: Don't follow any imports outside the specified files
follow_imports = skip
follow_imports_for_stubs = False

# Ignore missing imports completely
ignore_missing_imports = True

# Allow all untyped code
allow_untyped_calls = True
allow_untyped_defs = True
allow_incomplete_defs = True
allow_untyped_globals = True
check_untyped_defs = False

# Disable all warnings that might cause errors
disallow_untyped_calls = False
disallow_untyped_defs = False
disallow_incomplete_defs = False
warn_return_any = False
warn_unused_ignores = False

# Disable type validation errors (for external types like drv.CUstream)
disable_error_code = valid-type
""")

# Point mypy to this config by adding to sys.argv before mypyc runs
sys.argv.extend(["--config-file", mypy_config_path])

# List all Python modules in sampling_helper_func to compile
modules = [
    # Main module files
    "sampling_helper_func/__init__.py",
    "sampling_helper_func/funcs.py",
]

print(f"Compiling {len(modules)} modules with mypyc...")
print("Excluded: None")
print("")

try:
    ext_modules = mypycify(
        modules,
        opt_level="3",  # Maximum optimization
        multi_file=True,  # Allow cross-module references (needed for inheritance)
        verbose=True,  # Show what's being compiled
        separate=False,  # Compile into single .so (required for cross-module inheritance)
        strip_asserts=False,  # Keep assertions for debugging
    )

except Exception as e:
    print(f"Error during mypyc compilation: {e}")
    sys.exit(1)
finally:
    # Cleanup temp config
    if os.path.exists(mypy_config_path):
        try:
            os.remove(mypy_config_path)
        except OSError:
            pass

    # Remove --config-file arguments from sys.argv before calling setup()
    # This prevents setuptools from seeing arguments it doesn't understand
    while "--config-file" in sys.argv:
        idx = sys.argv.index("--config-file")
        sys.argv.pop(idx)  # Remove '--config-file'
        if idx < len(sys.argv):  # Remove the path that follows it
            sys.argv.pop(idx)

setup(
    name="sampling_helper_func_compiled",
    ext_modules=ext_modules,
    packages=["sampling_helper_func"],
    python_requires=">=3.8",
)
