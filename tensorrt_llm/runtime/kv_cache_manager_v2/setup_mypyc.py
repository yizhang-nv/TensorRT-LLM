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

# List all Python modules in kv_cache_manager_v2 to compile
#
# EXCLUDED FILES:
# - _exceptions.py: inherits from builtin Exception classes (mypyc limitation)
#
modules = [
    # Main module files
    "kv_cache_manager_v2/__init__.py",
    "kv_cache_manager_v2/_block_radix_tree.py",
    "kv_cache_manager_v2/_common.py",
    "kv_cache_manager_v2/_config.py",
    "kv_cache_manager_v2/_copy_engine.py",
    "kv_cache_manager_v2/_cuda_virt_mem.py",
    "kv_cache_manager_v2/_exceptions.py",
    "kv_cache_manager_v2/_life_cycle_registry.py",
    "kv_cache_manager_v2/_page.py",
    "kv_cache_manager_v2/_storage_manager.py",
    "kv_cache_manager_v2/_utils.py",
    # _core submodule
    "kv_cache_manager_v2/_core/__init__.py",
    "kv_cache_manager_v2/_core/_kv_cache_manager.py",
    "kv_cache_manager_v2/_core/_kv_cache.py",
    # _eviction_controller submodule
    "kv_cache_manager_v2/_eviction_controller/__init__.py",
    "kv_cache_manager_v2/_eviction_controller/_eviction_controller.py",
    # _storage submodule
    "kv_cache_manager_v2/_storage/__init__.py",
    "kv_cache_manager_v2/_storage/_config.py",
    "kv_cache_manager_v2/_storage/_core.py",
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

    # Generate stub files using stubgen
    print("Generating stub files...")
    import subprocess

    # We are inside kv_cache_manager_v2 directory when running this script,
    # but stubgen needs to see kv_cache_manager_v2 as a package from its parent directory.
    # The modules list above uses relative paths like "kv_cache_manager_v2/__init__.py".
    #
    # When running from tensorrt_llm/runtime (as the Makefile does), current dir is runtime.
    # So "kv_cache_manager_v2" is a subdirectory.

    # Convert file paths to module names for stubgen
    # We list modules explicitly to avoid recursing into rawref (which has manual stubs)
    stub_modules = []
    for m in modules:
        # Remove .py extension
        if m.endswith(".py"):
            mod = m[:-3]
        else:
            mod = m

        # Replace file separators with dots
        mod = mod.replace("/", ".")

        # Remove .__init__ suffix to get the package name
        if mod.endswith(".__init__"):
            mod = mod[:-9]

        stub_modules.append(mod)

    # We use --ignore-errors to avoid failing on imports
    cmd = ["stubgen"]
    for m in stub_modules:
        cmd.extend(["-m", m])
    cmd.extend(["--include-private", "-o", ".", "--ignore-errors"])

    # Ensure we run stubgen from the parent of kv_cache_manager_v2 package
    # If the script is run from inside kv_cache_manager_v2, we need to go up one level.
    cwd = os.getcwd()
    if os.path.basename(cwd) == "kv_cache_manager_v2":
        cwd = os.path.dirname(cwd)

    # Set PYTHONPATH to include the parent directory so kv_cache_manager_v2 can be imported
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    if pythonpath:
        pythonpath += os.pathsep + cwd
    else:
        pythonpath = cwd
    env["PYTHONPATH"] = pythonpath

    subprocess.check_call(cmd, cwd=cwd, env=env)

except Exception as e:
    print(f"Error during mypyc compilation: {e}")
    ext_modules = []
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
    name="kv_cache_manager_v2_compiled",
    ext_modules=ext_modules,
    packages=["kv_cache_manager_v2.rawref"],
    package_data={
        "kv_cache_manager_v2": ["*.pyi", "**/*.pyi"],
    },
    python_requires=">=3.8",
)
