#!/bin/bash

# Build script for compiling kv_cache_manager_v2 with mypyc
# Run this script from the kv_cache_manager_v2 directory or from the project root
#
# Usage:
#   ./build_mypyc.sh         Build the module
#   ./build_mypyc.sh clean   Clean mypyc-generated files (for debugging with .py files)

# Determine script directory and runtime directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to runtime directory
cd "$RUNTIME_DIR" || exit 1

# Clean function
do_clean() {
    echo "Cleaning mypyc-generated files..."
    rm -rf build/
    find kv_cache_manager_v2 -name "*.so" -type f ! -path "*/rawref/*" -print -delete 2>/dev/null
    echo "✓ Cleaned. You can now run the original .py files for debugging."
}

# Handle clean option
if [ "$1" = "clean" ]; then
    do_clean
    exit 0
fi

echo "========================================"
echo "Building kv_cache_manager_v2 with mypyc"
echo "========================================"
echo "Runtime directory: $RUNTIME_DIR"
echo ""

# Clean previous mypyc builds (exclude rawref/ which has hand-written C extension)
echo "Step 1: Cleaning previous mypyc builds..."
do_clean
echo ""

# Build with mypyc
echo "Step 2: Compiling with mypyc..."
echo ""

python "$SCRIPT_DIR/setup_mypyc.py" build_ext --inplace 2>&1 | \
    tee /tmp/mypyc_build_full.log | \
    grep -E "(^Compiling |^running |^building |^gcc|^clang)" || true

BUILD_EXIT_CODE=${PIPESTATUS[0]}
echo ""

# Check results
echo "========================================"
echo "Build Results"
echo "========================================"
echo ""

# Check for errors first
if grep -E ".+\.py:[0-9]+: error:" /tmp/mypyc_build_full.log >/dev/null 2>&1; then
    echo "✗ BUILD FAILED (mypyc errors detected)"
    echo ""
    echo "Errors in source files:"
    grep -E ".+\.py:[0-9]+: error:" /tmp/mypyc_build_full.log
    echo ""
    echo "Full build log saved to: /tmp/mypyc_build_full.log"
    exit 1
fi

SO_FILES=$(find kv_cache_manager_v2 -name "*.so" -type f 2>/dev/null)

if [ -n "$SO_FILES" ]; then
    echo "✓ SUCCESS! Compiled modules:"
    echo ""
    echo "$SO_FILES" | while read -r file; do
        size=$(ls -lh "$file" | awk '{print $5}')
        basename=$(basename "$file")
        echo "  • $basename ($size)"
    done
    echo ""
    num_files=$(echo "$SO_FILES" | wc -l)
    echo "Total: $num_files compiled modules"
    echo ""
    echo "The compiled extensions will be automatically used when you import the module."
    echo "Set PYTHONPATH to include $(pwd) to use the module standalone."
    echo ""
    exit 0
else
    echo "✗ BUILD FAILED"
    echo ""
    echo "No compiled modules found and no mypyc errors detected."
    echo "This might be a compilation issue. Check: /tmp/mypyc_build_full.log"
    echo ""
    echo "Full build log saved to: /tmp/mypyc_build_full.log"
    exit 1
fi
