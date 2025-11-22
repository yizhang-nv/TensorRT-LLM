#!/bin/bash

# Build script for compiling kv_cache_manager_v2 with mypyc
# Run this script from the kv_cache_manager_v2 directory or from the project root

# Determine script directory and runtime directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "========================================"
echo "Building kv_cache_manager_v2 with mypyc"
echo "========================================"
echo "Runtime directory: $RUNTIME_DIR"
echo ""

# Change to runtime directory for compilation (to make module standalone)
cd "$RUNTIME_DIR" || exit 1

# Clean previous builds
echo "Step 1: Cleaning previous builds..."
rm -rf build/
find kv_cache_manager_v2 -name "*.so" -delete 2>/dev/null
find kv_cache_manager_v2 -name "*.c" -delete 2>/dev/null
echo "  ✓ Cleaned"
echo ""

# Hide only the files that won't be compiled (to avoid import errors during type checking)
echo "Step 2: Hiding excluded files temporarily..."
TEMP_DIR=$(mktemp -d)
mkdir -p "$TEMP_DIR/kv_cache_manager_v2"
EXCLUDED_KV_FILES=(
    "kv_cache_manager_v2/_copy_engine.py"
    "kv_cache_manager_v2/_exceptions.py"
)
HIDDEN_COUNT=0
for excluded_file in "${EXCLUDED_KV_FILES[@]}"; do
    if [ -f "$excluded_file" ]; then
        mv "$excluded_file" "$TEMP_DIR/$excluded_file" && ((HIDDEN_COUNT++))
    fi
done

echo "  ✓ Temporarily hid $HIDDEN_COUNT files from compilation"
echo "  ℹ Note: Only _copy_engine.py and _exceptions.py are excluded from compilation"
echo ""

# Build with mypyc
echo "Step 3: Compiling with mypyc..."
echo ""

# Run compilation from the runtime directory
python "$SCRIPT_DIR/setup_mypyc.py" build_ext --inplace 2>&1 | \
    tee /tmp/mypyc_build_full.log | \
    grep -E "(^Compiling |^running |^building |^gcc|^clang)" || true

BUILD_EXIT_CODE=${PIPESTATUS[0]}
echo ""

# Restore all hidden files
echo "Step 4: Restoring hidden files..."

# Restore excluded kv_cache_manager_v2 files
for excluded_file in _copy_engine.py _exceptions.py; do
    if [ -f "$TEMP_DIR/kv_cache_manager_v2/$excluded_file" ]; then
        mv "$TEMP_DIR/kv_cache_manager_v2/$excluded_file" kv_cache_manager_v2/ 2>/dev/null
    fi
done

echo "  ✓ Restored all hidden files"

# Cleanup
rm -rf "$TEMP_DIR"
echo ""

# Check results
echo "========================================"
echo "Build Results"
echo "========================================"
echo ""

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

    # Check for errors in kv_cache_manager_v2
    if grep -q "kv_cache_manager_v2.*error:" /tmp/mypyc_build_full.log 2>/dev/null; then
        echo "Errors in kv_cache_manager_v2 files:"
        grep "kv_cache_manager_v2.*error:" /tmp/mypyc_build_full.log
    else
        echo "No errors found in kv_cache_manager_v2 files."
        echo "This might be a compilation issue. Check: /tmp/mypyc_build_full.log"
    fi
    echo ""
    echo "Full build log saved to: /tmp/mypyc_build_full.log"
    exit 1
fi
