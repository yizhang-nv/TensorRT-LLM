#!/bin/bash

# Script to clean all mypyc-compiled binaries
# This preserves the rawref C extension source and binary

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

echo "========================================"
echo "Cleaning mypyc-compiled binaries"
echo "========================================"
echo ""

# Clean the build directory
echo "1. Removing build/ directory..."
if [ -d "build/" ]; then
    rm -rf build/
    echo "  ✓ Removed build/"
else
    echo "  ℹ No build/ directory found"
fi
echo ""

# Clean mypyc-generated .so files (but preserve rawref C extension)
echo "2. Removing mypyc-compiled .so files..."
SO_COUNT=0

# Find all .so files, excluding the rawref directory
while IFS= read -r sofile; do
    # Skip if it's in the rawref directory
    if [[ "$sofile" == *"/rawref/"* ]]; then
        continue
    fi
    rm -f "$sofile"
    echo "  - Deleted: $(basename "$sofile")"
    ((SO_COUNT++))
done < <(find . -name "*.so" -type f 2>/dev/null)

if [ $SO_COUNT -eq 0 ]; then
    echo "  ℹ No mypyc .so files found"
else
    echo "  ✓ Removed $SO_COUNT mypyc-compiled .so files"
fi
echo ""

# Clean mypyc-generated .c files (but preserve rawrefmodule.c)
echo "3. Removing mypyc-generated .c files..."
C_COUNT=0

# Find all .c files, excluding rawrefmodule.c
while IFS= read -r cfile; do
    # Skip if it's rawrefmodule.c
    if [[ "$cfile" == *"/rawrefmodule.c" ]]; then
        continue
    fi
    rm -f "$cfile"
    echo "  - Deleted: $(basename "$cfile")"
    ((C_COUNT++))
done < <(find . -name "*.c" -type f 2>/dev/null)

if [ $C_COUNT -eq 0 ]; then
    echo "  ℹ No mypyc-generated .c files found"
else
    echo "  ✓ Removed $C_COUNT mypyc-generated .c files"
fi
echo ""

# Remove __pycache__ directories
echo "4. Removing __pycache__ directories..."
PYCACHE_COUNT=$(find . -type d -name "__pycache__" 2>/dev/null | wc -l)
if [ $PYCACHE_COUNT -gt 0 ]; then
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    echo "  ✓ Removed $PYCACHE_COUNT __pycache__ directories"
else
    echo "  ℹ No __pycache__ directories found"
fi
echo ""

# Remove .pyc files
echo "5. Removing .pyc files..."
PYC_COUNT=$(find . -name "*.pyc" -type f 2>/dev/null | wc -l)
if [ $PYC_COUNT -gt 0 ]; then
    find . -name "*.pyc" -type f -delete 2>/dev/null
    echo "  ✓ Removed $PYC_COUNT .pyc files"
else
    echo "  ℹ No .pyc files found"
fi
echo ""

echo "========================================"
echo "✓ Cleanup complete!"
echo "========================================"
echo ""
echo "Preserved:"
echo "  • rawref/rawrefmodule.c (C extension source)"
echo "  • rawref/_rawref.*.so (C extension binary)"
echo ""
