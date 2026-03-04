#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$REPO_DIR/data"
TARGET_DIR="$DATA_DIR/fineweb10B"
FINEWEB_SRC="$REPO_DIR/env/nanogpt/fineweb.py"

echo "=== FineWeb 10B Data Preparation ==="
echo "  Target dir: $TARGET_DIR"
echo ""

if ls "$TARGET_DIR"/fineweb_train_*.bin 1>/dev/null 2>&1; then
    NUM_SHARDS=$(ls "$TARGET_DIR"/fineweb_train_*.bin | wc -l)
    echo "Data already exists: $NUM_SHARDS train shards found in $TARGET_DIR"
    echo "Delete $TARGET_DIR contents to re-download."
    exit 0
fi

mkdir -p "$DATA_DIR"

# fineweb.py writes to os.path.dirname(__file__)/fineweb10B/
# So we copy it to data/ and run from there — it will create data/fineweb10B/
cp "$FINEWEB_SRC" "$DATA_DIR/fineweb.py"

cd "$DATA_DIR"
echo "Downloading FineWeb 10B dataset (this may take several hours)..."
conda run --live-stream -n aira-dojo python fineweb.py -v 10B

# Clean up the copy
rm -f "$DATA_DIR/fineweb.py"

echo ""
echo "Data ready at $TARGET_DIR/"
ls -lh "$TARGET_DIR"/*.bin | head -20
echo "..."
echo "Total shards: $(ls "$TARGET_DIR"/*.bin | wc -l)"
