#!/bin/bash
# Prepare minimal training data for the debug NanoGPT pipeline.
# Generates synthetic token data (~20MB each for train + val) in the correct
# binary format expected by NanoGPT's DistributedDataLoader.
#
# Data is stored in data/nanogpt_debug/fineweb10B/ and symlinked into
# env/nanogpt_debug/ so that shutil.copytree(symlinks=True) doesn't duplicate
# the data for every idea variant.
#
# Usage: cd Automated-AI-Researcher && bash scripts/prepare_debug_data.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="${PROJECT_DIR}/data/nanogpt_debug/fineweb10B"
ENV_LINK="${PROJECT_DIR}/env/nanogpt_debug/fineweb10B"

if [ -d "$DATA_DIR" ] && [ -f "$DATA_DIR/fineweb_val_000000.bin" ] && [ -f "$DATA_DIR/fineweb_train_000001.bin" ]; then
    echo "Debug data already exists at $DATA_DIR"
    echo "  Val shard: $(du -h "$DATA_DIR/fineweb_val_000000.bin" | cut -f1)"
    echo "  Train shard: $(du -h "$DATA_DIR/fineweb_train_000001.bin" | cut -f1)"
else
    echo "Generating synthetic debug data..."
    mkdir -p "$DATA_DIR"

    python3 -c "
import numpy as np
import os, sys

DATA_DIR = sys.argv[1]
NUM_TOKENS = 10_000_000  # 10M tokens per shard (~20MB)
VOCAB_SIZE = 50257       # GPT-2 vocab

def write_shard(filepath, num_tokens):
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520  # magic
    header[1] = 1         # version
    header[2] = num_tokens
    tokens = np.random.randint(0, VOCAB_SIZE, size=num_tokens, dtype=np.uint16)
    with open(filepath, 'wb') as f:
        f.write(header.tobytes())
        f.write(tokens.tobytes())
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f'  Written {filepath} ({num_tokens:,} tokens, {size_mb:.1f} MB)')

np.random.seed(42)
write_shard(os.path.join(DATA_DIR, 'fineweb_val_000000.bin'), NUM_TOKENS)
write_shard(os.path.join(DATA_DIR, 'fineweb_train_000001.bin'), NUM_TOKENS)
print('Done.')
" "$DATA_DIR"
fi

# Create/update symlink in env dir (absolute path so it works from any CWD)
if [ -L "$ENV_LINK" ]; then
    rm "$ENV_LINK"
elif [ -d "$ENV_LINK" ]; then
    echo "WARNING: $ENV_LINK is a real directory, removing it"
    rm -rf "$ENV_LINK"
fi
ln -s "$DATA_DIR" "$ENV_LINK"
echo "Symlinked: $ENV_LINK -> $DATA_DIR"

echo ""
echo "Debug data ready. You can now run:"
echo "  bash scripts/run_debug.sh [GPU_ID]"
