#!/bin/bash
# run.sh - Execute the doc-crawl Python script in a dedicated virtual environment.
# Usage: bash run.sh <url> [-o output_dir]
# Example: bash run.sh https://docs.nvidia.com/cuda/cuda-programming-guide -o doc

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
PYTHON_SCRIPT="$SCRIPT_DIR/crawl.py"

# Install uv if not available
if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Source the env so uv is on PATH
    export PATH="$HOME/.local/bin:$PATH"
fi

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR..."
    uv venv "$VENV_DIR"
fi

# Install dependencies
echo "Installing dependencies..."
uv pip install --python "$VENV_DIR/bin/python" \
    requests beautifulsoup4 markdownify

MERGE_SCRIPT="$SCRIPT_DIR/merge.py"

# Run the crawler
echo "Running doc crawler..."
"$VENV_DIR/bin/python" "$PYTHON_SCRIPT" "$@"

# Run merge on each top-level doc directory
OUTPUT_DIR="doc"
for arg in "$@"; do
    case "$prev" in
        -o|--output) OUTPUT_DIR="$arg" ;;
    esac
    prev="$arg"
done

echo "Running short doc merge..."
for dir in "$OUTPUT_DIR"/*/; do
    [ -d "$dir" ] && "$VENV_DIR/bin/python" "$MERGE_SCRIPT" "$dir"
done
