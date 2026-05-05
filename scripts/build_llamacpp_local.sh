#!/bin/bash
set -euo pipefail

IMAGE="nvidia/cuda:12.4.1-devel-ubuntu22.04"
BRANCH="b5540"
OUTPUT_DIR="/tmp/opencode/llamacpp-build"
BINARY="$OUTPUT_DIR/llama-server"

echo "=== Building llama-server locally in Docker ==="
echo "  Image:  $IMAGE"
echo "  Branch: $BRANCH"
echo "  Output: $BINARY"

docker run --rm \
    -v "$OUTPUT_DIR:/output" \
    "$IMAGE" \
    bash -c "
set -e
apt-get update -qq && apt-get install -y -qq git cmake build-essential libcurl4-openssl-dev > /dev/null
git clone --depth 1 --branch $BRANCH https://github.com/ggerganov/llama.cpp /opt/llama.cpp
cd /opt/llama.cpp
cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_NO_VMM=ON
cmake --build build --config Release -j\$(nproc) --target llama-server
cp build/bin/llama-server /output/llama-server
echo 'Build complete.'
ls -lh /output/llama-server
"

echo "=== Done ==="
ls -lh "$BINARY"
file "$BINARY"
