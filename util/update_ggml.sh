#!/bin/sh -e
COMMIT="${1:?usage: $0 <llama.cpp commit>}"
TMP="$(mktemp -d)"
git clone --filter=blob:none --no-checkout https://github.com/ggml-org/llama.cpp.git "${TMP}/llama.cpp"
git -C "${TMP}/llama.cpp" checkout "${COMMIT}" -- ggml
rm -rf lib/ggml
cp -R "${TMP}/llama.cpp/ggml" lib/ggml
rm -rf "${TMP}"
