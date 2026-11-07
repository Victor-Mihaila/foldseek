#!/bin/sh -e
# lib/ggml is the ggml/ directory of llama.cpp (https://github.com/ggml-org/llama.cpp, 81bc6b8 as of 60d943ed)
# plus the ModernProst CUDA patches of d68b7f22: head_dim 32 flash-attention tile and MMA instances, a rope kernel
# for small head sizes and F16 input/output for the norm kernels. Re-apply them after updating:
#   git show d68b7f22 -- lib/ggml | git apply
COMMIT="${1:?usage: $0 <llama.cpp commit>}"
TMP="$(mktemp -d)"
git clone --filter=blob:none --no-checkout https://github.com/ggml-org/llama.cpp.git "${TMP}/llama.cpp"
git -C "${TMP}/llama.cpp" checkout "${COMMIT}" -- ggml
rm -rf lib/ggml
cp -R "${TMP}/llama.cpp/ggml" lib/ggml
rm -rf "${TMP}"
