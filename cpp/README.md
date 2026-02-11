# llama.rn C++ sources - This is the entire logic - This build just compiles directly to Java / Android instead of using JS. If you want to use the JS / react-native version.

Checkout [llama.rn](https://github.com/mybigday/llama.rn)

## Still a work in progress / Things need to be adapted for a Java env

- Most files mirror [llama.cpp](https://github.com/ggml-org/llama.cpp) and are copied from `third_party/llama.cpp` during `npm run bootstrap`. Do not edit mirrored files directly; change the submodule and rerun bootstrap to re-sync.
- llama/ggml symbols are prefixed to `LM_`/`lm_` by the bootstrap script to avoid collisions with other native modules.
- llama.rn-specific code lives in `rn-*` files:
  - `rn-llama.*`: context wrapper and lifecycle
  - `rn-completion.*`: legacy completion flow
  - `rn-slot.*`, `rn-slot-manager.*`: parallel decoding/queueing
  - `rn-mtmd.hpp`: multimodal (vision/audio) helpers
  - `rn-tts.*`: TTS/vocoder integration
  - `rn-common.hpp`: shared helpers (tokenization, rerank formatting, etc.)
- If you need to patch llama.cpp behavior, edit the synced file here, then create a patch under `scripts/patches/` so bootstrap reapplies the change after the next sync.
