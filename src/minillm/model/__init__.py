"""Model code.

- M0: Llama forward pass written here (load Hugging Face weights with safetensors), own generation loop.
- M1: KV cache (now in `minillm/cache/`, with M4's slots and M5's paged pool).
- Phase 5: DeepSeek-V3 architecture (MLA + MoE), developed on Moonlight-16B-A3B, then Kimi K2.
"""
