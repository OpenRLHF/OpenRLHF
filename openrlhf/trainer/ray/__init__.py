# No implicit imports of deepspeed here to avoid vllm environment gets contaminated
from .vllm_engine import batch_vllm_engine_call, create_vllm_engines

__all__ = [
    "create_vllm_engines",
    "batch_vllm_engine_call",
]

# TokenSpeed backend — lazy import to avoid hard dependency
try:
    from .tokenspeed_engine import batch_engine_call as batch_tokenspeed_engine_call
    from .tokenspeed_engine import create_tokenspeed_engines

    __all__ += ["create_tokenspeed_engines", "batch_tokenspeed_engine_call"]
except ImportError:
    pass
