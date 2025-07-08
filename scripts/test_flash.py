"""

RUN THIS BEFORE RUNNIGN THIS SCRIPT:

module purge

module load cuda/12.4.1-fasrc01

module load gcc/12.2.0-fasrc01
module load cmake/3.31.6-fasrc01
module load cudnn

"""



import flash_attn, ctypes, platform, subprocess, torch
print("flash-attn OK, glibc:", platform.libc_ver())
print("torch :", torch.__version__, "CUDA:", torch.version.cuda)

# print vllm version
try:
    import vllm
    print("vllm:", vllm.__version__)
except ImportError:
    print("vllm not found")

import os
import torch
import asyncio
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine


async def main():
    """
    Check custom all-reduce compilation bug.
    """
    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        print(f"Found only {num_gpus} GPU. The custom all-reduce feature is used for multi-GPU communication.")
        print("This test requires at least 2 GPUs to be meaningful.")
        return

    model_path = "/n/holylabs/LABS/kempner_dev/Users/nikhilanand/Llama-3-8B-Instruct-HF"
    tensor_parallel_size = 2 

    print(f"Detected {num_gpus} GPUs. Testing vLLM engine initialization with tensor_parallel_size={tensor_parallel_size}...")

    engine_args = AsyncEngineArgs(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=0.5,
    )

    try:
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        print("vllm initialized successfully.")
        print("Custom all-reduce is supported in your environment.")
        engine._close_workers()


    except NotImplementedError as e:
        if "_C_custom_ar" in str(e):
            print(f"failed {e}")
        else:
            print(f"error {e}")
    except Exception as e:
        print(f"\n unexpected error occurred during engine init: {e}")

if __name__ == "__main__":
    asyncio.run(main())