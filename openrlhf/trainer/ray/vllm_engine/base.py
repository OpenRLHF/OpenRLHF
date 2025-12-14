import os
import queue

import ray

from ..utils import ray_noset_visible_devices


class BaseLLMRayActor:
    """Shared setup for all Ray actors backed by vLLM."""

    def __init__(self, *args, bundle_indices: list = None, **kwargs):
        import vllm
        from packaging import version

        # TODO: remove func_path from kwargs after refactoring
        kwargs.pop("agent_func_path", None)
        kwargs.pop("remote_rm_url", None)
        kwargs.pop("remote_rm_batch_size", None)

        noset_visible_devices = ray_noset_visible_devices()
        if kwargs.get("distributed_executor_backend") == "ray":
            # stop ray from manipulating *_VISIBLE_DEVICES at top-level when backend is ray
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            os.environ.pop("ROCR_VISIBLE_DEVICES", None)
            os.environ.pop("HIP_VISIBLE_DEVICES", None)
        elif noset_visible_devices:
            # Set CUDA_VISIBLE_DEVICES to the ray assigned GPU when backend is not ray
            os.environ["CUDA_VISIBLE_DEVICES"] = str(ray.get_gpu_ids()[0])

        num_gpus = kwargs.pop("num_gpus")
        if bundle_indices is not None:
            os.environ["VLLM_RAY_PER_WORKER_GPUS"] = str(num_gpus)
            os.environ["VLLM_RAY_BUNDLE_INDICES"] = ",".join(map(str, bundle_indices))
            print(f"creating LLM with bundle_indices={bundle_indices}")

        # Number of actors that will send prompt to this engine
        self.requests = {}
        self.response_queues = queue.Queue()

        full_determinism = kwargs.pop("full_determinism", False)
        if full_determinism:
            # https://github.com/vllm-project/vllm/blob/effc5d24fae10b29996256eb7a88668ff7941aed/examples/offline_inference/reproduciblity.py#L11
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

        self.kwargs = kwargs

        assert version.parse(vllm.__version__) > version.parse(
            "0.8.5"
        ), "Streaming VLLM version must be greater than 0.8.5"

        if version.parse(vllm.__version__) >= version.parse("0.9.0"):
            os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

        # fix: https://github.com/vllm-project/vllm/pull/21540
        if not os.environ.get("RAY_ADDRESS"):
            from ray._private.worker import global_worker

            os.environ["RAY_ADDRESS"] = global_worker.gcs_client.address

        os.environ["VLLM_USE_V1"] = "1"
