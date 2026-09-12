import os
from typing import MutableMapping

PYTORCH_ALLOC_CONF_ENV_VARS = (
    "PYTORCH_CUDA_ALLOC_CONF",
    "PYTORCH_ALLOC_CONF",
)


def configure_vllm_allocator_env(
    enable_sleep_mode: bool,
    env_vars: MutableMapping[str, str] | None = None,
) -> list[str]:
    """Make allocator settings compatible with vLLM sleep mode.

    vLLM's ``CuMemAllocator`` rejects ``expandable_segments:True``. Preserve
    every other allocator setting so the rollout actor does not discard user
    tuning unrelated to expandable segments.

    Returns the names of environment variables that were updated.
    """
    if not enable_sleep_mode:
        return []

    env_vars = os.environ if env_vars is None else env_vars
    updated = []

    for env_name in PYTORCH_ALLOC_CONF_ENV_VARS:
        value = env_vars.get(env_name)
        if value is None:
            continue

        kept_options = []
        removed = False
        for option in value.split(","):
            option = option.strip()
            if not option:
                continue
            key, separator, option_value = option.partition(":")
            if (
                separator
                and key.strip().lower() == "expandable_segments"
                and option_value.strip().lower() in {"true", "1"}
            ):
                removed = True
                continue
            kept_options.append(option)

        if not removed:
            continue
        if kept_options:
            env_vars[env_name] = ",".join(kept_options)
        else:
            env_vars.pop(env_name, None)
        updated.append(env_name)

    return updated
