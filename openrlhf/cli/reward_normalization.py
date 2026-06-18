"""Helpers for explaining reward normalization routing in PPO training."""

LOCAL_REWARD_MODEL = "local_reward_model"
PYTHON_REWARD_FUNC = "python_reward_func"
HTTP_REWARD_API = "http_reward_api"
AGENT_FUNC = "agent_func"


def as_reward_url_list(remote_url):
    if remote_url is None:
        return []

    urls = [remote_url] if isinstance(remote_url, str) else remote_url
    normalized_urls = []
    for url in urls:
        if url is None:
            continue
        normalized_url = str(url).strip()
        if normalized_url:
            normalized_urls.append(normalized_url)
    return normalized_urls


def classify_reward_source(remote_url=None, agent_func_path=None):
    """Classify the reward source using train_ppo_ray.py routing semantics."""
    if agent_func_path:
        return AGENT_FUNC

    urls = as_reward_url_list(remote_url)
    if not urls:
        return LOCAL_REWARD_MODEL
    if str(urls[0]).endswith(".py"):
        return PYTHON_REWARD_FUNC
    return HTTP_REWARD_API


def reward_normalization_warning(normalize_enable, reward_source):
    if not normalize_enable or reward_source == LOCAL_REWARD_MODEL:
        return None

    if reward_source == PYTHON_REWARD_FUNC:
        source_detail = "Python reward_func files"
        resolution = "Normalize custom rewards inside reward_func when that behavior is desired."
    elif reward_source == AGENT_FUNC:
        source_detail = "custom agent_func executors"
        resolution = "Normalize custom rewards inside the agent executor when that behavior is desired."
    else:
        source_detail = "remote reward APIs"
        resolution = "If the server is openrlhf.cli.serve_rm, pass --reward.normalize_enable to that server command."

    return (
        "[Warning] --reward.normalize_enable does not transform rewards returned by "
        f"{source_detail}. It only configures local reward/critic model heads. "
        "If a critic is active, critic values may still use this setting. "
        f"{resolution}"
    )
