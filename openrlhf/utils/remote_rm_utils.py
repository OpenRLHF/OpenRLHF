import time
import ray
import requests
import torch

from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)


def request_api_wrapper(url, data, score_key="score", try_max_times=10):
    """request api wrapper"""
    headers = {
        "Content-Type": "application/json",
    }
    for _ in range(try_max_times):
        try:
            result = requests.post(url=url, json=data, headers=headers)
            result = result.json()
            return result[score_key]
        except Exception as e:
            logger.info(f"request url error, please check: {e}")
            time.sleep(0.5)
    logger.info(f"Request url error for {try_max_times} times, return None. Please check the api server.")
    return None


def remote_rm_fn(api_url, queries, score_key="score"):
    """remote reward model API
    api_url: RM API, We assume that the API supports two modes: merging query + response and not merging
    queries: query+response with the template
    design is made optional.
    score_key: RM score key
    """
    scores = request_api_wrapper(api_url, {"query": queries}, score_key)
    return torch.tensor(scores)


@ray.remote
def remote_rm_fn_ray(api_url, queries, score_key="score"):
    return remote_rm_fn(api_url, queries, score_key)


if __name__ == "__main__":
    # test utils
    url = "http:xxx/get_rm_score"
    score = remote_rm_fn(url, ["example query"], ["example response"])
    print(score)