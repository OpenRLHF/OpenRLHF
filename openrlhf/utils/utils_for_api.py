import time
import ray
import requests

from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)


@ray.remote
def remote_rm_fn_ray(api_url, queries, responses=None, score_key="score", api_batch_size=1):
    return remote_rm_fn(api_url, queries, responses, score_key, api_batch_size)


def remote_rm_fn(api_url, queries, responses=None, score_key="score", api_batch_size=1):
    """remote reward model API
    api_url: RM API, We assume that the API supports two modes: merging query + response and not merging
    queries: query or query+response with the template
    responses: 'None' indicates that the response has been integrated into the query, which is appropriate for invoking
    the pure RM. The higher-level RM service is typically capable of caching identical queries or prefixes, hence this
    design is made optional.
    score_key: RM score key
    api_batch_size: RM API batch size.
    """
    if not api_url:
        api_url = "xxx"  # TODO: OpenLLMAI default remote RM API
    if not responses:
        responses = [""] * len(queries)
    if api_batch_size == 1:
        scores = [
            request_api_wrapper(api_url, {"query": query, "response": response}, score_key)
            for query, response in zip(queries, responses)
        ]
    else:
        scores = []
        # batch
        for i in range(0, len(queries), api_batch_size):
            scores.extend(
                [
                    request_api_wrapper(
                        api_url,
                        {"query": queries[i: i + api_batch_size], "response": responses[i: i + api_batch_size]},
                        score_key,
                    )
                ]
            )

    return scores


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
            continue
    logger.info(f"Request url error for {try_max_times} times, return None. Please check the api server.")
    return None


if __name__ == "__main__":
    # test utils
    url = "http:xxx/get_rm_score"
    score = remote_rm_fn(url, ["example query"], ["example response"])
    print(score)
