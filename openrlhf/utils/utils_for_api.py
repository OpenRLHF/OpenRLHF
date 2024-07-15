import time
import ray
import requests

from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)


# TODO: better to encapsulate it into a class.
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


def remote_ref_fn(
        api_url, queries, num_tokens, attention_mask=None, responses=None, score_key="logits", api_batch_size=1
):
    """remote reference model API, the actor model and the ref model should use the same tokenizer.
    Note: So far, only the direct use of the token IDs API is considered safe, with the text API reserved for future work.
    api_url: reference model API, We assume that the API supports two modes: merging query + response and not merging
    queries: token ids or query or query+response with the template.
    responses: 'None' indicates that the response has been integrated into the query, which is appropriate for invoking
    the pure reference model. The higher-level reference model service is typically capable of caching identical queries
    or prefixes, hence this design is made optional.
    score_key: token logit key.
    api_batch_size: ref model API batch size.
    """
    if not api_url:
        api_url = "xxx"
    if not responses:
        responses = [""] * len(queries)
    # token ids api
    data = {
        "sequences": queries.tolist(),
        "num_actions": num_tokens,
        "attention_mask": attention_mask.tolist(),
    }
    scores = request_api_wrapper(api_url, data, score_key)

    # text api: TODO
    # # the actor model and the ref model should use the same tokenizer -> same num_tokens/actions
    # num_token_ref = request_api_wrapper(api_url, {'query': queries[0], 'response': responses[0]}, score_key)
    # assert num_tokens[0] == num_token_ref, 'the actor model and the ref model should use the same tokenizer'
    #
    # if api_batch_size == 1:
    #     scores = [request_api_wrapper(api_url, {'query': query, 'response': response}, score_key) for query, response in \
    #               zip(queries, responses)]
    # else:
    #     scores = []
    #     # batch
    #     for i in range(0, len(queries), api_batch_size):
    #         scores.extend([request_api_wrapper(api_url, {'query': queries[i:i + api_batch_size], \
    #                                                      'response': responses[i:i + api_batch_size]}, score_key)])

    return scores


def request_api_wrapper(url, data, score_key="score", try_max_times=10):
    """request api wrapper"""
    headers = {
        "Content-Type": "application/json",
    }

    for _ in range(try_max_times):
        try:
            result = requests.post(url=url, json=data, headers=headers).json()
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
