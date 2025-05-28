import torch


def reward_func(queries, prompts, labels, **kwargs):
    # queries is prompts + responses
    # labels is answers
    print(queries)
    reward = torch.randint(0, 2, (len(queries),)).float()

    # rewards is used the calculate the advantage (with format reward)
    # scores is used in dynamic filtering (0-1 reward)
    # extra_logs is used to log the info into wandb
    return {"rewards": reward, "scores": reward, "extra_logs": {"dummy_scores": reward}}
