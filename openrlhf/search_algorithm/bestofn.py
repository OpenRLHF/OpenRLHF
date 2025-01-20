import re
import torch
import random

N = 8
TEMPERATURE = 1
strategy = "bestandworst"

def clean_pad_token(text, pad_token):
    return re.sub(pad_token, "", text)

def get_full_traj(traj, tokenizer, actor):
    input_ids = tokenizer(traj, return_tensors="pt")
    input_ids = {k: v.to(actor.model.device) for k, v in input_ids.items()}
    outputs = actor.model.generate(**input_ids, do_sample=True, max_new_tokens=1024,
                             temperature=TEMPERATURE, tokenizer=tokenizer, num_return_sequences=N,
                             pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    sequences = tokenizer.batch_decode(outputs, keep_special_tokens=True)
    sequences = [clean_pad_token(seq, tokenizer.pad_token) for seq in sequences]
    return sequences

def get_scores(trajs, tokenizer, critic):
    input_ids = tokenizer(trajs, return_tensors="pt", padding=True, truncation=True)
    input_ids = {k: v.to(critic.device) for k, v in input_ids.items()}
    outputs = critic.compute_value(**input_ids, return_dict=False)
    return (torch.clamp(outputs[0].squeeze(), min=-1, max=1) + 1) / 2

def search(query, tokenizer, actor, critic):
    trajs = get_full_traj(query, tokenizer, actor)
    scores = get_scores(trajs, tokenizer, critic)
    if strategy == "bestofn":
        best_idx = torch.argmax(scores)
        return [trajs[best_idx]]
    elif strategy == "bestandworst":
        best_idx = torch.argmax(scores)
        worse_idx = torch.argmin(scores)
        return [trajs[best_idx], trajs[worse_idx]]
    elif strategy == "random":
        return random.sample(trajs, 4)