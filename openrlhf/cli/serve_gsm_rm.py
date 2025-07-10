import argparse
import re

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from openrlhf.utils.logging_utils import init_logger
from time import sleep
import re
import numpy as np

from math_verify import parse,verify

logger = init_logger(__name__)

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"


def extract_answer(completion):
    try:
        last_number = re.findall(r"\d+", completion)[-1]
        return eval(last_number)
    except:
        return INVALID_ANS
    
def parse_query(query):
    pattern = r"<\|start_header_id\|>(user|assistant|system)<\|end_header_id\|>\s*(.*?)(?=<\|start_header_id\|>(?:user|assistant|system)<\|end_header_id\|>|$)"
    matches = re.findall(pattern, query, re.DOTALL)
    print(f"Matches: {matches}", flush=True)
    # Build the messages list
    messages = []
    for role, content in matches:
        messages.append({
            "role": role,
            "content": content.strip()  # Strip any unnecessary whitespace
        })
    
    return messages


def strip_sequence(text, pad_token, eos_token):
    pad_token_escaped = re.escape(pad_token)
    eos_token_escaped = re.escape(eos_token)

    pattern = f"^({eos_token_escaped}|{pad_token_escaped})+"
    text = re.sub(pattern, "", text)

    pattern = f"({eos_token_escaped}|{pad_token_escaped})+$"
    text = re.sub(pattern, "", text)
    return text


class RewardModelProxy:
    def __init__(self):
        pass

    def get_reward(self, queries, answers):
        scores = []

        for i in range(len(queries)):
            query = queries[i]
            logger.info(f"queries[{i}]: {query}")
            try:
                #response = query.split("<|start_header_id|>assistant<|end_header_id|>")[1].strip()
                answer = parse(str(answers[i]))
            except:
                print("Error in parsing")
                logger.info(f"Answer[{i}]: {answers[i]}")
                answer = INVALID_ANS
            messages = parse_query(query)
            print(f"MESSAGES: {messages}", flush=True)
            assistant_message = next((msg for msg in messages if msg["role"] == "assistant"), None)
            if not assistant_message:
                raise ValueError("No assistant message found in the query")
            prediction = parse(assistant_message['content'])
            #print(f"ANSWER: {answer}, PREDICTION: {prediction}, {verify(answer, prediction) * 1.0}", flush=True)
            scores.append(verify(answer, prediction) * 1.0)
        logger.info(f"scores: {scores}")
        return scores
       

    
app = FastAPI()
reward_model = RewardModelProxy()

@app.post("/get_reward")
async def get_reward(request: Request):
    data = await request.json()
    queries = data.get("query")
    answers = data.get("labels")
    rewards = reward_model.get_reward(queries, answers)
    result = {"rewards": rewards}
    logger.info(f"Sent JSON: {result}")
    return JSONResponse(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000, help="Port number for the server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="IP for the server")

    args = parser.parse_args()

    print(app)
    uvicorn.run("openrlhf.cli.serve_gsm_rm:app", host=args.host, port=args.port, log_level="info", workers = 96)