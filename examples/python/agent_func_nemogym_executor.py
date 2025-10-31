"""
Agent Executor for NeMo Gym Integration with vLLM.

This module implements an agent executor that bridges vLLM inference engine
with NeMo Gym's rollout collection system for reinforcement learning training.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict
from uuid import uuid4

import requests
import torch
import uvicorn
from fastapi import FastAPI, Request
from nemo_gym.cli import RunHelper
from nemo_gym.global_config import GlobalConfigDictParserConfig, find_open_port
from nemo_gym.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper
from nemo_gym.server_utils import get_global_config_dict
from omegaconf import OmegaConf, open_dict
from vllm import SamplingParams

from openrlhf.utils.agent import AgentExecutorBase, AgentInstanceBase

# Configure logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def prepare_data(question: str, expected_answer: str, sampling_params: SamplingParams) -> Dict[str, Any]:
    """
    Prepare data item for NeMo Gym rollout collection.

    Args:
        question: The math problem question to solve
        expected_answer: The expected correct answer for evaluation
        sampling_params: vLLM sampling parameters (not used in this function but kept for API consistency)

    Returns:
        Dictionary containing the structured data item for rollout collection
    """
    # Reference to the agent that will process this data
    agent_ref = {"type": "responses_api_agents", "name": "library_judge_math_simple_agent"}

    # Prepare the conversation template for the math problem
    responses_create_params = {
        "input": [
            {
                "role": "system",
                "content": "Your task is to solve a math problem.  Make sure to put the answer (and only the answer) inside \\boxed{}.",
            },
            {"role": "user", "content": question},
        ],
    }

    # Construct the complete data item
    data_item = {
        "responses_create_params": responses_create_params,
        "question": question,
        "expected_answer": expected_answer,
        "agent_ref": agent_ref,
    }
    return data_item


class AgentInstance(AgentInstanceBase):
    """
    Basic agent instance that passes states through without modification.

    This is a minimal implementation that serves as a placeholder for more
    complex agent logic if needed in the future.
    """

    async def step(self, states: dict, **kwargs) -> Dict[str, Any]:
        return states


class AgentExecutor(AgentExecutorBase):
    """
    Agent executor that integrates vLLM inference with NeMo Gym rollout collection.

    This class sets up a vLLM HTTP server compatible with OpenAI API and
    initializes NeMo Gym services for collecting rollouts during RL training.
    """

    def __init__(self, max_steps: int, max_length: int, llm_engine, hf_tokenizer, result_queue):
        """
        Initialize the AgentExecutor with vLLM and NeMo Gym services.

        Args:
            max_steps: Maximum number of steps per episode
            max_length: Maximum sequence length for generation
            llm_engine: The vLLM engine instance for inference
            hf_tokenizer: HuggingFace tokenizer for the model
            result_queue: Queue for collecting execution results
        """
        # Configure server endpoints
        self.vllm_server_host = "127.0.0.1"  # Local vLLM server host
        self.vllm_server_port = find_open_port()  # Auto-assign available port
        self.head_server_host = "0.0.0.0"  # NeMo Gym head server host
        self.head_server_port = find_open_port()  # Auto-assign available port

        # Start vLLM HTTP server with OpenAI-compatible API
        try:
            self._start_vllm_server(llm_engine, hf_tokenizer)
            logger.info("vLLM HTTP server started (OpenAI style /v1/chat/completions)")
        except Exception as e:
            logger.error(f"Failed to start vLLM HTTP server: {e}")
            raise

        # Initialize NeMo Gym services for rollout collection
        try:
            self._start_nemogym_services(hf_tokenizer)
            logger.info("NeMo Gym services started via RunHelper")
        except Exception as e:
            logger.error(f"Failed to start NeMo Gym services: {e}")
            raise

        # Initialize parent class
        super().__init__(AgentInstance, max_steps, max_length, llm_engine, hf_tokenizer, result_queue)

    def _wait_for_vllm_server(self) -> bool:
        """
        Wait for the vLLM HTTP server to become ready by polling the health endpoint.

        Returns:
            True if server is ready, False if max retries exceeded
        """
        max_retries = 50
        retry_interval = 1  # seconds

        for i in range(max_retries):
            try:
                response = requests.get(f"http://{self.vllm_server_host}:{self.vllm_server_port}/health", timeout=5)
                if response.status_code == 200:
                    logger.info("vLLM HTTP server is ready")
                    return True
            except Exception as e:
                logger.debug(f"Waiting for vLLM server (attempt {i+1}/{max_retries}): {e}")
            time.sleep(retry_interval)

        logger.error(f"vLLM server did not start after {max_retries} attempts")
        return False

    def _start_vllm_server(self, llm_engine, hf_tokenizer):
        """
        Start a FastAPI server that wraps vLLM with OpenAI-compatible API endpoints.

        Args:
            llm_engine: The vLLM engine instance for inference
            hf_tokenizer: HuggingFace tokenizer for the model
        """
        app = FastAPI(title="vLLM OpenAI-Compatible Server")

        @app.get("/health")
        async def health():
            """Health check endpoint to verify server is running."""
            return {"status": "healthy", "model": getattr(hf_tokenizer, "name_or_path", "unknown")}

        @app.post("/tokenize")
        async def tokenize(request: Request):
            """
            Tokenize endpoint compatible with vLLM tokenizer API and OpenAI chat format.

            Converts chat messages to token IDs, using chat template if available.
            """
            body = await request.json()
            messages = body["messages"]

            # Try to use the tokenizer's chat template if available
            if hasattr(hf_tokenizer, "apply_chat_template"):
                try:
                    text = hf_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                except Exception as e:
                    logger.warning(f"apply_chat_template failed: {e}, falling back to manual formatting")
                    # Fallback to manual formatting
                    text = ""
                    for msg in messages:
                        role = msg.get("role", "")
                        content = msg.get("content", "")
                        if role == "system":
                            text += f"System: {content}\n"
                        elif role == "user":
                            text += f"User: {content}\n"
                        elif role == "assistant":
                            text += f"Assistant: {content}\n"
            else:
                # Manual formatting if chat template is not available
                text = ""
                for msg in messages:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role == "system":
                        text += f"System: {content}\n"
                    elif role == "user":
                        text += f"User: {content}\n"
                    elif role == "assistant":
                        text += f"Assistant: {content}\n"

            # Encode the formatted text to token IDs
            token_ids = hf_tokenizer.encode(text)
            return {"tokens": token_ids, "count": len(token_ids), "max_model_len": self.max_length, "token_strs": None}

        @app.post("/v1/chat/completions")
        async def chat_completions(request: Request):
            """
            OpenAI-compatible chat completions endpoint.

            Generates text responses using vLLM with logprobs for RL training.
            """
            body = await request.json()
            model_name = body.get("model")

            # Verify model name matches
            expected_model = getattr(hf_tokenizer, "name_or_path", "unknown")
            if model_name != expected_model:
                raise ValueError(f"Model name mismatch: {model_name} != {expected_model}")

            messages = body.get("messages", [])
            if not messages:
                raise ValueError("No messages provided")

            # Convert messages to prompt text using chat template
            observation = hf_tokenizer.apply_chat_template(
                messages, tokenize=False, add_special_tokens=True, add_generation_prompt=False
            )
            prompt_token_ids = hf_tokenizer.encode(observation, add_special_tokens=True)

            # Configure sampling parameters with appropriate max tokens
            sampling_params = self.sampling_params
            sampling_params.max_tokens = self.max_length - len(prompt_token_ids)

            # Generate response using vLLM engine
            request_output = await self.generate(prompt_token_ids, sampling_params)

            # Extract generation results
            generated_text = request_output.outputs[0].text or ""
            generated_tokens = request_output.outputs[0].token_ids
            generated_logprobs = request_output.outputs[0].logprobs

            # Format logprobs in OpenAI-compatible format
            generated_logprobs_list = [
                {
                    "token": f"token_id:{generated_tokens[i]}",
                    "logprob": logprob[generated_tokens[i]].logprob,
                    "top_logprobs": [
                        {"token": f"token_id:{token_id}", "logprob": logprob_obj.logprob}
                        for token_id, logprob_obj in logprob.items()
                    ],
                }
                for i, logprob in enumerate(generated_logprobs)
            ]

            # Calculate token usage
            completion_tokens = len(hf_tokenizer.encode(generated_text)) if generated_text else 0

            # Return OpenAI-compatible response format
            return {
                "id": f"resp_{uuid4().hex}",
                "created": int(time.time()),
                "model": getattr(hf_tokenizer, "name_or_path", "policy-model"),
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": generated_text,
                        },
                        "finish_reason": "stop",
                        "logprobs": {"content": generated_logprobs_list},
                    }
                ],
                "usage": {
                    "prompt_tokens": len(prompt_token_ids),
                    "completion_tokens": completion_tokens,
                    "total_tokens": len(prompt_token_ids) + completion_tokens,
                },
            }

        def run_server():
            """Run uvicorn server in a separate thread."""
            uvicorn.run(app, host=self.vllm_server_host, port=self.vllm_server_port, log_level="info", loop="asyncio")

        # Start the server in a daemon thread
        self.vllm_server_thread = threading.Thread(target=run_server, daemon=True)
        self.vllm_server_thread.start()

        # Wait for server to become ready
        if not self._wait_for_vllm_server():
            raise RuntimeError("Failed to start vLLM HTTP server after waiting")

        logger.info(f"vLLM HTTP server successfully started on {self.vllm_server_host}:{self.vllm_server_port}")

    def _start_nemogym_services(self, hf_tokenizer):
        """
        Initialize NeMo Gym services with configuration for the math agent.

        Sets up resources servers, agents, and model servers for rollout collection.

        Args:
            hf_tokenizer: HuggingFace tokenizer to extract model name
        """
        # Configure policy model connection details
        policy_base_url = f"http://{self.vllm_server_host}:{self.vllm_server_port}/v1"
        policy_api_key = "EMPTY"  # No authentication required for local server
        policy_model_name = getattr(hf_tokenizer, "name_or_path", "policy-model")
        logger.info(f"Configuring NeMo Gym with policy model: {policy_model_name}")

        # Create configuration dictionary for NeMo Gym services
        initial_cfg = OmegaConf.create(
            {
                # Resources server for math evaluation
                "library_judge_math": {
                    "resources_servers": {
                        "library_judge_math": {
                            "entrypoint": "app.py",
                            "judge_model_server": {"type": "responses_api_models", "name": "policy_model"},
                            "judge_responses_create_params": {"input": []},
                            "should_use_judge": False,  # Disable judging for now
                            "domain": "math",
                        }
                    }
                },
                # Simple agent configuration for math problems
                "library_judge_math_simple_agent": {
                    "responses_api_agents": {
                        "simple_agent": {
                            "entrypoint": "app.py",
                            "resources_server": {"type": "resources_servers", "name": "library_judge_math"},
                            "model_server": {"type": "responses_api_models", "name": "policy_model"},
                        }
                    }
                },
                # Policy model configuration (points to local vLLM server)
                "policy_model": {
                    "responses_api_models": {
                        "vllm_model": {
                            "entrypoint": "app.py",
                            "base_url": policy_base_url,
                            "api_key": policy_api_key,
                            "model": policy_model_name,
                            "return_token_id_information": True,  # Required for RL training
                            "uses_reasoning_parser": False,
                        }
                    }
                },
                # Head server configuration
                "head_server": {"host": self.head_server_host, "port": self.head_server_port},
            }
        )

        # Initialize and start NeMo Gym services
        self.rh = RunHelper()
        self.rh.start(
            GlobalConfigDictParserConfig(
                initial_global_config_dict=initial_cfg,
                skip_load_from_cli=True,
                skip_load_from_dotenv=True,
            )
        )

        # Store global config for later use
        self.global_config_dict = get_global_config_dict()
        logger.info("NeMo Gym services configuration completed")

    async def execute(self, prompt: str, label: str, sampling_params: SamplingParams):
        """
        Execute a single rollout for the given prompt and collect results.

        This method runs the agent through NeMo Gym's rollout collection system,
        generates a response, evaluates it, and stores the result in the queue.

        Args:
            prompt: The input prompt/question
            label: The expected answer for reward calculation
            sampling_params: vLLM sampling parameters for generation
        """
        async with self.semaphore:
            # Configure rollout collection for this execution
            rch_config_dict = get_global_config_dict()
            with open_dict(rch_config_dict):
                rch_config_dict.agent_name = "library_judge_math_simple_agent"
                rch_config_dict.input_jsonl_fpath = ""
                rch_config_dict.output_jsonl_fpath = ""

            rchConfig = RolloutCollectionConfig.model_validate(rch_config_dict)
            rch = RolloutCollectionHelper(rchConfig=rchConfig)

            # Prepare data and run rollout
            data_item = prepare_data(prompt, label, sampling_params)
            self.sampling_params = sampling_params

            examples = [data_item]
            rollouts_response = await rch.run_examples(examples)

            # Extract rollout information
            response_output = rollouts_response[0]["response"]["output"][0]
            observation_tokens = response_output.get("generation_token_ids", [])
            rollout_log_probs = response_output.get("generation_log_probs", [])
            action_ranges = [(0, len(observation_tokens))]
            reward = rollouts_response[0]["library_reward"]

            # Convert to expected format
            observation_tokens = torch.tensor(observation_tokens, dtype=torch.long).tolist()

            # Package final response for training
            final_response = {
                "prompt": prompt,
                "label": label,
                "observation_tokens": observation_tokens,
                "reward": reward,
                "scores": reward,
                "extra_logs": {},
                "action_ranges": action_ranges,
                "rollout_log_probs": rollout_log_probs,
            }

            # Add to result queue
            await self.result_queue.put(final_response)

    def shutdown(self):
        """
        Gracefully shutdown all services.

        Stops NeMo Gym services and waits for the vLLM server thread to terminate.
        """
        logger.info("Shutting down AgentExecutor services...")
        try:
            self.rh.shutdown()
            logger.info("NeMo Gym services shut down successfully")
        except Exception as e:
            logger.error(f"Error shutting down NeMo Gym services: {e}")

        try:
            self.vllm_server_thread.join(timeout=5.0)
            logger.info("vLLM server thread terminated")
        except Exception as e:
            logger.error(f"Error joining vLLM server thread: {e}")

    def __del__(self):
        """Destructor to ensure cleanup on object deletion."""
        try:
            self.shutdown()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
