"""VLM Multi-Turn Agent Example

A reference implementation showing how to build a multi-turn VLM agent
for RLHF training with OpenRLHF. Supports images in both the initial
prompt (from dataset) and in environment feedback (e.g. screenshots).

Interaction flow:
    Dataset prompt (with image) --> Model generates action
        --> Agent.step() returns feedback (optionally with new images)
        --> Model generates next action
        --> ... repeat until done

Key points for customization:
    1. Override `reset()` to set up your environment (e.g. launch browser, load task)
    2. Override `step()` to process the model's action and return feedback
    3. Return "environment_images" in step() to inject new images (e.g. screenshots)
    4. Return "rewards" as a scalar tensor for RL optimization

Usage:
    python3 -m openrlhf.cli.train_ppo_ray \\
        --train.agent_func_path examples/python/vlm_multiturn_agent.py \\
        --actor.model_name_or_path Qwen/Qwen3.5-0.8B \\
        --data.image_key images \\
        --data.max_images_per_prompt 4 \\
        ...
"""

import random
from typing import Any, Dict

import numpy as np
import torch
from PIL import Image

from openrlhf.utils.agent import AgentInstanceBase, MultiTurnAgentExecutor

# ──────────────────────────────────────────────────────────────────────
#  Replace this with your own environment / tool interaction logic
# ──────────────────────────────────────────────────────────────────────


def get_screenshot_from_environment():
    """Placeholder: capture a screenshot from your environment.

    In a real setup this would be e.g.:
        - pyautogui.screenshot()
        - selenium driver.get_screenshot_as_png()
        - gym env.render()

    Returns a PIL.Image or None if no screenshot is available.
    """
    rng = np.random.RandomState(random.randint(0, 10000))
    return Image.fromarray(rng.randint(0, 256, (28, 28, 3), dtype=np.uint8))


def evaluate_action(action_text: str, label: str) -> float:
    """Placeholder: score the model's action against the ground truth.

    Returns a float reward value. In a real setup this could be:
        - Exact match / fuzzy match against expected answer
        - Task completion check from the environment
        - LLM-as-judge score
    """
    if label.lower() in action_text.lower():
        return 1.0
    return 0.0


# ──────────────────────────────────────────────────────────────────────
#  Agent Implementation
# ──────────────────────────────────────────────────────────────────────

# Chat template tokens (Qwen-style, adjust for your model)
USER_START = "\n<|im_start|>user\n"
USER_END = "<|im_end|>\n<|im_start|>assistant\n"
IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"


class AgentInstance(AgentInstanceBase):
    """Multi-turn VLM agent with configurable step count.

    Each episode runs for `max_steps` turns. Intermediate steps can
    return text-only feedback or feedback with new images. The final
    step assigns a reward based on the model's last action.

    Customize by modifying:
        - max_steps: number of interaction rounds
        - step(): your environment logic (execute action, observe result)
        - The feedback templates to match your model's chat format
    """

    def __init__(self, *args, **kwargs):
        self.step_idx = 0
        self.max_steps = random.choice([3, 4])  # randomize for diversity

    async def reset(self, states: dict, **kwargs):
        """Called at the start of each episode.

        `states["observation"]` is the initial prompt from the dataset
        (already formatted with chat template and image placeholders).

        You can modify it here or set up your environment.
        """
        self.step_idx = 0
        self.max_steps = random.choice([3, 4])
        # Initialize your environment here if needed:
        # self.env = MyEnvironment(task=states["label"])
        return {"observation": states["observation"]}

    async def step(self, states: dict, **kwargs) -> Dict[str, Any]:
        """Process one model action and return environment feedback.

        Args:
            states["observation_text"]: full conversation so far (text)
            states["action_text"]:      model's latest generation
            states["label"]:            ground truth from dataset

        Returns a dict with these keys:
            rewards (torch.Tensor):             scalar reward for this step
            environment_feedback (str):         text to append to conversation
            environment_images (list, optional): new PIL images for this step
            done (bool):                        whether episode is finished
            scores (torch.Tensor, optional):    score for dynamic filtering
            extra_logs (dict, optional):         custom metrics to log

        Image feedback format:
            When returning images, include the model-specific image
            placeholder token in environment_feedback. For Qwen-style:
                "<|vision_start|><|image_pad|><|vision_end|>"
            One placeholder per image in environment_images list.
        """
        action_text = states["action_text"]
        label = states["label"]

        self.step_idx += 1
        done = self.step_idx >= self.max_steps

        # ── Final step: assign reward ──
        if done:
            reward_val = evaluate_action(action_text, label)
            return {
                "rewards": torch.tensor(reward_val),
                "scores": torch.tensor(reward_val),
                "environment_feedback": "\n<|im_end|>\n",
                "done": True,
                "extra_logs": {
                    "step_count": self.step_idx,
                    "max_steps": self.max_steps,
                },
            }

        # ── Intermediate step with screenshot ── (customize this)
        if self.step_idx % 2 == 0:
            screenshot = get_screenshot_from_environment()
            return {
                "rewards": torch.tensor(0.0),
                "environment_feedback": (
                    f"{USER_START}"
                    f"{IMAGE_PLACEHOLDER}\n"
                    f"Here is the updated screen. What should we do next?\n"
                    f"{USER_END}"
                ),
                "environment_images": [screenshot],
                "done": False,
            }

        # ── Intermediate step, text-only ── (customize this)
        return {
            "rewards": torch.tensor(0.0),
            "environment_feedback": (
                f"{USER_START}" f"Not quite right. Please look more carefully and try again.\n" f"{USER_END}"
            ),
            "done": False,
        }


# ──────────────────────────────────────────────────────────────────────
#  Required export: vllm_engine.py loads this class by name
# ──────────────────────────────────────────────────────────────────────


class AgentExecutor(MultiTurnAgentExecutor):
    def __init__(self):
        super().__init__(AgentInstance)
