"""
Example: Safe token handling in agentic RLHF training.

Demonstrates how to use AgentTokenHandler to ensure turn-boundary integrity
when training multi-turn agents with custom rewards.

Reference: https://github.com/OpenRLHF/OpenRLHF/issues/1128
"""

import logging

from transformers import AutoTokenizer

from openrlhf.utils.agent_tokenization import (
    DefaultAgentTokenHandler,
    TokenSequence,
)

logging.basicConfig(level=logging.DEBUG)


def safe_agentic_training_example():
    """Example of safe token handling using a local Llama 3.1 model."""
    # Replace this with the absolute path to your Llama 3.1 directory
    # e.g., "/Users/lucasmac/models/Llama-3.1-8B-Instruct"
    model_path = "/PATH/TO/YOUR/LOCAL/LLAMA-3.1"

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    except Exception as e:
        logging.warning("Local path failed: %s. Falling back to GPT-2 for logic test.", e)
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    token_handler = DefaultAgentTokenHandler(
        tokenizer=tokenizer,
        verbose=True,
        add_newline_before_feedback=True,
    )

    # Simulated agentic training loop: Turn 1 (Observation -> Action -> Feedback)
    observation_text = "You are in a room. A door is in front of you."
    observation_tokens = tokenizer(
        observation_text,
        add_special_tokens=True,
        return_tensors="pt",
    )[
        "input_ids"
    ][0].tolist()

    # Simulated vLLM output: action tokens without EOS (issue #1128)
    action_tokens_from_vllm = [512, 513, 514]
    action_text = "Open the door"

    # Process action with token handler; EOS is appended if missing
    action_seq = token_handler.handle_action_tokens(
        action_tokens_from_vllm,
        action_text,
    )
    print(f"Action sequence after processing: {action_seq}")

    # Environment feedback
    feedback_text = "The door opens. You see a hallway."
    feedback_seq = token_handler.handle_feedback_tokens(feedback_text)
    print(f"Feedback sequence: {feedback_seq}")

    # Concatenate all turn sequences
    turn_sequences = [
        TokenSequence(observation_tokens, "observation", observation_text),
        action_seq,
        feedback_seq,
    ]

    concatenated_tokens, report = token_handler.concatenate_sequences(turn_sequences)

    print(f"\n{report.summary()}")
    print(f"Turn boundaries: {report.turn_boundary_indices}")

    if report.is_valid:
        print("Turn-boundary integrity validated.")
    else:
        print(f"Warnings: {report.token_offset_warnings}")

    # Use concatenated_tokens for model training.
    # Token ranges can be extracted from report.turn_boundary_indices
    # for gradient computation on specific turns.


if __name__ == "__main__":
    safe_agentic_training_example()
