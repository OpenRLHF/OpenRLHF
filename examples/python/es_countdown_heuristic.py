import ast
import re
from typing import Dict, List


class CountdownAccuracyHeuristic:
    gpus = 0
    cpus = 2

    def should_call(self, label: str) -> bool:
        return bool(label and str(label).strip())

    def __call__(self, queries: List[str], prompts: List[str], labels: List[str]) -> Dict[str, List[float]]:
        rewards = []
        for query, label in zip(queries, labels):
            try:
                # 1. Safely handle pre-parsed lists from the dataset
                parsed = ast.literal_eval(label) if isinstance(label, str) else label

                expected_nums = sorted(int(n) for n in parsed[0])
                tgt = int(parsed[1][0]) if isinstance(parsed[1], list) else int(parsed[1])

                ans_matches = re.findall(r"<answer>(.*?)<\/answer>", query, re.DOTALL)
                if not ans_matches:
                    rewards.append(0.0)
                    continue

                raw_ans = ans_matches[-1].strip()

                # Remove the equation result if the model explicitly wrote it
                # e.g., converts "48 - 3 + 48 = 93" into "48 - 3 + 48"
                raw_ans = re.sub(rf"=\s*{tgt}\s*$", "", raw_ans)  # Catches " = 93" at the end
                raw_ans = re.sub(rf"^\s*{tgt}\s*=", "", raw_ans)  # Catches "93 = " at the start

                extracted_nums = [int(x) for x in re.findall(r"\d+", raw_ans)]

                if sorted(extracted_nums) != expected_nums:
                    rewards.append(0.0)
                    continue

                clean_expr = re.sub(r"[^0-9+\-*/(). ]", "", raw_ans).strip()

                if not clean_expr:
                    rewards.append(0.0)
                    continue

                # 2. Use an empty dict {} for builtins instead of None to prevent TypeError
                if "**" in clean_expr:
                    rewards.append(0.0)
                    continue

                # 2. Use an empty dict {} for builtins instead of None to prevent TypeError
                # Note this is not a security risk since this is an optional example to run on LLM generated code that we have already cleaned.
                result = float(eval(clean_expr, {"__builtins__": {}}, {}))

                if abs(result - tgt) < 1e-5:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)

            except Exception as e:
                # Print the exception to surface any unexpected errors during evaluation
                print(f"Accuracy calculation failed for label '{label}': {e}")
                rewards.append(0.0)

        return {"countdown_accuracy": rewards}


class CountdownFormatHeuristic:
    gpus = 0
    cpus = 2

    def should_call(self, label: str) -> bool:
        return bool(label and str(label).strip())

    def __call__(self, queries: List[str], prompts: List[str], labels: List[str]) -> Dict[str, List[float]]:
        return {
            "countdown_format": [
                max(
                    (0.1 if re.search(r"<think>.*?<\/think>", q, re.DOTALL) else 0.0)
                    + (0.5 if re.search(r"<answer>.*?<\/answer>", q, re.DOTALL) else 0.0),
                    1.0 if re.match(r"^<think>.*?<\/think>\n<answer>.*?<\/answer>$", q, re.DOTALL) else 0.0,
                )
                for q in queries
            ]
        }


HEURISTICS = [CountdownAccuracyHeuristic, CountdownFormatHeuristic]
