from typing import List, Optional

from openrlhf.trainer.ppo_utils.experience_maker import Experience
from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)


class FilterHookBase:
    """Filter hook to optionally drop samples and report pass rate."""

    def apply(self, experiences: List[Experience]) -> List[Experience]:
        raise NotImplementedError

    def pass_rate(self) -> Optional[float]:
        return None

    def reset(self):
        """Reset internal stats if any."""
        return


class NoOpFilterHook(FilterHookBase):
    def apply(self, experiences: List[Experience]) -> List[Experience]:
        return experiences


class DynamicFilteringHook(FilterHookBase):
    """Group-level filtering based on avg reward in a target range."""

    def __init__(self, args):
        self.n_samples = args.n_samples_per_prompt
        self.min_r, self.max_r = args.dynamic_filtering_reward_range
        self.total_groups = 0
        self.valid_groups = 0

    def apply(self, experiences: List[Experience]) -> List[Experience]:
        if len(experiences) != self.n_samples:
            return []
        self.total_groups += 1

        scores = [exp.scores[0].item() for exp in experiences]
        avg_reward = sum(scores) / len(scores)

        is_valid = self.min_r < avg_reward < self.max_r
        if is_valid:
            self.valid_groups += 1
            return experiences

        logger.info(
            f"Filtered out: avg_reward={avg_reward:.2f}, threshold=({self.min_r:.2f}, {self.max_r:.2f}), scores={scores}"
        )
        return []

    def pass_rate(self) -> Optional[float]:
        if self.total_groups == 0:
            return None
        return self.valid_groups / self.total_groups * 100

    def reset(self):
        self.total_groups = 0
        self.valid_groups = 0
