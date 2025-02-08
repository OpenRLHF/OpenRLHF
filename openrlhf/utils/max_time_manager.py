# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright (c) 2025, Jian Hu.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# File is modified from https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/cli/train_sft.py

import re
import time

class MaxTimeManager:
    def __init__(self, max_time: str):
        """
        Stops training after time interval has been reached.
        Args:
            max_time (str): The max time to run in HH:MM:SS format.

        Raises:
            ValueError: If the time interval format is invalid.
        """
        self.save_interval = self._parse_time_interval(max_time)
        self.max_time_reached = False
        self.start_time = time.time()

    def _parse_time_interval(self, interval_str):
        match = re.fullmatch(r"(\d+):(\d+):([0-5]?\d):([0-5]?\d)", interval_str)
        if not match:
            raise ValueError(
                f"Invalid time interval format: '{interval_str}'. Use DD:HH:MM:SS format."
            )
        days, hours, minutes, seconds = map(int, match.groups())
        return days * 86400 + hours * 3600 + minutes * 60 + seconds

    def check(self):
        current_time = time.time()
        if (current_time - self.start_time) >= self.save_interval:
            print(f"\n\n>>> Max time has been reached. Signalling to save a checkpoint.\n\n")
            self.max_time_reached = True
        return self.max_time_reached
