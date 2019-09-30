# Copyright 2019 Zuru Tech HK Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Event definition as Enum."""

from enum import Enum, auto


class Event(Enum):
    """Define all possible events."""

    ON_BATCH_START = auto()
    ON_BATCH_END = auto()
    ON_TRAIN_START = auto()
    ON_TRAIN_END = auto()
    ON_EPOCH_START = auto()
    ON_EPOCH_END = auto()
    ON_EXCEPTION = auto()
