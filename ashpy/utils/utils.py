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

"""
Set of utility functions
"""
from typing import List, Type


def validate_objects(elements: List, element_class: Type) -> bool:
    """
    Check if all the elements are instance of the element_class
    Args:
        elements: List of objects that should be instance of element_class
        element_class: class of the objects

    Returns:
        True if all the elements are instance of element_class
    Raises:
        ValueError if one element is not an instance of element_class
    """
    for element in elements:
        if not isinstance(element, element_class):
            raise ValueError(f"{element} is not a {element_class}")
    return True
