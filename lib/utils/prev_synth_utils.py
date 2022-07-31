"""
utils for the interpretor + translator style synthesis
"""

import re
from typing import Tuple

pregex1 = re.compile(r"(\w+)[(](.+)[)]")
pregex2 = re.compile(r".*(\w+)[(](.+)[)].*")


def parse_spec(spec, asp=False) -> Tuple[str, str]:
    # print(spec)
    if asp:
        parsed_spec = re.match(pregex2, spec)
    else:
        parsed_spec = re.match(pregex1, spec)
    func = parsed_spec.group(1)
    args = parsed_spec.group(2)

    return func, args