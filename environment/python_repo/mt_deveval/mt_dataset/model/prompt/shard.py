DECOMPOSE_INSTRUCTION_PROMPT = """Given a python function code and its description, your task is to convert it into a basic requirement instruction and multiple restriction instructions.

The instructions you generated simulates the process of users gradually clarifying their originally vague needs when using AI.

Please make sure:
1. The number of rounds is scientifically determined based on the complexity of the task, but cannot exceed 5 rounds
2. The instructions you give are verifiable and will not conflict with the previous ones. 
3. Later instructions should not be easy for users to complete earlier.

--- Python function code ---
{original_code}

--- Function description ---
{original_requirement}
"""

DECOMPOSE_OUTPUT_FORMAT = """You must output in the following json format wrapped with ```json ```:
```json
[
    {
        "turn": 1,
        "prompt": Basic requirement instruction,including function signature. 
    },
    {
        "turn": k,
        "prompt": Restrictive instruction in round k.
    },
    ...
]
```
"""