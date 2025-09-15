##### Inference prompt #####

# The first round prompt for all methods
DEVEVAL_DIRECT_TURN1 = """Please complete a Python function that solves the following requirement in a markdown code block:
## The contexts above the function:
```Python
{contexts_above}
```
## Requirement
{requirement}
## Output Format
The function should be put in a markdown code block, you should start with:
```python
{function_signature}
```
"""

# Full History
DEVEVAL_DIRECT_TURNK = """Refine the code based on the new requirement:{requirement}"""

# Code Edit
DEVEVAL_EDIT_TURNK = """I will provide you with a code snippet and an edit instruction. Your task is to edit the code to suit the needs.
## The contexts above the function:
```Python
{contexts_above}
```
## Previous Code:
```python
{previous_code}
```
Please write a Python function that solves the following requirement in a markdown code block:
{requirement}
"""

#####  Test case synthesis prompt  #####

# Prompt for test case synthesis
TEST_GEN_PROMPT = """You will be given an original reference code, original test cases, a previous code implementation (if it exists), a list of features that need to be implemented, and features that should not be implemented.

Your task is to rewrite the original reference code and test cases.

Your rewritten code and tests must satisfy:
1. The code must implement all the required features and **must not implement any of the prohibited features**. The original reference code implements all features, so you can remove the parts that shouldn't be implemented; or if the previous code only implements part of the required features, you can modify it to implement all currently required features.
2. The tests should focus on the parts that the previous code failed to implement and **ensure that the current code passes these test cases while the previous code would fail them**.

--- Original reference code  ---
```python
{original_solution}
```
--- Original test cases ---
```python
{original_test_case}
```
---  Features to implement   ---
{current_requirement}
--- Features NOT to implement ---
{not_to_impl}

--- Reference code in previous round (If exists) ---
```python
{last_solution}
```

When rewriting test cases, pay special attention to:
- Strictly follow the format of the original test case.
- If the original test is inside a class, use `self` as the first parameter. If the original test is a standalone test, do not use `self` as the first parameter.
- Follow the API usage in the original test. Never define `class XX` in the test block.
- Import statements should be within the function block, not global!
- If the original test used annotations, you should use them as well.

Here's an example for reference:
If the original test case is:
```
@patch(...)
def test_with_working_ipv6(self, socket_mock):
    socket_mock.return_value = Mock()
    assert network.try_ipv6_socket()
```

A correct rewrite would be:
```
@patch(...)
def test_try_ipv6_socket_turn1(self, socket_mock):
    import ...
    # ...
```
Because: 1. It uses annotations like the original. 2. The original test has a `self` parameter, so this test should be the same. 3. It puts import statements inside the function block rather than globally.

An incorrect rewrite would be:
```
import socket
import network
def test_try_ipv6_socket_turn1():
    class dummyReq:
        ...
    from unittest.mock import patch, Mock
    with patch("socket.has_ipv6", True), patch("socket.socket") as socket_mock:
        socket_mock.return_value = Mock()
        assert network.try_ipv6_socket() is True 
```
Because: 1. It doesn't follow the original annotation pattern. 2. It doesn't include the `self` parameter like the original. 3. It doesn't put import statements inside the function block. 4. It should not define any class!
"""

# Format for test case synthesis, will be concatenated to the end
TEST_GEN_FORMAT_PROMPT = """You must output in the following json format wrapped with ```json ```:
```json
[
    {
        "solution": The reference function code for this round of requirements. Strictly follow the signature of the original code. **Do not output any code beyond the original reference code, such as import statement or class!!!**.
        "test": This round of test code. Function names end with `_turnk` such as `_turn1`. Follow the rule above.
    }
]
```
"""

# Prompt for feedback if there is an error
TEST_GEN_FEEDBACK_PROMPT = """Based on the execution, the solution code or test code you provided seems to have problems. Please rethink the code and test code seriously based on the feedback and refine them to meet the requirements. 

--- feedback from human programmer and runtime environment  ---
{feedback}
"""

# Format for feedback, will be concatenated to the end
TEST_GEN_FEEDBACK_FORMAT_PROMPT = """You must only output in the following json format wrapped with ```json ```:
```json
[
    {
        "reason": Why does this happen? Is there a problem with the test or the answer code? Analyze it in detail.
        "plan": Your plan to handle this problem.
        "solution": The reference code you refined.
        "test": The test **function** (not class level) you refined. Follow the rule above. 
    }
]
```
"""

# Model response
ASSISTANT_PROMPT_REPLACE = """
```json
[
    {
        "solution": "{solution}",
        "test": "{test}"
    }
]
```
"""
