# Failure-Attribution Prompt

```text
You will be provided with cumulative requirements for a coding task, a model-generated code snippet, the current-round tests, and the test execution information. The record is known to have failed the execution-based tests. Your task is to attribute the most plausible reason for the failure.

Evaluation steps:
1. Read the cumulative requirements, including the current-round requirement.
2. Read the model-generated code.
3. Read the current-round tests and the test execution information.
4. Compare the generated code against the requirements and the tests.
5. Assign one primary failure category from the taxonomy.
6. Optionally assign one secondary failure category if another category is clearly relevant.

Cumulative requirements:
{CUMULATIVE_REQUIREMENTS}

Model-generated code:
{GENERATED_CODE}

Current-round tests:
{CURRENT_TESTS}

Test execution information:
{EXECUTION_INFO}

Taxonomy of failure categories:
1. Missing dependency declarations
2. No error messages for unexpected input cases
3. Inefficiency, unnecessary statements
4. Edge case not handled
5. Logic error
6. Function or variable not defined
7. Existing function, variable, or API used incorrectly
8. Code not completed
9. Test mismatch with requirement

Output a JSON object only:
{
  "primary_category": "<one category name from the taxonomy>",
  "secondary_category": "<one category name from the taxonomy, or empty string>",
  "reason": "<brief explanation within 150 words>"
}
```
