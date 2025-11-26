## Performance in one-shot version v.s. iterative development

Results show that the one-shot setting consistently outperforms the iterative setting in accuracy, indicating that LLMs face considerable challenges in iterative code generation—even top-performing models achieve relatively low accuracy in this scenario.

| Models | Setting | py-func | py-repo | java-func | java-repo |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **DS-T** | one-shot | **0.2958** | **0.4196** | **0.5349** | **0.4750** |
| | iterative | 0.1408 | 0.0225 | 0.1977 | 0.1000 |
| **QW3-Coder** | one-shot | **0.3239** | **0.3750** | **0.5465** | **0.4250** |
| | iterative | 0.1972 | 0.0225 | 0.1919 | 0.1500 |
| **OSS-20-T** | one-shot | **0.3099** | **0.3571** | **0.5116** | **0.1000** |
| | iterative | 0.1620 | 0.0337 | 0.1977 | 0.1000 |
| **OSS-20** | one-shot | **0.3099** | **0.3482** | **0.4942** | **0.1282** |
| | iterative | 0.1901 | 0.0112 | 0.2035 | 0.1250 |

## Other Observation

We analyze the cases where the OSS-20(CE)’s Complete Rate dropped to 0% in Table6.

Our analysis reveals a 'snowball effect':76.4% of tasks failed in Round 1, and 46% remained broken throughout, indicating that early errors in repository level tasks are difficult to rectify.

The dominant error type was AssertionError(45%), while AttributeErrors increased in later rounds, implying the model incorrectly altered function signatures.

Critically, 'Pass-to-Error' transitions(13.9%) exceeded 'Error-to-Pass'(12.8%), suggesting that attempts to fix bugs frequently introduced regressions.
