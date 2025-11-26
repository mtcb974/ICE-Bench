
## Results

Note: The one-shot setting measures the pass@1 metric, while the iterative setting uses the complete rate metric, defined as "the proportion of tasks where the model successfully completes all required turns."

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
