# ICE-Bench

## About

ICE-Bench is a benchmark specifically designed to assess LLMs for Iterative Code gEneration under progressive requirement refinement scenarios.

This repository contains the data-construction pipeline, the benchmark dataset, the evaluation toolchain, and the experimental results. The released result tracks are described in the [Results](#results) section below.

## Setup

Ensure that [uv](https://github.com/astral-sh/uv/releases/tag/0.8.11) is installed. Then run the following command to set up the environment.

```bash
# Install dependency
uv sync
# Activate the virtual environment
source .venv/bin/activate
```

Set up api key in `.env` file:
```bash
mv .env.example .env
```

API Format
```bash
{API_PROVIDER}_API_KEY=...
{API_PROVIDER}_BASE_URL=...
```

Set up sandbox and repository environment: see `environment` directory.

## Dataset

All datasets are saved in the `./dataset` directory, and the directory structure is as follows:
```
├── java_function.jsonl
├── java_repository.jsonl
├── python_function.jsonl
└── python_repository.jsonl
```

The unified data format is:
```py
@dataclass
class MultiTurnDataInstance:
    hash_id: str    # As the unique identifier of data
    total_turn: int
    turn_datas: List[PerTurnDataInstance] # Each sample corresponds to data from multiple rounds.
    metadata: Dict[str,Any] # Store the information of the original dataset
    mt_id: Optional[int] = field(default=None)     # Used as the primary key of the database, it has no impact on the evaluation

@dataclass
class PerTurnDataInstance:  # A minimal evaluation unit
    turn_num: int
    instruction: str
    solution: Optional[str] = field(default=None)
    test: Optional[str] = field(default=None)
```

## Inference

For first-time users of this dataset, please run the following command to initialize the SQLite database:
```bash
PYTHONPATH=. python -m evaluation.setup_db
```

Please configure the parameters in `./script/run_inference.sh`, and then directly run this inference script:
```bash
bash script/run_inference.sh
```

Currently, the Python repository level still needs to rely on the original repository. Please go to `environment/python_repo`.

## Evaluation

For the function level, we support concurrent evaluation. Please use the following code and replace the parameters with those you need:

```python
from data_construction.data import data_manager
from dotenv import load_dotenv
from evaluation.eval import MTEvaluation
import asyncio

load_dotenv()
eval_module = MTEvaluation(
    llm_provider="",
    llm="",
    context_setting="", # base, golden
    prompt_setting="fh", # fh, edit, ci
    think_flag="qwen3-think", # For gpt-oss series with thinking mode, use `oss-think`, for qwen3 series with thinking mode, use `qwen3-think`
    lang="python", # python, java, java-repo
    run_id="default", # Just keep it unchanged; it is used for the output file name.
    max_workers=5,
    command="evaluation"
)
asyncio.run(eval_module.run_evaluation())
```

For repository-level Java tasks, if you want to use concurrency, please prepare multiple Docker containers by yourself. This is because there is a risk in concurrent file replacement operations. To reduce the slow speed caused by downloading mvn dependencies during the first visit, you can use the file in `data_construction/data/seed/mrgbench/docker_warmup.py` to warm up the docker before the evaluation.

After the environment is configured, you can run it using the following code:
```py
from data_construction.data import data_manager
from dotenv import load_dotenv
from evaluation.eval import MTEvaluation
llm_providers = ["deepinfra","deepinfra","deepinfra"]
llms = ["Qwen/Qwen3-Coder-480B-A35B-Instruct","openai/gpt-oss-20b","openai/gpt-oss-20b"]
think_flags = ["None","oss-think","None"]
for llm_provider, llm, think_flag in zip(llm_providers,llms, think_flags):
    for ct in ['base','golden']:
        for ps in ['fh','edit','ci']:
            if ct == 'golden' and ps == 'ci':
                continue
            eval_module = MTEvaluation(
                llm_provider=llm_provider,
                llm=llm,
                context_setting=ct,
                prompt_setting=ps,
                think_flag=think_flag,
                lang="java-repo",
                run_id="default",
                max_workers=1,
                command="evaluation"
            )
            eval_module.run_evaluation_for_repo()
```

Currently, the Python repository level still needs to rely on the original repository. Please go to `environment/python_repo`.

## Data Construction

Run multi-agent based requirement generation:
```bash
# python function level
PYTHONPATH=. python -m data_construction.pipeline --decomposition --dataset bigcodebench
# java function level
PYTHONPATH=. python -m data_construction.pipeline --decomposition --dataset autocodebench
# java repository level
PYTHONPATH=. python -m data_construction.pipeline --decomposition --dataset mrgbench
```

Run semantic-aware discriminative test case generation:
```bash
# python function level
PYTHONPATH=. python -m data_construction.pipeline --codegen --dataset bigcodebench --max_iter_number 10
# java function level
PYTHONPATH=. python -m data_construction.pipeline --codegen --dataset autocodebench --max_iter_number 10
# java repository level
PYTHONPATH=. python -m data_construction.pipeline --codegen --dataset mrgbench --max_iter_number 10
```

Other params:
- `--auto_skip`: skip human-in-the-loop
- `--init_ablation_study_db`: option for rq3, should run it first.
- `--skip_evaluator`: option for rq3
- `--skip_evaluator_and_distinctiveness`: option for rq3

## Results

This directory contains the released experimental results, numbered in the order of the corresponding Discussion subsections. Each track keeps the data needed to inspect the reported result.

### Main Experiment

`results/main_experiment/` contains the full model predictions, execution outputs, and per-cell metrics for every evaluation run of the main study and the RQ3 ablation study. `trajectories/` holds all 136 paper cells of RQ1 and RQ2 (Basic and Golden contexts, Full History / Code Edit / Cumulative Instruction prompting) as trajectory files, each recording the prompt, the generated solution, the execution status, and the execution detail for every turn, with the matching per-cell metric file alongside. `rq3/` holds the 16 ablation cells (wo Evaluator and wo Distinctiveness Validation & Evaluator) with the trajectory, evaluation, and metric files per cell.

### 1. Commercial-Model Subset (Discussion Section VI-A)

`results/1_commercial_model_subset/` contains the deterministic 40-task subset, all 40 final evaluation cells with their inference, evaluation, and metric files, the aggregate summary, and the per-turn accuracy figure.

### 2. Impact of the Number of Turns (Discussion Section VI-B)

`results/2_impact_of_turns/` contains the per-depth completion-rate summaries and the CR@$k$ figure.

### 3. Impact of Intermediate Outcomes (Discussion Section VI-C)

`results/3_intermediate_outcomes/` contains the per-subset and per-strategy summaries of how intermediate-turn performance relates to the final solution, the full trajectory-level outcomes, and the figure.

### 4. Manual Analysis of Source Tasks (Discussion Section VI-D)

`results/4_manual_review/` contains the questionnaire, three anonymized returns for 920 task turns, and the final majority and unanimous-agreement summary.

### 5. Failure Analysis (Discussion Section VI-E)

`results/5_failure_analysis/taxonomy/` contains the failure-attribution classification: the classification prompt, the final category proportions over the analysis population, and the paper-category mapping. The reported category proportions are computed from the primary category of each of the 9,779 model outputs; the secondary category is not mixed into the main distribution. `results/5_failure_analysis/failure_attribution/` contains the questionnaire, three anonymized returns for 180 outputs, and the final majority-accept and unanimous-agreement summary used to validate the attribution. The category-distribution figure is under `results/5_failure_analysis/figure/`, and the case-study figure is under `results/5_failure_analysis/case_study/`.

### 6. LLM-Based Evaluation (Discussion Section VI-F)

`results/6_judge_evaluation/judge/` contains the final cell, subset, and Judge-model metric tables. The Judge used GPT-5.6-Luna with temperature 0. The aggregate covers 63,102 matched turns from the 15,033 retained trajectories. `results/6_judge_evaluation/judge_reliability/` contains the questionnaire, three anonymized returns for 317 records, and the final human-human and Judge-human agreement statistics.

### 7. User Study (Discussion Section VI-G)

`results/7_user_study/` contains 80 sampled tasks, three anonymized rating files, the English questionnaire, and the final ordinal-agreement statistics.

The anonymized rating records are released for verification of the reported aggregation only. They do not identify individual participants.

## Release Boundary

This package contains no manuscript or response-letter source files. It excludes provider diagnostics, API-error and retry logs, staging snapshots, local test and interpreter caches, and unpublished intermediate scripts. Raw provider requests and replies behind the released aggregate results remain outside this public release.

## 🙏 Acknowledgement

- [BigcodeBench](https://github.com/bigcode-project/bigcodebench)
- [DevEval](https://github.com/seketeam/DevEval?tab=readme-ov-file)
- [AutoCodeBench](https://github.com/Tencent-Hunyuan/AutoCodeBenchmark/tree/main)
- [MRGBench](https://github.com/MRG-Bench/MRG-Bench/)
