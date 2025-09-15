# Repository environment for Python tasks

Same as DevEval,before running the evaluation, researchers need to download the repositories, and dependency data.

The original repositories can be downloaded from [Link](https://zenodo.org/records/15580764). Users need to uncompressed the repositories and put them in the directory (e.g., mt_deveval/Source_Code).

## Environment Setup

```bash
cd mt_deveval

# create evaluation environment
conda create --name mt_deveval --file environment.txt
conda activate mt_deveval
# install dependency
pip install -r requirement.txt

# replace the path with your own path
echo "export NLTK_DATA=/home/user/mt_deveval/nltk_data" >> ~/.bashrc
source ~/.bashrc
```

## Inference with llm

```bash
python 03_infer.py --backend deepseek --llm deepseek-v3 --prompt_mode direct
```

Params:
- `dataset`: You also can specify dataset.
- `llm`: Specify your llm.
- `golden`: Whether to use the correct answer as historical information
- `backend`: Your Api backend
- `qwen3_thinking_mode`: Whether the Qwen3 model has activated the thinking mode
- `prompt_mode`: **direct** for `Full History`, **edit** for `Code Edit`, **append** for `Cumulative instruction`

## Evaluation

```bash
python pass_k.py --infer_file your_infer_file
```
