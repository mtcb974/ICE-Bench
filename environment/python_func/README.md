# Multi-round evaluation suite based on BigCodeBench

The BigCodeBench multi-turn evaluation suite developed using FastApi, and it is evaluated locally.

## Quick Start

It is strongly recommended to create an independent environment to run the evaluation:
```bash
conda create -n bigcode-eval python=3.10
conda activate bigcode-eval
```

Install bigcodebench
```bash
pip install bigcodebench --upgrade
```

Install the evaluation environment of bigcodebench:
```bash
pip install -r ./Requirements/requirements-eval.txt
```

Run FastApi:
```bash
uvicorn main:app --reload --port 8199 
```
