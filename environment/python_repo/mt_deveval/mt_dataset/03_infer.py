from typing import List
from dataset.util import load_dataset
from dataset import MTDevEvalInstance,DevEvalTurnResult,DevEvalMTResult
from model.prompt.inference import DEVEVAL_DIRECT_TURN1,DEVEVAL_DIRECT_TURNK,DEVEVAL_EDIT_TURNK
from model import call_llm_inference
import argparse
from util import extract_python
import json
import os
from typing import Dict,Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm

def get_output_path_and_create(llm:str,golden:bool,qwen3_thinking_mode:bool, prompt_mode:str) -> str:

    golden_str = "golden" if golden else "base"
    think_str = "_think" if qwen3_thinking_mode else ""
    prompt_mode = "_"+prompt_mode if prompt_mode != "direct" else ""
    llm = llm.replace("/","_")
    llm = llm.replace(":","_")

    output_file = f"./mt_deveval_result_{llm}_{golden_str}{think_str}{prompt_mode}.jsonl"
    if not os.path.exists(output_file):
        try:
            with open(output_file, 'w') as f:
                pass
            print(f"Result file {output_file} does not exist, created for you")
        except OSError as e:
            print(f"Failed to create file '{output_file}': {e}")
    else:
        print(f"Result file '{output_file}' already exists")
    return output_file

def get_infer_prompt(
        trajectories:List[DevEvalTurnResult],
        requirement:str,
        context:str,
        # setting
        golden:bool,
        mt_data:Optional[MTDevEvalInstance],
        # prompt mode
        qwen3_thinking_mode:bool,
        prompt_mode:str,
        llm: str
    )->List[Dict[str,str]]:

    if golden == True and mt_data is None:
        raise ValueError("If golden is true, then user must provide mt_data")
    
    messages = []
    if not qwen3_thinking_mode and ('Qwen' in llm or 'qwen' in llm):
        messages.append({"role":"system","content": "/no_think"})
    # Full History
    if prompt_mode == 'direct':
        trajectories_len = len(trajectories)
        if trajectories_len > 0:
            task_prompt = DEVEVAL_DIRECT_TURNK.format(requirement=requirement)
            for idx,traj in enumerate(trajectories):
                if idx == 0:
                    # turn 1
                    user_msg = {"role": "user","content": DEVEVAL_DIRECT_TURN1.format(contexts_above=context,requirement=traj["requirement"],function_signature=mt_data['function_signature'])}
                else:
                    # turn k
                    user_msg = {"role": "user","content": DEVEVAL_DIRECT_TURNK.format(requirement=traj["requirement"])}
                if golden:
                    assistant_msg = {"role": "assistant", "content": f"```python\n{mt_data['mt'][idx]['gt']}```"}
                else:
                    assistant_msg = {"role": "assistant", "content": traj["raw_completion"]}
                messages.append(user_msg)
                messages.append(assistant_msg)
            messages.append({"role": "user", "content": task_prompt})
        else:
            task_prompt = DEVEVAL_DIRECT_TURN1.format(contexts_above=context,requirement=requirement,function_signature=mt_data['function_signature'])
            messages.append({"role": "user", "content": task_prompt})
    # Code Edit
    elif prompt_mode == 'edit':
        trajectories_len = len(trajectories)
        if trajectories_len > 0:
            if golden:
                previous_code = mt_data['mt'][trajectories_len-1]['gt']
            else:
                previous_code = trajectories[-1]["completion"]

            task_prompt = DEVEVAL_EDIT_TURNK.format(contexts_above=context,previous_code=previous_code,requirement=requirement)
            messages.append({"role": "user", "content": task_prompt})
        else:
            task_prompt = DEVEVAL_DIRECT_TURN1.format(contexts_above=context,requirement=requirement,function_signature=mt_data['function_signature'])
            messages.append({"role": "user", "content": task_prompt})
    # Cumulative
    elif prompt_mode == 'append':
        if golden:
            raise ValueError("Append Mode do not support golden")
        requirements = ""
        trajectories_len = len(trajectories)
        for idx, traj in enumerate(trajectories):
            requirements = requirements + '\n' + traj["requirement"].strip()
        
        requirements  = requirement + '\n' + requirements
        task_prompt = DEVEVAL_DIRECT_TURN1.format(contexts_above=context,requirement=requirements,function_signature=mt_data['function_signature'])
        messages.append({"role": "user", "content": task_prompt})
    else:
        raise ValueError("Unsupported prompt mode")
    return messages
    
def process_single_data(data, llm, backend, golden, qwen3_thinking_mode, prompt_mode, output_file, lock):
    infer_result: dict = {
        'namespace': data['namespace'],
        'completions': []
    }
    trajectories: List[DevEvalTurnResult] = []

    for turn_data in data['mt']:
        turn_result: DevEvalTurnResult = {
            "turn": turn_data['turn'],
            "requirement": turn_data["requirement"],
            "messages": [],
            "raw_completion": "",
            "completion": ""
        }
        print(f"Start infer {data['namespace']} in turn {turn_data['turn']}")
        messages = get_infer_prompt(trajectories=trajectories,
                                  requirement=turn_data["requirement"],
                                  context=data["context"],
                                  golden=golden,
                                  mt_data=data,
                                  qwen3_thinking_mode=qwen3_thinking_mode,
                                  prompt_mode=prompt_mode, llm=llm)
        raw_result = call_llm_inference(llm=llm, messages=messages, backend=backend)
        turn_result["messages"] = messages
        turn_result['raw_completion'] = raw_result
        turn_result['completion'] = extract_python(raw_result)
        print(f"Finish infer {data['namespace']} in turn {turn_data['turn']}")
        trajectories.append(turn_result)
    
    infer_result["completions"] = trajectories
    
    with lock:
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(infer_result, ensure_ascii=False) + '\n')
    
    return data['namespace']

def run_multi_turn_infer(
    dataset: str,
    llm: str,
    backend: str,
    golden: bool,
    qwen3_thinking_mode: bool,
    prompt_mode: str,
):
    datas:List[MTDevEvalInstance] = load_dataset(dataset)
    print(f"Running Multi-turn Inference with total inference instance: {len(datas)}.")
    
    output_file = get_output_path_and_create(llm=llm,golden=golden,qwen3_thinking_mode=qwen3_thinking_mode,prompt_mode=prompt_mode)
    processed_namespaces = set()
    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            result = json.loads(line)
            processed_namespaces.add(result['namespace'])
    print(f"Skipped processed samples: {len(processed_namespaces)}")
    
    filtered_datas = [data for data in datas if data['namespace'] not in processed_namespaces]
    print(f"Samples to process: {len(filtered_datas)}")
    
    lock = threading.Lock()
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for data in filtered_datas:
            future = executor.submit(
                process_single_data,
                data=data,
                llm=llm,
                backend=backend,
                golden=golden,
                qwen3_thinking_mode=qwen3_thinking_mode,
                prompt_mode=prompt_mode,
                output_file=output_file,
                lock=lock
            )
            futures.append(future)
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Completed inference samples"):
            namespace = future.result()
            print(f"Completed processing: {namespace}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",type=str,default="./final_dataset.jsonl")
    parser.add_argument("--backend",type=str,default="openai")
    parser.add_argument("--llm",type=str,default="deepseek-v3")
    parser.add_argument("--golden",action="store_true")
    parser.add_argument("--prompt_mode",default="direct",choices=["direct","edit","append"],help="Prompt mode")
    parser.add_argument("--qwen3_thinking_mode",action="store_true")
    args = parser.parse_args()

    run_multi_turn_infer(
        dataset=args.dataset,
        backend=args.backend,
        llm=args.llm,
        golden=args.golden,
        prompt_mode=args.prompt_mode,
        qwen3_thinking_mode=args.qwen3_thinking_mode
    )