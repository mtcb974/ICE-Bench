from pathlib import Path
import json
import subprocess
import psutil
from subprocess import run
from tqdm import tqdm
import os
import numpy as np
from argparse import ArgumentParser
import textwrap
from func_timeout import func_set_timeout
from func_timeout.exceptions import FunctionTimedOut
import func_timeout
from dataset import ShardDevEval,MTDevEvalInstance,DevEvalTurnResult,DevEvalMTResult
from dataset.util import load_dataset
from contextlib import ExitStack
from ast_util import add_test_function_to_file
from typing import List,Dict
import traceback
import datetime
from tqdm import tqdm
from collections import defaultdict

SOURCE_CODE_ROOT = Path('Source_Code')

def adjust_indent(code, new_indent):
    new_indent = new_indent - 4 if new_indent >= 4 else new_indent
    # remove original indentation
    dedented_code = textwrap.dedent(code)
    # add new indentation
    indented_code = textwrap.indent(dedented_code, ' ' * new_indent)
    return indented_code

def filter_irrelevant_information(output:str):
    return '\n'.join(
        line for line in output.split('\n') 
        if not 'DeprecatedConfig' in line and
           not 'DeprecatedInstaller' in line and 
           not 'warning' in line and 
           not 'fetch_build_eggs' in line and
           not '*************************' in line and
           not 'PEP 517' in line and not 'pep517' in line and not 'license_file' in line and not 'install_requires' in line and
           not 'setuptools' in line and not 'egg-info' in line  and not 'LICENSE' in line  and not 'MANIFEST' in line
    )

##### Verify the correctness of the code and return a status and detail. #####
def Setup_Replace_TEST(full_instance:MTDevEvalInstance,test_code:str):

    project_path = os.path.join(SOURCE_CODE_ROOT, full_instance['project_path'])
    test_selector = full_instance['tests'][0]
    test_file_path = os.path.join(project_path,test_selector.split("::")[0])

    head_tail = os.path.split(test_file_path)
    test_file_tmp_path = os.path.join(head_tail[0], 'tmp_' + head_tail[1])
    run(['cp', test_file_path, test_file_tmp_path])


    try:
        add_test_function_to_file(
            file_path=test_file_path,
            selector=test_selector,
            new_test_code=test_code,
        )
    except Exception as e:
        run(['mv', test_file_tmp_path, test_file_path])
        print(f"The test file has been rolled back: {test_file_path}")
        raise e

def Setup_Replace_GT(full_instance:MTDevEvalInstance, completion:str):
    completion = adjust_indent(completion, full_instance['indent'])

    completion_path = Path(full_instance['completion_path'])
    completion_path = os.path.join(SOURCE_CODE_ROOT, completion_path)
    head_tail = os.path.split(completion_path)
    completion_tmp_path = os.path.join(head_tail[0], 'tmp_' + head_tail[1])
    run(['cp', completion_path, completion_tmp_path])


    try:
        with open(completion_path, 'r') as f:
            file_lines = f.readlines()

        signature_line_no_1based = full_instance['signature_position'][0]
        current_line = signature_line_no_1based - 1
        
        decorator_lines = []
        while current_line > 0:
            prev_line = current_line - 1
            line_content = file_lines[prev_line].strip()
            
            if (line_content.startswith('@') or 
                (decorator_lines and not line_content.endswith(')'))):
                decorator_lines.insert(0, prev_line)
                current_line = prev_line
            else:
                break

        replace_start = decorator_lines[0] if decorator_lines else (signature_line_no_1based - 1)
        
        eos = full_instance['body_position'][1]
        
        file_lines = file_lines[:replace_start] + [completion, '\n'] + file_lines[eos:]
        with open(completion_path, 'w') as f:
            f.write(''.join(file_lines))
    except Exception as e:
        run(['mv', completion_tmp_path, completion_path])
        print(f"The code file has been rolled back: {completion_path}")
        raise e

@func_set_timeout(60)
def execution_test_selectors(full_instance:MTDevEvalInstance,tests:List[str]):
    project_path = os.path.join(SOURCE_CODE_ROOT, full_instance['project_path'])
    command = ['python', 'setup.py', 'pytest', '--addopts']
    results = {
        'status': 'pass', 
        'detail': {} # Key: selector, Value: detail
    }
    for test in tests:
        process = subprocess.Popen(
            command + [test],
            cwd=project_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
            )
        test_result = {
            'status': 'pass',
            'stdout': '',
            'stderr': '',
            'error': None
        }
        try:
            while True:
                process_id = process.pid
                process_memory = psutil.Process(process_id).memory_info().rss
                if process_memory > 5 * 1024 * 1024 * 1024: # 5GB memory usage per test
                    stdout, stderr = process.communicate()
                    test_result.update({
                        'status': 'OOM',
                        'error': 'Out of memory (exceeded 5GB limit)',
                    })
                    results['status'] = 'OOM'  # 
                    break
                return_code = process.poll()
                if return_code is not None:
                    stdout, stderr = process.communicate()
                    if return_code != 0:
                        test_result.update({
                            'status': 'Error',
                            'stdout': filter_irrelevant_information(stdout),
                            'stderr': filter_irrelevant_information(stderr),
                            'error': f"Return code: {return_code}"
                        })
                        results['status'] = 'Error'  # 
                    break
        except Exception as e:
            stdout, stderr = process.communicate()
            test_result.update({
                'status': 'Error',
                'error': f"Execution exception: {filter_irrelevant_information(str(e))}"
            })
            results['status'] = 'Error'
        finally:
            process.terminate()
        results['detail'][test] = test_result
    return results

def TearDown_GT(full_instance:MTDevEvalInstance):
    completion_path = Path(full_instance['completion_path'])
    completion_path = os.path.join(SOURCE_CODE_ROOT, completion_path)
    head_tail = os.path.split(completion_path)
    completion_tmp_path = os.path.join(head_tail[0], 'tmp_' + head_tail[1])
    run(['mv', completion_tmp_path, completion_path])
    print(f"rollback gt code: {completion_path}")

def TearDown_TEST(full_instance:MTDevEvalInstance):
    project_path = os.path.join(SOURCE_CODE_ROOT, full_instance['project_path'])
    test_selector = full_instance['tests'][0]
    test_file_path = os.path.join(project_path,test_selector.split("::")[0])
    head_tail = os.path.split(test_file_path)
    test_file_tmp_path = os.path.join(head_tail[0], 'tmp_' + head_tail[1])
    run(['mv', test_file_tmp_path, test_file_path])
    print(f"roll back test code: {test_file_path}")

def verify_llm_testgen(turn_llm_code:str,turn_llm_test:str,full_instance:MTDevEvalInstance,tests:List[str]):
    try:
        # Replace with the code generated by LLM. Back up the original code file.
        Setup_Replace_GT(full_instance=full_instance,completion=turn_llm_code)
        # Insert LLM-generated tests. Backup the test files.
        Setup_Replace_TEST(full_instance=full_instance,test_code=turn_llm_test)

        # Verify: Execute the test cases after replacement and obtain the execution feedback.
        feedback = execution_test_selectors(full_instance=full_instance,tests=tests)

        return feedback["status"],feedback["detail"]
    except Exception as e:
        print(f"Error during verify:{e}")
        traceback.print_exc()
        try:
            TearDown_GT(full_instance=full_instance)
        except:
            pass
        try:
            TearDown_TEST(full_instance=full_instance)
        except:
            pass
        raise e
    finally:
        try:
            TearDown_GT(full_instance=full_instance)
        except:
            pass
        try:
            TearDown_TEST(full_instance=full_instance)
        except:
            pass

def verify_llm_completion(turn_llm_code:str,full_instance:MTDevEvalInstance,turn:int):

    turn_llm_test = full_instance['mt'][turn-1]['test_code']
    tests = full_instance['mt'][turn-1]['tests']
    try:
        Setup_Replace_GT(full_instance=full_instance,completion=turn_llm_code)
        Setup_Replace_TEST(full_instance=full_instance,test_code=turn_llm_test)

        feedback = execution_test_selectors(full_instance=full_instance,tests=tests)

        return feedback["status"],feedback["detail"]
    except FunctionTimedOut as fe:
        print(f"OOT:{fe}")
        try:
            TearDown_GT(full_instance=full_instance)
        except:
            pass
        try:
            TearDown_TEST(full_instance=full_instance)
        except:
            pass
        return "TIMEOUT",str(fe)
    except Exception as e:
        print(f"Error:{e}")
        traceback.print_exc()
        try:
            TearDown_GT(full_instance=full_instance)
        except:
            pass
        try:
            TearDown_TEST(full_instance=full_instance)
        except:
            pass
        raise e
    finally:
        try:
            TearDown_GT(full_instance=full_instance)
        except:
            pass
        try:
            TearDown_TEST(full_instance=full_instance)
        except:
            pass

def compute_pass_at_k(n, c, k):
    """
    n: total number of completions per task
    c: number of completions that pass all tests
    k: k in pass_at_k
    """
    if n - c < k:
        return 1
    else:
        return 1.0 - np.prod(1.0 - k / np.arange(n-c+1, n+1))

def get_repo_grouth_truth(args):
    benchmark_data = {}
    with open(args.data_file, 'r') as f:
        for line in f:
            js = json.loads(line)
            namespace = js['namespace']
            benchmark_data[namespace] = js
    
    for data_key in benchmark_data.keys():
        data = benchmark_data[data_key]
        completion_path = Path(data['completion_path'])
        completion_path = os.path.join(args.source_code_root, completion_path)
        sos, eos = data['body_position'][0]-1, data['body_position'][1]
        sign_sos = data['signature_position'][0]-1
        # Read the code in completion_path from sign_sos to sign_sos + 1 and from sos to eos, and then assign it to data['gt']
        with open(completion_path,'r') as f:
            file_lines = f.readlines()
            signature = ''.join(file_lines[sign_sos:sign_sos+1])  
            body = ''.join(file_lines[sos:eos])  
            data['gt'] = signature + body  

    # Sort by the length of the body_position interval
    sorted_data = dict(sorted(benchmark_data.items(), 
                            key=lambda x: x[1]['body_position'][1] - x[1]['body_position'][0]))

    with open('ground_truth_temp.json','w') as f:
        json.dump(sorted_data, f, indent=2)
    
    with open('filter_SA_failcase_line10_cross_w_gt.jsonl', 'w') as f:
        for namespace, data in sorted_data.items():
            f.write(json.dumps(data) + '\n')

class EvalMetric:
    total_samples: int
    correct: int
    accuracy: float

    def __init__(self):
        self.total_samples = 0
        self.correct = 0
        self.accuracy = 0.0

class FullyCompleted:
    count: int
    total: int
    ratio: float

    def __init__(self):
        self.count = 0
        self.total = 0
        self.ratio = 0.0

class DetailResult:
    namespace :str
    turn: int
    status: str
    completion: str
    test_detail: str

class EvalReport:
    timestamp: str
    infer_file: str
    metrics: Dict[str,EvalMetric] # key : turn_1、turn_2
    fully_completed: FullyCompleted
    detailed_results: List[DetailResult]


def evaluate(infer_file:str,output_file:str,dataset:str):
    all_dataset:List[MTDevEvalInstance] = load_dataset(dataset)
    dataset_dict = {}
    for ds in all_dataset: 
        dataset_dict[ds['namespace']] = ds
    
    infer_results:List[DevEvalMTResult] = load_dataset(infer_file)

    report = {
        'timestamp': datetime.datetime.now().isoformat(),
        'infer_file': infer_file,
        'metrics': defaultdict(EvalMetric),
        'fully_completed': FullyCompleted(),
        'detailed_results': []
    }
    report['fully_completed'].total = len(infer_results)

    max_turn = max(len(res['completions']) for res in infer_results) if infer_results else 0
    for turn in range(1, max_turn + 1):
        report['metrics'][f'turn_{turn}'] = EvalMetric()
    
    with tqdm(total=len(infer_results), desc="Evaluating") as pbar:
        for infer_res in infer_results:
            namespace = infer_res['namespace']
            all_turns_passed = True

            for turn_infer_res in infer_res['completions']:
                turn:int = turn_infer_res['turn']
                completion:str = turn_infer_res['completion']
                status, test_detail = verify_llm_completion(
                    turn_llm_code=completion,
                    full_instance=dataset_dict[namespace],
                    turn=turn
                )

                turn_key = f'turn_{turn}'
                report['metrics'][turn_key].total_samples += 1
                if status == 'pass':
                    report['metrics'][turn_key].correct += 1
                else:
                    all_turns_passed = False

                report['detailed_results'].append({
                    'namespace': namespace,
                    'turn': turn,
                    'status': status,
                    'completion': completion,
                    'test_detail': test_detail
                })
            
            if all_turns_passed:
                report['fully_completed'].count += 1

            pbar.update(1) 
        
    for metric in report['metrics'].values():
        if metric.total_samples > 0:
            metric.accuracy = round(metric.correct / metric.total_samples,4)

    if report['fully_completed'].total > 0:
        report['fully_completed'].ratio = report['fully_completed'].count / report['fully_completed'].total

    print("\n" + "="*50)
    print("Final Evaluation Report Summary")
    print("-"*50)
    print(f"Total samples: {report['fully_completed'].total}")
    print(f"Fully completed: {report['fully_completed'].count} ({report['fully_completed'].ratio:.2%})")
    print("\nPer-turn accuracy:")
    for turn, metric in sorted(report['metrics'].items(), key=lambda x: int(x[0].split('_')[1])):
        print(f"  {turn}: {metric.accuracy:.2%} ({metric.correct}/{metric.total_samples})")
    print("="*50 + "\n")

    # Write the report to the output file
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2, default=lambda o: o.__dict__)
    
    return report

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--source_code_root', type=Path, default=Path('Source_Code'))
    # Inference file
    parser.add_argument('--infer_file', type=str,help="inference file") # data.jsonl
    parser.add_argument('--dataset', type=str,default='final_dataset.jsonl',help="original dataset") # data.jsonl

    args =  parser.parse_args()

    if args.infer_file.startswith('./'):
        raise ValueError("The Infer file should not start with./ and should be at the same level as the current folder.")
    args.output_file = f"eval_{args.infer_file}"

    evaluate(infer_file=args.infer_file,output_file=args.output_file,dataset=args.dataset)