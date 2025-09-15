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
import func_timeout
from mt_dataset.dataset import ShardDevEval,MTDevEvalInstance
from mt_dataset.ast_util import add_test_function_to_file
from typing import List
import traceback
import json

SOURCE_CODE_ROOT = Path('Source_Code')

def adjust_indent(code, new_indent):
    new_indent = new_indent - 4 if new_indent >= 4 else new_indent
    # remove original indentation
    dedented_code = textwrap.dedent(code)
    # add new indentation
    indented_code = textwrap.indent(dedented_code, ' ' * new_indent)
    return indented_code

def get_signature(full_instance):

    # Find file path, backup original file
    completion_path = Path(full_instance['completion_path'])
    completion_path = os.path.join(SOURCE_CODE_ROOT, completion_path)

    with open(completion_path, 'r') as f:
        file_lines = f.readlines()
    sos = full_instance['signature_position'][0]-1
    eos = full_instance['signature_position'][1]
    # Replace all content from decorator (or function signature) to function body end
    while file_lines[sos-1].strip().startswith('@'):
        sos -= 1
        
    file_lines = file_lines[sos:eos]
    return file_lines
    # with open(completion_path, 'w') as f:
    #     f.write(file_lines + '\n')

if __name__ == '__main__':
    """
    Read the dataset jsonl file, do some preprocessing
    """

    # Read the dataset    
    with open('final_dataset.jsonl', 'r', encoding='utf-8') as f:
        data_list = [json.loads(line) for line in f]
    
    # Find function signature
    with open('./signature.jsonl','w') as f:
        for data in data_list:
            signature = get_signature(data)
            data['function_signature'] = ''.join(signature)
            f.write(json.dumps(data,ensure_ascii=False) + '\n')
    
    with open('./signature2.txt','w') as f:
        for data in data_list:
            signature = get_signature(data)
            f.write(''.join(signature) + '\n')
