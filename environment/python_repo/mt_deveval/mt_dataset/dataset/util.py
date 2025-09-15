from typing import List
import json

def load_dataset(file_path:str) -> List:
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
    return data


def load_context_map(context_source_file:str="./Experiments/prompt/LM_prompt_elements.jsonl"):
    context_map = {}
    try:
        with open(context_source_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    if 'namespace' in data and 'contexts_above' in data:
                         context_map[data['namespace']] = data['contexts_above']
                    else:
                         print(f"Warning: Skip the data with missing 'namespace' or 'contexts_above' fields: {line.strip()}")
        print(f"Read {len(context_map)} context mappings from {context_source_file}.")
        return context_map
    except FileNotFoundError:
        print(f"Error: File {context_source_file} not found")
        return
    except json.JSONDecodeError:
         print(f"Error: Failed to parse file {context_source_file}, please check the file format.")
         return
    