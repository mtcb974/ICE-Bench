import json
import re
from typing import List,Dict,Any

def clean_markdown_code_block(output: str) -> str:
    return re.sub(r'^```json\s*|\s*```$', '', output, flags=re.MULTILINE)

def extract_code_block(text):
    pattern = r'```(?:\w+)?\n(.*?)```'
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    else:
        return None

def parse_json_output(output: str) -> List[Dict[str, Any]]:
    try:
        raw_data = json.loads(output)
        if not isinstance(raw_data, list):
            raise ValueError("Model output should be a JSON array.")
        return raw_data
    except json.JSONDecodeError as e:
        print(f"json serialization error, model output:\n\n{output}")
        raise e
    
def extract_python(code):
    pattern = r"```python\s*([\s\S]*?)\s*```"
    match = re.search(pattern, code, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    else:
        return code