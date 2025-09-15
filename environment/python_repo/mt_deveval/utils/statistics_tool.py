import json
from typing import List, Dict,TypedDict

class ShardDevEval(TypedDict):
    turn: int
    requirement: str
    test_code: str
    tests: List[str]
    gt: str

class MTDevEvalInstance(TypedDict):
    namespace: str
    type: str
    project_path: str
    completion_path: str
    signature_position: List[int]
    body_position: List[int]
    dependency: Dict
    requirement: Dict
    tests: List[str]
    indent: int
    domain: str
    gt: str
    context: str
    test_codes: List[str]
    mt: List[ShardDevEval]
    mt_tests: Dict[int, List[str]]
    function_signature: str

def analyze_dataset(file_path: str):
    total_instructions = 0
    total_test_cases = 0
    total_requirement_lines = 0
    total_requirement_chars = 0
    total_gt_lines = 0
    total_gt_chars = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data: MTDevEvalInstance = json.loads(line)
                if not data.get('mt'):
                    continue
                    
                for instruction in data['mt']:
                    total_instructions += 1
                    
                    # Count test cases
                    test_cases = len(instruction.get('tests', []))
                    total_test_cases += test_cases
                    
                    # Count requirement lines and characters (include context)
                    req = instruction.get('requirement', '')
                    context = data.get('context', '')
                    combined_req = f"{context}\n{req}" if context else req
                    req_lines = len(combined_req.splitlines())
                    req_chars = len(combined_req)
                    total_requirement_lines += req_lines
                    total_requirement_chars += req_chars
                    
                    # Count gt lines and characters
                    gt = instruction.get('gt', '')
                    gt_lines = len(gt.splitlines())
                    gt_chars = len(gt)
                    total_gt_lines += gt_lines
                    total_gt_chars += gt_chars
                    
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                continue
    
    if total_instructions == 0:
        print("No instructions found in the dataset.")
        return
    
    avg_test_cases = total_test_cases / total_instructions
    avg_req_lines = total_requirement_lines / total_instructions
    avg_req_chars = total_requirement_chars / total_instructions
    avg_gt_lines = total_gt_lines / total_instructions
    avg_gt_chars = total_gt_chars / total_instructions
    
    print(f"Total instructions analyzed: {total_instructions}")
    print("\nAverage per instruction:")
    print(f"- Test cases: {avg_test_cases:.2f}")
    print(f"- Requirement lines: {avg_req_lines:.2f}")
    print(f"- Requirement characters: {avg_req_chars:.2f}")
    print(f"- GT lines: {avg_gt_lines:.2f}")
    print(f"- GT characters: {avg_gt_chars:.2f}")

if __name__ == '__main__':
    analyze_dataset('./final_dataset.jsonl')