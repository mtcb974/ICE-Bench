import json
from typing import Dict, List

def analyze_mt_data(file_path: str):
    total_instances = 0
    total_turns = 0
    max_turns = 0
    min_turns = float('inf')
    turns_distribution: Dict[int, int] = {}  # key: turn count, value: number of instances
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            total_instances += 1
            data = json.loads(line)
            mt_length = len(data.get('mt', []))
            
            total_turns += mt_length
            
            if mt_length > max_turns:
                max_turns = mt_length
            if mt_length < min_turns:
                min_turns = mt_length
                
            turns_distribution[mt_length] = turns_distribution.get(mt_length, 0) + 1
    
    avg_turns = total_turns / total_instances if total_instances > 0 else 0
    
    print(f"Total number of data: {total_instances}")
    print(f"Total number of turns: {total_turns}")
    print(f"Average number of turns per sample: {avg_turns:.2f}")
    print(f"Maximum number of turns: {max_turns}")
    print(f"Minimum number of turns: {min_turns}")
    print("\nTurn distribution statistics:")
    for turn_count in sorted(turns_distribution.keys()):
        print(f"Number of samples with {turn_count} turns: {turns_distribution[turn_count]}")

if __name__ == "__main__":
    file_path = "final_dataset.jsonl"
    analyze_mt_data(file_path)