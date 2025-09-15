from typing import Literal,List,Set,Dict,Any
import json
import os

### 推理结果jsonl ###
class InferenceTurnResult:
    def __init__(self, turn_number: int, prompt: str, solution: str, raw_solution: str, prompt_tokens: int, completion_tokens: int, total_tokens: int):
        self.turn_number = turn_number
        self.prompt = prompt
        self.solution = solution
        self.raw_solution = raw_solution
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


    @classmethod
    def from_dict(cls, data):
        """
        支持从dict或InferenceTurnResult对象创建实例，避免AttributeError
        """
        if isinstance(data, cls):
            # 已经是InferenceTurnResult对象，直接返回
            return data
        # 否则假定是dict
        return cls(
            turn_number=data.get("turn_number"),
            prompt=data.get("prompt"),
            solution=data.get("solution"),
            raw_solution=data.get("raw_solution"),
            prompt_tokens=data.get("prompt_tokens"),
            completion_tokens=data.get("completion_tokens"),
            total_tokens=data.get("total_tokens")
        )

    def to_dict(self):
        return {
            "turn_number": self.turn_number,
            "prompt": self.prompt,
            "solution": self.solution,
            "raw_solution": self.raw_solution,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens
        }

class InferenceResult:
    def __init__(self, hash_id: str, mt_id: str, language: str, solutions: List[InferenceTurnResult], prompt_tokens: int, completion_tokens: int, total_tokens: int):
        self.hash_id = hash_id
        self.mt_id = mt_id
        self.language = language
        self.solutions = solutions
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens

    @classmethod
    def from_dict(cls, data):
        """
        支持从dict或InferenceResult对象创建实例，避免AttributeError
        """
        if isinstance(data, cls):
            return data
        # 兼容solutions为对象或dict的情况
        solutions = [InferenceTurnResult.from_dict(item) for item in data.get("solutions", [])]
        return cls(
            hash_id=data.get("hash_id"),
            mt_id=data.get("mt_id"),
            language=data.get("language"),
            solutions=solutions,
            prompt_tokens=data.get("prompt_tokens"),
            completion_tokens=data.get("completion_tokens"),
            total_tokens=data.get("total_tokens")
        )

    def to_dict(self):
        return {
            "hash_id": self.hash_id,
            "mt_id": self.mt_id,
            "language": self.language,
            "solutions": [s.to_dict() for s in self.solutions],
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens
        }

class InferenceResultJsonlManager:
    """
    用于管理推理结果jsonl文件的类
    """
    def __init__(self, jsonl_path):
        self.jsonl_path = str(jsonl_path)
        # 如果文件不存在，则创建一个空的jsonl文件
        if not os.path.exists(self.jsonl_path):
            with open(self.jsonl_path, 'w', encoding='utf-8') as f:
                pass  # 创建一个空文件

    def append(self, result: InferenceResult):
        """追加一条数据"""
        with open(self.jsonl_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result.to_dict(), ensure_ascii=False) + '\n')

    def append_batch(self, results: List[InferenceResult]):
        """追加一批数据"""
        with open(self.jsonl_path, 'a', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result.to_dict(), ensure_ascii=False) + '\n')

    def read_all(self) -> List[InferenceResult]:
        """读取所有数据，返回InferenceResult的List"""
        results = []
        try:
            with open(self.jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    results.append(InferenceResult.from_dict(data))
        except FileNotFoundError:
            print("❌文件没有找到!")
            # 如果文件不存在，则创建一个空文件
            with open(self.jsonl_path, 'w', encoding='utf-8') as f:
                pass
        return results

    def get_all_hash_ids(self) -> Set[str]:
        """获取所有hash_id，返回一个集合"""
        hash_ids = set()
        try:
            with open(self.jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    hash_id = data.get("hash_id")
                    if hash_id is not None:
                        hash_ids.add(hash_id)
        except FileNotFoundError:
            print("❌文件没有找到")
            # 如果文件不存在，则创建一个空文件
            with open(self.jsonl_path, 'w', encoding='utf-8') as f:
                pass
        return hash_ids
