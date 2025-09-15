from typing import Iterator
from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
from datasets import load_dataset
from pathlib import Path
from typing import Iterator, Dict, Any, Set, Optional, List
import hashlib
import random
import pandas as pd

@dataclass
class StandardSeedSample:
    """标准种子数据结构"""
    instruction: str
    solution: str
    test: str
    metadata: Dict[str,Any] = None
    hash_id: Optional[str] = None    # use for checkpoint


class SeedDataReader(ABC):
    """
    抽象基类：种子数据读取器
    """
    def __init__(self,seed_data_path:Optional[str] = None) -> None:
        # 种子数据集路径
        self.seed_data_path = seed_data_path

    @abstractmethod
    def _standardize_sample(self,raw_sample:Dict[str,Any]) -> StandardSeedSample:
        ...
    
    @abstractmethod
    def read_from_jsonl_with_filter(self,ids:Set[str]) -> Iterator[StandardSeedSample]:
        """
        从jsonl文件中加载种子数据,由子类实现
        当样本中问题字段的哈希值在ids里面时，跳过这个样本
        
        Parmas:
            ids: 字符串集合，即hash_id。

        Usage:
            for item in read_from_jsonl_with_filter():
                print(item)
        """
        ...
    
    def read_from_jsonl_random_sample(self, sample_size: int = 5, random_seed: Optional[int] = 42) -> List[StandardSeedSample]:
        """
        从jsonl文件中随机采样指定数量的种子数据
        
        Args:
            sample_size: 采样数量，默认为5
            random_seed: 随机种子，默认为42，用于确保采样的可复现性
            
        Returns:
            包含指定数量样本的列表
            
        Usage:
            samples = read_from_jsonl_random_sample(sample_size=10, random_seed=123)
            for sample in samples:
                print(sample.instruction)
        """
        # 设置随机种子确保可复现性
        if random_seed is not None:
            random.seed(random_seed)
        
        path = Path(self.seed_data_path)
        if not path.exists():
            raise ValueError(f"seed_data_path: {path} 不存在")
        if not path.is_file():
            raise ValueError(f"seed_data_path: {path} 不是一个文件")
        
        # 首先读取所有有效的样本
        all_samples = []
        with path.open('r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw_sample = json.loads(line)
                    sample = self._standardize_sample(raw_sample)
                    all_samples.append(sample)
                except json.JSONDecodeError as e:
                    print(f"JSON 解析错误: {e}，行内容: {line}")
        
        # 如果样本总数少于请求的采样数，返回所有样本
        if len(all_samples) <= sample_size:
            return all_samples
        
        # 随机采样指定数量的样本
        sampled_samples = random.sample(all_samples, sample_size)
        return sampled_samples
    
    def read_from_jsonl(self) -> Iterator[dict]:
        """
        从jsonl文件中加载种子数据，返回一个生成器
        
        Usage:
            for item in read_from_jsonl():
                print(item)
        """
        path = Path(self.seed_data_path)

        if not path.exists():
            raise ValueError(f"seed_data_path: {path} 不存在")
        if not path.is_file():
            raise ValueError(f"seed_data_path: {path} 不是一个文件")
        
        with path.open('r',encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line:
                    try:
                        yield  self._standardize_sample(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"JSON 解析错误: {e}，行内容: {line}")

    def read_from_hf(self,hf_dataset_name:str,split:str="test") ->Iterator[dict]:
        """
        从HuggingFace中读取数据集，保存为jsonl文件。
        返回一个逐行读取的生成器
        """
        save_dir = (Path(__file__).parent / 'seed').resolve()
        save_dir.mkdir(parents=True, exist_ok=True)
        
        safe_name = hf_dataset_name.replace('/', '_')
        save_path = save_dir / f"{safe_name}.jsonl"

        print(f"正在从 Hugging Face 加载数据集: {hf_dataset_name}")
        ds = load_dataset(hf_dataset_name)

        if split not in ds:
            raise ValueError(f"数据集 '{hf_dataset_name}' 中不存在 split: '{split}'")
        
        print(f"正在将 '{split}' split 保存为: {save_path}")
        ds[split].to_json(save_path,lines=True)
        print(f"数据已保存到: {save_path}")

        # 返回一个逐行读取的生成器
        def _generator():
            with save_path.open('r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            yield self._standardize_sample(json.loads(line))
                        except json.JSONDecodeError as e:
                            print(f"第 {line_num} 行 JSON 解析失败: {e} -> {line[:100]}...")
        return _generator()

    def _generate_hash_id(self, input: str) -> str:
        """
        为长字符串生成高效哈希ID，适合checkpoint场景
        
        参数:
            input: 输入的长字符串（如1000字符）
        返回:
            24位十六进制哈希字符串
        """
        # 转换为字节流
        byte_data = input.encode('utf-8')
        
        # 使用SHA-1算法（比SHA-256更快，对1000字符足够）
        # 取前24位（12字节）平衡唯一性和长度
        return hashlib.sha1(byte_data).hexdigest()[:24]
    
class BigCodeBenchReader(SeedDataReader):
    def __init__(self, seed_data_path: Optional[str] = None) -> None:
        bigcodebench_path = (Path(__file__).parent / 'seed' / 'bigcodebench_hard.jsonl' ).resolve()
        self.seed_data_path = seed_data_path or bigcodebench_path
    
    def _standardize_sample(self, raw_sample: Dict[str, Any]) -> StandardSeedSample:
        """将BigCodeBench的数据结构转化为标准数据结构，方便处理"""
        return StandardSeedSample(
            hash_id=self._generate_hash_id(raw_sample['instruct_prompt']),
            instruction=raw_sample['instruct_prompt'],
            solution=raw_sample['canonical_solution'],
            test=raw_sample['test'],
            metadata= {
                'source': 'bigcodebench',
                'task_id': raw_sample['task_id'],
                'libs': raw_sample['libs']
            }
        )

    def read_from_jsonl_with_filter(self, ids: Set[str]) -> Iterator[StandardSeedSample]:
        """
        从jsonl文件中加载种子数据，过滤掉hash_id在ids集合中的样本
        """
        path = Path(self.seed_data_path)
        if not path.exists():
            raise ValueError(f"seed_data_path: {path} 不存在")
        if not path.is_file():
            raise ValueError(f"seed_data_path: {path} 不是一个文件")
        with path.open('r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw_sample = json.loads(line)
                    hash_id = self._generate_hash_id(raw_sample['instruct_prompt'])
                    if hash_id in ids:
                        continue
                    yield self._standardize_sample(raw_sample)
                except json.JSONDecodeError as e:
                    print(f"JSON 解析错误: {e}，行内容: {line}")

class AutoCodeBenchReader(SeedDataReader):
    def __init__(self, seed_data_path: Optional[str] = None) -> None:
        autocodebench_path = (Path(__file__).parent / 'seed' / 'autocodebench'/ 'java_autocodebench.jsonl' ).resolve()
        self.seed_data_path = seed_data_path or autocodebench_path

    
    def _standardize_sample(self, raw_sample: Dict[str, Any]) -> StandardSeedSample:
        """将AutoCodeBench的数据结构转化为标准数据结构，方便处理"""
        return StandardSeedSample(
            hash_id=self._generate_hash_id(raw_sample['question']),
            instruction=raw_sample['question'],
            solution=raw_sample['canonical_solution'],
            test=raw_sample['full_test_func'],
            metadata= {
                'source': 'autocodebench',
                'language': raw_sample['language'],
                'difficulty': raw_sample['difficulty']
            }
        )

    def read_from_jsonl_with_filter(self, ids: Set[str]) -> Iterator[StandardSeedSample]:
        """
        从jsonl文件中加载种子数据，过滤掉hash_id在ids集合中的样本
        """
        path = Path(self.seed_data_path)
        if not path.exists():
            raise ValueError(f"seed_data_path: {path} 不存在")
        if not path.is_file():
            raise ValueError(f"seed_data_path: {path} 不是一个文件")
        with path.open('r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw_sample = json.loads(line)
                    hash_id = self._generate_hash_id(raw_sample['question'])
                    if hash_id in ids:
                        continue
                    yield self._standardize_sample(raw_sample)
                except json.JSONDecodeError as e:
                    print(f"JSON 解析错误: {e}，行内容: {line}")

class MRGBenchReader(SeedDataReader):
    def __init__(self, seed_data_path: str | None = None) -> None:
        mrgbench_path = (Path(__file__).parent / 'seed' / 'mrgbench' / 'v3-MRGBench.json').resolve()
        self.seed_data_path = seed_data_path or mrgbench_path
    
    def read_from_jsonl_with_filter(self, ids: Set[str]) -> Iterator[StandardSeedSample]:
        """
        从json文件中加载种子数据，过滤掉hash_id在ids集合中的样本。
        注意：此函数现在读取的是一个包含所有样本的json数组文件，而不是jsonl格式。
        """
        path = Path(self.seed_data_path)
        if not path.exists():
            raise ValueError(f"seed_data_path: {path} 不存在")
        if not path.is_file():
            raise ValueError(f"seed_data_path: {path} 不是一个文件")
        with path.open('r', encoding='utf-8') as file:
            try:
                data = json.load(file)
                if not isinstance(data, list):
                    raise ValueError(f"文件内容不是数组格式: {path}")
                for raw_sample in data:
                    hash_id = self._generate_hash_id(raw_sample['instruction'])
                    if hash_id in ids:
                        continue
                    yield self._standardize_sample(raw_sample)
            except json.JSONDecodeError as e:
                print(f"JSON 解析错误: {e}，文件路径: {path}")

    def _standardize_sample(self, raw_sample: Dict[str, Any]) -> StandardSeedSample:
        """将AutoCodeBench的数据结构转化为标准数据结构，方便处理"""
        return StandardSeedSample(
            hash_id=self._generate_hash_id(raw_sample['instruction']),
            instruction=raw_sample['instruction'],
            solution=raw_sample['func'],
            test=raw_sample['test_code'],
            metadata= {
                'source': 'mrgbench',
                'func': raw_sample["func"],
                'repo': raw_sample["project"],
                'file_path': raw_sample["file_path"],
                'func_name': raw_sample["func_name"],
                'context': raw_sample["context"],
                'func_start': raw_sample["func_start"],
                'func_end': raw_sample["func_end"],
                'body_len': raw_sample["body_len"],
                'test_file': raw_sample["test_file"],
                'test_start': raw_sample["test_start"],
                'test_end': raw_sample["test_end"],
                'test_code': raw_sample["test_code"],
                'test_instruction': raw_sample["test_instruction"],
                'language': 'java',
            }
        )