from typing import TypedDict, List, Any, Dict
from data_construction.data import MTDataManager,MultiTurnDataInstance,PerTurnDataInstance
from pathlib import Path
import json
import hashlib

class DevEvalDependency(TypedDict):
    intra_class: List[str]
    intra_file: List[str]
    cross_file: List[str]

class DevEvalRequirement(TypedDict):
    Functionality: str
    Arguments: str

class ShardDevEval(TypedDict):
    turn: int
    requirement: str
    test_code: str
    tests: List[str]
    gt: str

class MTDevEvalInstance(TypedDict):
    # 原数据集
    namespace: str
    type: str
    project_path: str
    completion_path: str
    signature_position: List[int]
    body_position: List[int]
    dependency: DevEvalDependency
    requirement: DevEvalRequirement
    tests: List[str]
    indent: int
    # 后处理补充
    domain: str
    gt: str
    context: str
    test_codes: List[str] # 原始测试的字符串
    # 多轮相关新字段
    mt: List[ShardDevEval] # 分解后的数据集，包含: (分解时生成)turn,requirement (测试用例合成)test_code,tests[测试选择器],gt
    mt_tests: Dict[int,List[str]] # 多轮测试选择器字段，Key为turn, Value为测试选择器列表
    # 函数签名
    function_signature:str


if __name__ == '__main__':
    # 1：读取 Path(__file__).parent / mt_deveval.jsonl，拿到List[MTDevEvalInstance]
    data_path = Path(__file__).parent / "mt_deveval.jsonl"
    mt_deveval_instances: List[MTDevEvalInstance] = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                mt_deveval_instances.append(json.loads(line))
    print(f"成功读取多轮DevEval数据集，共{len(mt_deveval_instances)}条样本。")

    # 2. 插入到新数据
    manager = MTDataManager()

    for instance in  mt_deveval_instances:
        data = MultiTurnDataInstance(
            hash_id=instance["namespace"],
            total_turn=len(instance["mt"]),
            turn_datas=[],
            metadata={
                "source": "deveval",
                "task_id": instance["namespace"],
            }
        )
        for idx,t_data in enumerate(instance["mt"]):
            turn_data = PerTurnDataInstance(
                turn_num=idx+1,
                instruction=t_data["requirement"],
                solution=t_data["gt"],
                test=t_data["test_code"]
            )
            data.turn_datas.append(turn_data)
        
        manager.add(data)
        print(f"成功添加数据:{instance["namespace"]}")
    
    