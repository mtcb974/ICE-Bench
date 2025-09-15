from dotenv import load_dotenv
import argparse
from typing import Literal,List
from pathlib import Path
from .service.decomposition import DecompositionModule
from .service.codegen import CodegenModule

load_dotenv()

def decomposition(dataset:str = "bigcodebench",max_iter_number: int = 5):
    decomposition_module = DecompositionModule(dataset=dataset,
                                               max_iter_number=max_iter_number)
    decomposition_module.execute_decompostion()

def codegen(dataset:str = "bigcodebench",
            max_iter_number:int = 8,
            auto_skip:bool=False,
            exclude_module: List[Literal["evaluator","distinctiveness"]] = []):
    # 判断是否做消融实验
    if len(exclude_module) == 2:
        db_path = (Path(__file__).parent / 'data' / f'ablation_study_evaluator_distinctiveness.db').resolve()
        log_db_path = (Path(__file__).parent / 'data' / f'log_ablation_study_evaluator_distinctiveness.db').resolve()
    elif len(exclude_module) == 1:
        db_path = (Path(__file__).parent / 'data' / f'ablation_study_evaluator.db').resolve()
        log_db_path = (Path(__file__).parent / 'data' / f'log_ablation_study_evaluator.db').resolve()
    
    if len(exclude_module) > 0:
        module = CodegenModule(dataset=dataset,
                            max_iter_number=max_iter_number,
                            exclude_module=exclude_module,
                            auto_skip=auto_skip,
                            db_path=db_path,
                            log_db_path=log_db_path)
        # 对于消融实验，开多线程快速执行
        module.execute_codegen_loop()
    else:
        module = CodegenModule(dataset=dataset,
                            max_iter_number=max_iter_number,
                            exclude_module=exclude_module,
                            auto_skip=auto_skip)
        module.execute_codegen_loop()

def get_mt_data_info():
    from data_construction.data.data_manager import MTDataManager
    from data_construction.data.data_reader import BigCodeBenchReader
    data_manager = MTDataManager()
    reader = BigCodeBenchReader()

    # 统计分解有问题的样本信心
    ids = data_manager.get_all_hashids()
    iter = reader.read_from_jsonl_with_filter(ids)
    undecomposed_task_ids = []
    for data in iter:
        undecomposed_task_ids.append(data.metadata["task_id"])


    print(f"当前一共有数据:{data_manager.count()}条")
    # bigcodebench中有多少没处理的
    print(f"未分解的指令一共有: {len(undecomposed_task_ids)}条，分别是:{undecomposed_task_ids}")
    

def del_turn_number_over_5():
    """一键删除数据库从轮次超过5的样本，方便重新合成与排查"""
    from data_construction.data.data_manager import MTDataManager
    data_manager = MTDataManager()
    number = data_manager.delete_by_turn(turn=5)
    print(f"成功删除了数据中轮次超过5的样本: {number}条")
    # 查询剩余样本数量
    print(f"当前剩余样本数量: {data_manager.count()}")

def init_ablation_study_db():
    """初始化消融实验的数据库：导出原始数据并排出solution和test，然后设置到新数据库"""
    from data_construction.data.data_manager import MTDataManager

    # 1. 导出JSONL
    output_path = (Path(__file__).parent / "decomposition_data.jsonl").resolve()
    original_data_manager = MTDataManager()
    original_data_manager.export_jsonl_without_solution_and_test(output_path=output_path)
    # 2. 设置到新数据库
    db_path_1 = (Path(__file__).parent / 'data' / f'ablation_study_evaluator_distinctiveness.db').resolve()
    db_path_2 = (Path(__file__).parent / 'data' / f'ablation_study_evaluator.db').resolve()

    manager_1 = MTDataManager(db_path=db_path_1)    
    manager_2 = MTDataManager(db_path=db_path_2)

    manager_1.setup_with_jsonl(jsonl_path=output_path)
    manager_2.setup_with_jsonl(jsonl_path=output_path)


def main(args):
    if args.decomposition:
        # 分解
        decomposition(dataset=args.dataset,max_iter_number=args.max_iter_number)
    elif args.codegen:
        # 代码和测试生成
        # 支持消融实验
        if args.skip_evaluator:
            exclude_module = ["evaluator"]
        elif args.skip_evaluator_and_distinctiveness:
            exclude_module = ["evaluator", "distinctiveness"]
        else:
            exclude_module = []
        codegen(dataset=args.dataset,auto_skip=args.auto_skip,max_iter_number=args.max_iter_number,exclude_module=exclude_module)
    elif args.del_turn_number_over_5:
        del_turn_number_over_5()
    elif args.get_mt_data_info:
        get_mt_data_info()
    elif args.init_ablation_study_db:
        init_ablation_study_db()


def create_arg_parser():
    parser = argparse.ArgumentParser(description="Data construction pipeline")
    # 通用参数
    parser.add_argument("--max_iter_number",help="最大迭代次数",type=int,default=5)
    # pipeline1: 执行分解
    parser.add_argument("--decomposition",help="执行分解",action="store_true")
    parser.add_argument("--dataset",help="数据集名称",choices=["bigcodebench","autocodebench","mrgbench"],default="bigcodebench")
    # pipeline2: 执行代码生成
    parser.add_argument("--codegen",help="执行代码生成",action="store_true")
    parser.add_argument("--auto_skip",help="是否自动跳过Human In The Loop",action="store_true")
    parser.add_argument("--skip_evaluator",help="消融实验，跳过评估智能体模块",action="store_true")
    parser.add_argument("--skip_evaluator_and_distinctiveness",help="消融实验，跳过评估智能体模块和区分度实验模块",action="store_true")
    # db操作
    parser.add_argument("--get_mt_data_info",help="获取分解数据集的统计信息",action="store_true")
    parser.add_argument("--del_turn_number_over_5",help="删除轮次大于5的数据",action="store_true")
    parser.add_argument("--init_ablation_study_db",help="初始化消融实验的数据库",action="store_true")

    return parser

if __name__ == '__main__':
    parser = create_arg_parser()
    args = parser.parse_args()
    assert not(args.skip_evaluator == True and args.skip_evaluator_and_distinctiveness == True)
    main(args)