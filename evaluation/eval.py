from typing import Literal,List,Set,Dict,Any,Optional
import json
import argparse
import uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from threading import Lock
import threading
import asyncio
from dotenv import load_dotenv
import time

from data_construction.data.data_manager import MTDataManager,MultiTurnDataInstance
from data_construction.service.sandbox_client import BigCodeSandboxClient,AutoCodeBenchSandboxClient, MrgBenchSandboxClient
from .data import InferenceResultJsonlManager,InferenceResult,InferenceTurnResult
from .llm import CoderAgent

load_dotenv()

class MTEvaluation:
    def __init__(self,
                 llm_provider:str = "deepseek",
                 llm:str = None,
                 context_setting: Literal["base","golden"] = "base",
                 prompt_setting: Literal["fh","edit",'ci'] = "fh",
                 lang: Literal["python","java","java-repo"] = "java",
                 run_id: str = "default",
                 max_workers: int = 8,
                 think_flag: Literal["oss-think","qwen3-think","None"] = None,
                 db_path: str | None = None,
                 command: str = "inference"
                 ) -> None:
        # 参数设置
        self.llm_provider = llm_provider
        self.llm = llm
        self.context_setting = context_setting
        self.prompt_setting = prompt_setting
        self.lang = lang
        if run_id == "default":
            if think_flag != "None":
                self.run_id = f"{llm_provider}_{llm.replace("/","_")}_think_{context_setting}_{prompt_setting}_{lang}"
            else:
                self.run_id = f"{llm_provider}_{llm.replace("/","_")}_{context_setting}_{prompt_setting}_{lang}"
        else:
            self.run_id = run_id
        self.max_workers = max_workers
        # 推理参数设置
        self.think_flag = think_flag
        
        # 输出管理
        self.output_path_base = Path(__file__).parent / 'results'
        if not self.output_path_base.exists():
            self.output_path_base.mkdir(parents=True, exist_ok=True)
        self.output_path = self.output_path_base / f"{self.run_id}.jsonl"

        self.output_manager = InferenceResultJsonlManager(self.output_path)

        # 数据管理
        
        self.data_manger = MTDataManager(db_path=db_path)

        # 大模型管理
        self.eval_llm = CoderAgent(
            model_provider=self.llm_provider,
            model_name=self.llm,
            think_flag=think_flag
        )

        # 写锁
        self.lock = Lock()

        ### 评估 ###
        
        if command == 'evaluation':
            if self.lang == 'python':
                self.sandbox_client = BigCodeSandboxClient()
            elif self.lang == 'java':
                self.sandbox_client = AutoCodeBenchSandboxClient()
            elif self.lang == 'java-repo':
                self.sandbox_client = MrgBenchSandboxClient()
                # 确保所有容器都健康
                if not self.sandbox_client.check_health():
                    raise RuntimeError("Docker容器连接失败")
        
    def run_inference(self):
        """运行推理"""
        # 1. 获取数据, 并过滤掉上次执行过的样本
        if self.lang == 'python':
            data_instances = self.data_manger.list_all_by_source(source="bigcodebench")
        elif self.lang == 'java':
            data_instances = self.data_manger.list_all_by_source(source="autocodebench")
        elif self.lang == 'java-repo':
            data_instances = self.data_manger.list_all_by_source(source="mrgbench")
        # data_instances = [self.data_manger.get("c2246cae13f63349f2a97210")]

        existing_hash_ids = self.output_manager.get_all_hash_ids()

        filtered_mt_dataset:List[MultiTurnDataInstance] = []
        for mt_data in data_instances:
            hash_id = mt_data.hash_id
            if hash_id not in existing_hash_ids:
                filtered_mt_dataset.append(mt_data)
        
        print(f"Total tasks to infer: {len(filtered_mt_dataset)}, skipped {len(data_instances) - len(filtered_mt_dataset)} already processed tasks.")

        # 2. 多线程执行, 并append到jsonl中
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self._run_inference_single,task_data)
                for task_data in filtered_mt_dataset
            ]
            for future in as_completed(futures):
                try:
                    future.result(timeout=120)
                except TimeoutError as e:
                    print(f"出现样本运行超时了: {e}")
                except Exception as e:
                    print(f"Error occurred during inference: {e}")

        print(f"推理结束，所有样本已推理完成,路径:{self.output_path}.")

    def _run_inference_single(self,task_data:MultiTurnDataInstance):
        """代表一条线程，执行某个样本的评估"""
        try:
            infer_result:Dict[str,Any] = {
                "hash_id": task_data.hash_id,
                "mt_id": task_data.mt_id,
                "language": self.lang,
                "solutions": [],
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
            trajectories: List[InferenceTurnResult] = []
            
            for turn_data in task_data.turn_datas:
                print(f"正在处理任务: MT-ID={task_data.mt_id}, 轮次: {turn_data.turn_num}/{task_data.total_turn}")
                # 1. 获取message
                turn_prompt = turn_data.instruction
                turn_number = turn_data.turn_num
                turn_result: Dict[str,Any] = {
                    "turn_number": turn_number,
                    "prompt": turn_prompt,
                    "raw_solution": "",
                    "solution": "",
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                }
                turn_message = self._get_infer_prompt(trajectories,turn_prompt,task_data)
                # print(f"=== turn_message in turn {turn_number} ===\n {turn_message} ")
                # 2. 调用大模型
                turn_solution,turn_raw_solution,usage_dict = self.eval_llm.chat_for_eval(messages=turn_message,language=self.lang)
                # 3. 保存轮次结果，更新总结果
                turn_result['raw_solution'] = turn_raw_solution
                turn_result['solution'] = turn_solution
                turn_result['prompt_tokens'] = usage_dict['prompt_tokens']
                turn_result['completion_tokens'] = usage_dict['completion_tokens']
                turn_result['total_tokens'] = usage_dict['total_tokens']
                infer_result['prompt_tokens'] = infer_result['prompt_tokens'] + turn_result['prompt_tokens']
                infer_result['completion_tokens'] = infer_result['completion_tokens'] + turn_result['completion_tokens']
                infer_result['total_tokens'] = infer_result['total_tokens'] + turn_result['total_tokens']

                # print(f"=== solution in turn {turn_number} ===\n{turn_solution} ")
                trajectories.append(InferenceTurnResult.from_dict(turn_result))
                print(f"Finish turn {turn_number} in task: {task_data.mt_id}!")

            # 写日志
            infer_result['solutions'] = trajectories
            result = InferenceResult.from_dict(infer_result)
            with self.lock:
                self.output_manager.append(result)
                print(f"Successfully processed and append solution: {task_data.mt_id}")

            return True            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Exception in task {task_data.mt_id}: {e}")
            return False
        
    def _get_infer_prompt(self,
                          trajectories:List[InferenceTurnResult],
                          prompt:str,
                          mt_data: Optional[MultiTurnDataInstance]
                          ) -> List[Any]:
        """获取评估的提示词"""
        if self.context_setting == 'golden' and mt_data is None:
            raise ValueError("If golden is true, then user must provide mt_data")

        lang = "java" if "java" in self.lang else "python"

        messages = []
        # 1. 设置系统提示词
        messages.append({"role": "system" , "content": self.eval_llm.system_prompt(self.lang)})

        # 2. 设置消息历史和用户提示词
        task_prompt = f"programming problem:\n{prompt.strip()}\n"
        
        context = ""
        ## 仓库级需要设置context信息
        if self.lang == 'java-repo' and len(trajectories) == 0:
            context = mt_data.metadata.get("context","")
            task_prompt = f"## The contexts above the function ## \n {context} \n {task_prompt}"

        ## prompt策略1: 全历史
        if self.prompt_setting == 'fh':
            # add history data
            if len(trajectories) > 0:
                for idx,traj in enumerate(trajectories):
                    user_msg = {"role": "user","content": traj.prompt}
                    if self.context_setting == 'golden':
                        assistant_msg = {"role": "assistant", "content": f"```{lang}\n{mt_data.turn_datas[idx].solution}```"}
                    elif self.context_setting == 'base':
                        assistant_msg = {"role": "assistant", "content": traj.raw_solution}
                    messages.append(user_msg)
                    messages.append(assistant_msg)
            # add current turn prompt
            messages.append({"role": "user","content": task_prompt})
        elif self.prompt_setting == 'edit':
            trajectories_len = len(trajectories)
            if trajectories_len > 0:
                previous_code = mt_data.turn_datas[trajectories_len-1].solution if self.context_setting == 'golden' else trajectories[-1].solution
                ## 仓库级需要设置Context
                if self.lang == 'java-repo':
                    task_prompt = self.eval_llm.edit_prompt(language=self.lang,previous_code=previous_code,instruction=prompt.strip(),context=context)
                else:
                    task_prompt = self.eval_llm.edit_prompt(language=self.lang,previous_code=previous_code,instruction=prompt.strip())
                messages.append({"role": "user", "content": task_prompt})
            else:
                messages.append({"role": "user", "content": task_prompt})
        elif self.prompt_setting == 'ci':
            if self.context_setting == 'golden':
                raise ValueError("Append Mode do not support golden")
            
            instructions = ""
            trajectories_len = len(trajectories)
            
            for idx, traj in enumerate(trajectories):
                instructions = instructions + '\n' + traj.prompt.strip()
            
            instructions  = instructions + '\n' + prompt
            if self.lang == 'java-repo':
                task_prompt = f"## The contexts above the function ## \n {context} \n  ## Programming problem ## \n{instructions.strip()}"
            else:
                task_prompt = f"programming problem: \n{instructions.strip()}"
            messages.append({"role": "user", "content": task_prompt})

        return messages        

    def run_evaluation_for_repo(self):
        """运行评估，生成一个报告（并发版）"""

        # 0. 初始化预处理指标
        metric_output_path = self.output_path_base / f"EvalMetric_{self.run_id}.json"
        log_output_path = self.output_path_base / f"EvalLog_{self.run_id}.jsonl"

        metrics = {
            'llm_provider': self.llm_provider,
            'llm': self.llm,
            'language': self.lang,
            'context_setting': self.context_setting,
            'prompt_setting': self.prompt_setting,
            'timestamp': None,
            'dataset_size': 0,
            'answer_size': 0,
            'turn_acc': {
                f'turn_{i}': {'sample_size': 0, 'correct_size': 0, 'acc': 0.0, 'pass_mt_ids': []}
                for i in range(1, 6)
            },
            'avg_acc': 0.0,
            'fully_completed': {
                'correct_size': 0,
                'acc': 0.0,
                'fully_completed_mt_ids': []
            },
            'avg_total_token': 0.0,
        }
        logs = []

        total_token_cost = 0.0

        # 1. 读取jsonl获取数据
        infer_results = self.output_manager.read_all()
        if self.lang == 'python':
            datasets = self.data_manger.list_all_by_source(source='bigcodebench')
        elif self.lang == 'java':
            datasets = self.data_manger.list_all_by_source(source='autocodebench')
        elif self.lang == 'java-repo':
            datasets = self.data_manger.list_all_by_source(source='mrgbench')

        metrics['dataset_size'] = len(datasets)
        metrics['answer_size'] = len(infer_results)
        print(f"数据集长度: {metrics['dataset_size']}, 答案长度: {metrics['answer_size']}")
        assert metrics['dataset_size'] == metrics['answer_size']

        # 计算每个轮次的 sample_size
        for ds in datasets:
            for i in range(1, ds.total_turn + 1):
                metrics['turn_acc'][f'turn_{i}']['sample_size'] += 1

        # 构建 hash_id 到数据集的映射
        hashid2ds = {ds.hash_id: ds for ds in datasets}

        # 2. 逐条同步评估
        print(f"开始并行评估，共 {len(infer_results)} 个样本，使用 {self.sandbox_client.get_available_clients_count()} 个容器...")
        start_time = time.time()

        ##### 评估一条样本 ######

        def process_single_item(infer_res):
            log_copy = infer_res.to_dict().copy()
            original_ds = hashid2ds.get(infer_res.hash_id)
            local_token_cost = infer_res.total_tokens or 0.0
            all_pass = True
            mt_id = getattr(original_ds, "mt_id", getattr(original_ds, "id", None))
            metadata = original_ds.metadata
            
            for idx, turn_res in enumerate(infer_res.solutions):
                turn_num = idx + 1
                turn_solution = turn_res.solution
                turn_test = original_ds.turn_datas[idx].test
                
                try:
                    exec_res = self.sandbox_client.execute_code_with_test(turn_solution, turn_test, metadata)
                    exec_status = exec_res["status"]
                    exec_detail = exec_res["detail"]
                except Exception as e:
                    exec_status = "error"
                    exec_detail = f"Execution failed: {str(e)}"
                
                log_copy['solutions'][idx].setdefault('exec_status', exec_status)
                log_copy['solutions'][idx].setdefault('exec_detail', exec_detail)
                
                if exec_status == 'pass':
                    # 使用线程安全的操作更新指标
                    with self.metrics_lock:
                        metrics['turn_acc'][f'turn_{turn_num}']['correct_size'] += 1
                        if mt_id is not None:
                            metrics['turn_acc'][f'turn_{turn_num}']['pass_mt_ids'].append(mt_id)
                elif exec_status == 'error':
                    all_pass = False
                    print(f"❌ 出现沙盒请求错误，mt_id: {mt_id}, detail: {exec_detail}")
                else:
                    all_pass = False
            
            # fully_completed 统计
            if all_pass:
                with self.metrics_lock:
                    metrics['fully_completed']['correct_size'] += 1
                    if mt_id is not None:
                        metrics['fully_completed']['fully_completed_mt_ids'].append(mt_id)
            
            return log_copy, local_token_cost, mt_id
        
        # 添加线程锁用于指标更新
        self.metrics_lock = threading.Lock()
        
        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=self.sandbox_client.get_available_clients_count()) as executor:
            future_to_item = {
                executor.submit(process_single_item, infer_res): infer_res 
                for infer_res in infer_results
            }
            
            for future in as_completed(future_to_item):
                try:
                    log_copy, local_token_cost, mt_id = future.result()
                    logs.append(log_copy)
                    total_token_cost += local_token_cost
                    print(f"✅ 完成评估: {mt_id}")
                except Exception as e:
                    print(f"❌ 评估失败: {e}")
        
        print(f"并行评估完成，耗时: {time.time() - start_time:.2f} 秒")

        # 3. 汇总和计算 metrics
        turn_acc_sum, turn_count = 0.0, 0
        for i in range(1, 6):
            sample_size = metrics['turn_acc'][f'turn_{i}']['sample_size']
            correct_size = metrics['turn_acc'][f'turn_{i}']['correct_size']
            if sample_size > 0:
                acc = correct_size / sample_size
                metrics['turn_acc'][f'turn_{i}']['acc'] = round(acc, 4)
                turn_acc_sum += acc
                turn_count += 1
            else:
                metrics['turn_acc'][f'turn_{i}']['acc'] = 0.0

        metrics['avg_acc'] = round(turn_acc_sum / turn_count, 4) if turn_count > 0 else 0.0
        metrics['fully_completed']['acc'] = round(
            metrics['fully_completed']['correct_size'] / metrics['dataset_size'], 4
        ) if metrics['dataset_size'] > 0 else 0.0
        metrics['avg_total_token'] = round(
            total_token_cost / metrics['dataset_size'], 2
        ) if metrics['dataset_size'] > 0 else 0.0

        metrics['timestamp'] = time.strftime("%Y%m%d%H%M%S", time.localtime())

        # 4. 写日志
        with open(log_output_path, "w", encoding="utf-8") as f_log:
            for log_item in logs:
                f_log.write(json.dumps(log_item, ensure_ascii=False) + "\n")

        with open(metric_output_path, "w", encoding="utf-8") as f_metric:
            json.dump(metrics, f_metric, ensure_ascii=False, indent=2)

        print(f"评估完成，指标已写入: {metric_output_path}，日志已写入: {log_output_path}")


    async def run_evaluation(self):
        """运行评估，生成一个报告（多线程加速版）"""

        # import threading
        # from concurrent.futures import ThreadPoolExecutor, as_completed
        # from tqdm import tqdm

        # 0. 初始化预处理指标
        metric_output_path = self.output_path_base / f"EvalMetric_{self.run_id}.json"
        log_output_path = self.output_path_base / f"EvalLog_{self.run_id}.jsonl"

        metrics = {
            'llm_provider': self.llm_provider,
            'llm': self.llm,
            'language': self.lang,
            'context_setting': self.context_setting,
            'prompt_setting': self.prompt_setting,
            'timestamp': None,
            'dataset_size': 0,
            'answer_size': 0,
            'turn_acc': {
                'turn_1': {'sample_size': 0, 'correct_size': 0, 'acc': 0.0, 'pass_mt_ids': []},
                'turn_2': {'sample_size': 0, 'correct_size': 0, 'acc': 0.0, 'pass_mt_ids': []},
                'turn_3': {'sample_size': 0, 'correct_size': 0, 'acc': 0.0, 'pass_mt_ids': []},
                'turn_4': {'sample_size': 0, 'correct_size': 0, 'acc': 0.0, 'pass_mt_ids': []},
                'turn_5': {'sample_size': 0, 'correct_size': 0, 'acc': 0.0, 'pass_mt_ids': []},
            },
            'avg_acc': 0.0,
            'fully_completed': {
                'correct_size': 0,
                'acc': 0.0,
                'fully_completed_mt_ids': []
            },
            'avg_total_token': 0.0,
        }
        logs = []

        total_token_cost = 0.0
        total_token_cost_lock = asyncio.Lock()
        metrics_lock = asyncio.Lock()
        logs_lock = asyncio.Lock()

        # 1. 读取jsonl获取数据
        infer_results = self.output_manager.read_all()
        if self.lang == 'python':
            datasets = self.data_manger.list_all_by_source(source='bigcodebench')
        elif self.lang == 'java':
            datasets = self.data_manger.list_all_by_source(source='autocodebench')

        metrics['dataset_size'] = len(datasets)
        metrics['answer_size'] = len(infer_results)
        print(f"数据集长度: {metrics['dataset_size']}, 答案长度: {metrics['answer_size']}")
        assert metrics['dataset_size'] == metrics['answer_size']

        # 计算每个轮次的sample_size
        for ds in datasets:
            total_turn = ds.total_turn
            for i in range(1, total_turn + 1):
                metrics['turn_acc'][f'turn_{i}']['sample_size'] += 1

        # 构建hash_id到数据集的映射，避免多线程下频繁查找
        hashid2ds = {ds.hash_id: ds for ds in datasets}

        # 确保sandbox的session初始化
        await self.sandbox_client.init_session()

        async def eval_one(infer_res):
            nonlocal total_token_cost
            log_copy = infer_res.to_dict().copy()
            original_ds = hashid2ds.get(infer_res.hash_id)
            local_token_cost = infer_res.total_tokens or 0.0
            all_pass = True
            mt_id = getattr(original_ds, "mt_id", getattr(original_ds, "id", None))

            # 并发执行所有轮次的测试
            turn_tasks = []
            for idx, turn_res in enumerate(infer_res.solutions):
                turn_num = idx + 1
                turn_solution = turn_res.solution
                turn_test = original_ds.turn_datas[idx].test
                ### 核心：通过沙盒验证大模型生成的答案 ###
                task = asyncio.create_task(
                    self.sandbox_client.async_execute_code_with_test(turn_solution,turn_test)
                )
                turn_tasks.append((turn_num, task, idx))
                # exec_res = self.sandbox_client.execute_code_with_test(code=turn_solution, test=turn_test)
                # exec_status, exec_detail = exec_res["status"], exec_res["detail"]
                # log_copy['solutions'][idx].setdefault('exec_status', exec_status)
                # log_copy['solutions'][idx].setdefault('exec_detail', exec_detail)
            
            # 等待所有轮次完成
            for turn_num,task,idx in turn_tasks:
                try:
                    print(f"完成校验: mt_id: {mt_id}, turn: {turn_num}")
                    exec_res = await task
                    exec_status = exec_res["status"]
                    exec_detail = exec_res["detail"]
                except Exception as e:
                    exec_status = "error"
                    exec_detail = f"Async execution failed: {str(e)}"

                log_copy['solutions'][idx].setdefault('exec_status', exec_status)
                log_copy['solutions'][idx].setdefault('exec_detail', exec_detail)

                # 线程安全地更新metrics
                async with metrics_lock:
                    if exec_status == 'pass':
                        metrics['turn_acc'][f'turn_{turn_num}']['correct_size'] += 1
                        if mt_id is not None:
                            metrics['turn_acc'][f'turn_{turn_num}']['pass_mt_ids'].append(mt_id)
                    elif exec_status == 'error':
                        all_pass = False
                        print(f"❌ 出现沙盒请求错误，mt_id: {mt_id}, detail: {exec_detail}")
                    else:
                        all_pass = False
            
            # fully_completed 统计
            async with metrics_lock:
                if all_pass:
                    metrics['fully_completed']['correct_size'] += 1
                    if mt_id is not None:
                        metrics['fully_completed']['fully_completed_mt_ids'].append(mt_id)
            
            # 线程安全地累加token消耗
            if local_token_cost > 0:
                async with total_token_cost_lock:
                    total_token_cost += local_token_cost
            # 线程安全地写入日志
            async with logs_lock:
                logs.append(log_copy)
            print(f"✅所有轮次数据完成验证: {mt_id}")

        
        # 使用asyncio进行并发
        print(f"开始异步并发评估，共 {len(infer_results)} 个样本...")
        start_time = time.time()

        try:
            # 控制并发数（防止 aiohttp 连接过多）
            semaphore = asyncio.Semaphore(self.max_workers)  # 控制并发协程数

            async def sem_eval_one(infer_res):
                async with semaphore:
                    result =  await eval_one(infer_res)
                    await asyncio.sleep(0.1)
                    return result
            # 提交所有任务
            tasks = [sem_eval_one(infer_res) for infer_res in infer_results]
            await asyncio.gather(*tasks)
        except Exception as e:
            print(f"异步评估过程中发生异常: {e}")
        finally:
            # 关闭 sandbox session
            await self.sandbox_client.close()

        print(f"异步评估完成，耗时: {time.time() - start_time:.2f} 秒")

        # 使用ThreadPoolExecutor并发评测
        # with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
        #     futures = [executor.submit(eval_one, infer_res) for infer_res in infer_results]
        #     for future in tqdm(as_completed(futures),total=len(futures),desc="Evaluting..."):
        #         # 捕获异常，防止线程池崩溃
        #         try:
        #             future.result()
        #         except Exception as e:
        #             print(f"评测线程异常: {e}")

        # 3. 汇总和计算metrics，并最终写入output_path
        # 3.1 turn_acc['turn_{i}']['acc']计算
        turn_acc_sum = 0.0
        turn_count = 0
        for i in range(1, 6):
            sample_size = metrics['turn_acc'][f'turn_{i}']['sample_size']
            correct_size = metrics['turn_acc'][f'turn_{i}']['correct_size']
            if sample_size > 0:
                acc = correct_size / sample_size
                metrics['turn_acc'][f'turn_{i}']['acc'] = round(acc, 4)
                turn_acc_sum += acc
                turn_count += 1
            else:
                metrics['turn_acc'][f'turn_{i}']['acc'] = 0.0

        # 3.2 avg_acc: 所有轮次的acc求平均
        metrics['avg_acc'] = round(turn_acc_sum / turn_count, 4) if turn_count > 0 else 0.0

        # 3.3 fully_completed['acc']: 所有轮次都pass的样本的acc
        metrics['fully_completed']['acc'] = round(
            metrics['fully_completed']['correct_size'] / metrics['dataset_size'], 4
        ) if metrics['dataset_size'] > 0 else 0.0

        # 3.4 avg_total_token: total_token_cost / dataset_size
        metrics['avg_total_token'] = round(total_token_cost / metrics['dataset_size'], 2) if metrics['dataset_size'] > 0 else 0.0

        # 记录时间戳
        metrics['timestamp'] = time.strftime("%Y%m%d%H%M%S", time.localtime())

        # 4. 将LOG写入JSONL文件
        with open(log_output_path, "w", encoding="utf-8") as f_log:
            for log_item in logs:
                f_log.write(json.dumps(log_item, ensure_ascii=False) + "\n")

        # 5. 将metrics写入JSON文件
        with open(metric_output_path, "w", encoding="utf-8") as f_metric:
            json.dump(metrics, f_metric, ensure_ascii=False, indent=2)

        print(f"评估完成，指标已写入: {metric_output_path}，日志已写入: {log_output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ### 核心命令 ### 
    parser.add_argument("--command",default="inference",choices=["inference","evaluation"],type=str)
    
    ### 推理模式 ###
    # 数据集
    parser.add_argument("--lang",default="python",choices=["python","java","java-repo"],type=str)
    # llm
    parser.add_argument("--llm_provider",default="deepinfra",type=str)
    parser.add_argument("--llm",default="",type=str)
    parser.add_argument("--think_flag",default="",type=str,choices=["oss-think","qwen3-think","None"])
    # 实验设置
    parser.add_argument("--context_setting",type=str,choices=["base","golden"],default="base")
    parser.add_argument("--prompt_setting",type=str,choices=["fh","edit","ci"],default="fh")
    # 运行ID
    parser.add_argument("--run_id",type=str,default="default")
    # 其他参数
    parser.add_argument("--max_workers",type=int, default=8)
    
    ### 消融实验 ###
    parser.add_argument("--skip_evaluator",help="Ablation experiment, skipping the evaluation agent module",action="store_true")
    parser.add_argument("--skip_evaluator_and_distinctiveness",help="Ablation experiment: skip the evaluation agent module and the discrimination experiment module",action="store_true")

    args = parser.parse_args()

    db_path = None 
    if args.skip_evaluator_and_distinctiveness:
        db_path = (Path(__file__).parent.parent / 'data_construction' / 'data' / f'ablation_study_evaluator_distinctiveness.db').resolve()
    elif args.skip_evaluator:
        db_path = (Path(__file__).parent.parent / 'data_construction' / 'data' / f'ablation_study_evaluator.db').resolve()

    print(f"db_path: {db_path}")

    eval_module = MTEvaluation(
        llm_provider=args.llm_provider,
        llm=args.llm,
        context_setting=args.context_setting,
        prompt_setting=args.prompt_setting,
        lang=args.lang,
        run_id=args.run_id,
        max_workers=args.max_workers,
        think_flag=args.think_flag,
        db_path=db_path,
        command=args.command
    )

    if args.command == 'inference':
        eval_module.run_inference()
    elif args.command == 'evaluation':
        if args.lang == 'java-repo':
            eval_module.run_evaluation_for_repo()
        else:
            import asyncio
            asyncio.run(eval_module.run_evaluation())
