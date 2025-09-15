from ..agent.coder import CoderAgent,CodegenEvaluatorAgent
from ..data import BigCodeBenchReader,AutoCodeBenchReader,MTDataManager,MultiTurnDataInstance,PerTurnDataInstance
from ..data.log_manager import LogManager,LogItem,InteractionItem
from .sandbox_client import BigCodeSandboxClient,AutoCodeBenchSandboxClient, MrgBenchSandboxClient

from typing import Annotated,TypedDict,List,Dict,Any,Literal,Optional
from langgraph.graph import StateGraph,START,END
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command,interrupt
from langchain_core.runnables import Runnable
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph.message import add_messages
from pydantic import BaseModel,Field
from operator import add
from pathlib import Path
from datetime import datetime

class CodegenState(TypedDict):
    """代码生成相关状态"""
    # input state
    ## 数据
    hash_id: str
    turn_datas: List[PerTurnDataInstance]
    total_turn: int
    metadata: Dict[str,Any]
    mt_id: Optional[int]
    # output state
    ## coder
    code: str
    test: str
    ## eval
    feedback: str
    # log state
    interactions: Annotated[List[InteractionItem],add]
    fail_reason: str = None
    total_tokens: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    # config state
    iter_number: int
    current_codegen_turn: int
    # human in the loop
    human_feedback: Dict = None

class CodegenModule:
    def __init__(self,
        dataset:Optional[str] = None,
        exclude_module:List[Literal["evaluator","distinctiveness"]] = [],
        db_path: Optional[str] = None,
        log_db_path: Optional[str] = None,
        max_iter_number: int = 10,
        auto_skip: bool = False
    ) -> None:
        """
        初始化代码生成模块
        参数：
            - dataset: 数据集名称
            - exclude_module: 消融实验用，用于去除某些节点
        """
        self.dataset = dataset
        if dataset not in ['bigcodebench','autocodebench','mrgbench','deveval']:
            raise ValueError("Dataset not support.")
        
        # 数据管理模块: 含指令数据加载和最终数据管理
        self.data_manager = MTDataManager(db_path=db_path)

        # 日志管理器
        self.log_manager = LogManager(db_path=log_db_path)

        # Agent管理
        if self.dataset == 'bigcodebench':
            self.coder_llm = CoderAgent(name="coder",dataset=self.dataset,system_prompt_name="coder")
        elif self.dataset == 'autocodebench':
            self.coder_llm = CoderAgent(name="coder_autocodebench",dataset=self.dataset,system_prompt_name="coder_autocodebench")
        elif self.dataset == 'mrgbench':
            self.coder_llm = CoderAgent(name="coder_mrgbench",dataset=self.dataset,system_prompt_name="coder_mrgbench")
        
        self.evaluator_llm = CodegenEvaluatorAgent()

        # 沙盒客户端
        if dataset == 'bigcodebench':
            self.sandbox_client = BigCodeSandboxClient()
        elif dataset == 'autocodebench':
            self.sandbox_client = AutoCodeBenchSandboxClient()
        elif dataset == 'mrgbench':
            self.sandbox_client = MrgBenchSandboxClient()

        # graph
        self.graph = self._build_codegen_graph()

        # 配置参数
        self.max_iter_number = max_iter_number
        self.auto_skip = auto_skip

        # 消融实验
        self.exclude_module = exclude_module
    
    def _coder_node(self,state:CodegenState
    ) -> Command[Literal["success_handle",
                         "failure_handle",
                         "check_correctness",
                         "__end__"
                         ]]:
        """ Coder节点，专注代码和测试生成 """

        iter_number = state.get("iter_number")
        current_codegen_turn = state.get("current_codegen_turn")
        print(f"进入Coder节点,迭代次数:{iter_number},当前轮次:{current_codegen_turn}")
        if iter_number >= self.max_iter_number:
            print("⚠️超过最大迭代次数，正在跳转到END节点")
            return Command(
                goto="failure_handle",
                update={
                    "fail_reason": "超过最大迭代次数",
                }
            )
        if current_codegen_turn > state.get("total_turn"):
            print("⚠️警告: 当前轮次超过总共轮次，但仍旧进入Coder节点，请检查代码逻辑")
            print("将直接跳到END节点，不接受本条样本")
            return Command(
                goto="__end__"
            )

        # 1. 构造提示
        current_instruction = state.get("turn_datas")[current_codegen_turn-1].instruction
        feedback = state.get("feedback")
        if current_codegen_turn > 1:
            previous_solution = state.get("turn_datas")[current_codegen_turn-2].solution
        else:
            previous_solution = "None. Because the current round is the first round."
        
        if self.dataset == 'mrgbench':
            original_code = state.get("metadata").get("func","")
            original_test = state.get("metadata").get("test_code","")

            if not feedback:
                user_prompt = f"""--- Current Turn ---
{current_codegen_turn}
--- Current Instruction ---
{current_instruction}
--- Original Repository Code ---
```java
{original_code}
```
--- Original Test Code ---
```java
{original_test}
```
--- Previous Round's Solution (if any) ---
```java
{previous_solution}
```
"""
            else:
                user_prompt = f"""--- Current Turn ---
{current_codegen_turn}
--- Current Instruction ---
{current_instruction}
--- Original Repository Code ---
```java
{original_code}
```
--- Original Test Code ---
```java
{original_test}
```
--- Previous Round's Solution (if any) ---
```java
{previous_solution}
```
--- The current round of code and test that you provided to me in the last interaction ---
Code you provided to me in the last interaction:
```java
{state.get("code")}
```
Test you provided to me in the last interaction:
```java
{state.get("test")}
```
--- Feedback ---
{feedback}
"""
            
        elif not feedback:
            # 无feedback
            user_prompt = f"""--- Current Round ---
{current_codegen_turn}
--- Current Instruction ---
{current_instruction}
--- Previous Round's Solution (if any) ---
```
{previous_solution}
```
"""
        else:
           # 有feedback  
           user_prompt = f"""--- Current Round ---
{current_codegen_turn}
--- Current Instruction ---
{current_instruction}
--- Previous Round's Solution (if any) ---
```
{previous_solution}
```
--- The current round of code and test that you provided to me in the last interaction ---
Code you provided to me in the last interaction: 
```
{state.get("code")}
```
Test you provided to me in the last interaction:
```
{state.get("test")}
```
--- Feedback ---
{feedback}
""" 
        
        # 2. 执行生成
        codegen_result,response = self.coder_llm.single_turn_chat_with_structure_output(user_prompt)
        # 记录日志数据
        interaction_log = self._generation_interaction_log("coder",response.response_metadata,codegen_result)
        token_infos = self._get_updated_usage(state,response)

        # 3. 进入正确性测试
        return Command(
            goto="check_correctness",
            update={
                # 生成结果
                'code': codegen_result.solution,
                'test': codegen_result.test,
                # 参数
                'iter_number': state.get('iter_number')+1,
                # 日志
                'interactions': [interaction_log],
                **token_infos
            }
        )
    
    def _check_correctness_node(self,state:CodegenState) -> Command[Literal["coder","check_distinctiveness"]]:
        """校验生成代码正确性的节点"""
        print("进入check_correctness节点")
        code = state.get("code")
        test = state.get("test")
        # 沙盒执行
        if self.dataset == 'mrgbench':
            exec_result = self.sandbox_client.execute_code_with_test(code,test,state.get("metadata"))
        else:
            exec_result = self.sandbox_client.execute_code_with_test(code,test)
        exec_status,exec_detail = exec_result["status"],exec_result["detail"]

        # 记录日志
        interaction_log = self._generation_interaction_log("check_correctness",None,exec_result)

        if exec_status == 'pass':
            print("正确性校验通过")
            # 判断轮次，只有轮次大于1才需要进入区分度测试
            current_codegen_turn = state.get("current_codegen_turn")
            if current_codegen_turn > 1:
                # 进入区分度检验
                return Command(
                    goto="check_distinctiveness",
                    update={
                        "interactions": [interaction_log]
                    }
                )
            else:
                print("当前轮次为第一轮，直接进入Evaluator")
                return Command(
                    goto="codegen_evaluator",
                    update={
                        "interactions": [interaction_log]
                    }
                )

        else:
            print(f"正确性校验失败，正在构建feedback并返回coder.\n执行输出:{exec_detail}")
            # 回到coder，提供反馈
            feedback = f"""Based on execution environment, the solution code in current round fails the test case you haved generated.
Status: {exec_status}
Stdout: {exec_detail}
"""
            return Command(
                goto="coder",
                update={
                    "feedback": feedback,
                    "interactions": [interaction_log]
                }
            )
        
    def _check_distinctiveness_node(self,state:CodegenState) -> Command[Literal["coder","codegen_evaluator"]]:
        """区分度校验"""
        print("进入check_distinctiveness节点")
        # 如果开启了消融实验，直接进入evaluator
        if "distinctiveness" in self.exclude_module:
            return Command(
                goto="codegen_evaluator",
            )
        test = state.get("test")
        current_codegen_turn = state.get("current_codegen_turn")
        if current_codegen_turn > 1:
            previous_solution = state.get("turn_datas")[current_codegen_turn-2].solution
        # 沙盒执行
        if self.dataset == 'mrgbench':
            exec_result = self.sandbox_client.execute_code_with_test(previous_solution,test,state.get("metadata"))
        else:
            exec_result = self.sandbox_client.execute_code_with_test(previous_solution,test)
        exec_status,exec_detail = exec_result["status"],exec_result["detail"]

        # 记录日志
        interaction_log = self._generation_interaction_log("check_distinctiveness",None,exec_result)

        if exec_status == 'fail':
            print("区分度校验通过")
            # 进入对齐智能体
            return Command(
                goto="codegen_evaluator",
                update={
                    "interactions": [interaction_log]
                }
            )
        else:
            print("区分度校验未通过，准备提供feedback并返回coder")
            # 回到coder，提供反馈
            feedback = f"""Based on the execution environment, although the current round of code you generated can pass the test cases, the previous round of code also passed the test cases. This means that the discriminability of the test cases is insufficient.
"""
            return Command(
                goto="coder",
                update={
                    "feedback": feedback,
                    "interactions": [interaction_log]
                }
            )

    def _codegen_evaluator_node(self,state:CodegenState) -> Command[Literal["coder","human_in_the_loop","success_handle"]]:
        """指令-测试对齐智能体"""
        print("进入evaluator节点")

        # 可选跳过这个节点
        if "evaluator" in self.exclude_module:
            current_codegen_turn = state.get("current_codegen_turn")
            # 1. 记录本轮次生成的数据
            turn_datas = state.get("turn_datas")
            turn_datas[current_codegen_turn-1].solution = state.get("code")
            turn_datas[current_codegen_turn-1].test = state.get("test") 
            # 2. 检查轮次信息，决定跳转
            if current_codegen_turn == state.get("total_turn"):
                print("已完成所有轮次，准备前往success_handle")
                # 完成所有轮次
                return Command(
                    goto="success_handle",
                    update={
                        # 更新生成数据
                        "turn_datas": turn_datas,
                    }
                )
            elif current_codegen_turn < state.get("total_turn"):
                print(f"已完成当前轮次[{current_codegen_turn}/{state.get("total_turn")}]，准备前往coder节点")
                # 进入下一轮的生成
                return Command(
                    goto="coder",
                    update={
                        # 更新生成数据
                        "turn_datas": turn_datas,
                        # 清空feedback
                        "feedback": None,
                        # 参数
                        "iter_number": 0,
                        "current_codegen_turn": current_codegen_turn + 1,
                    }
                )

        current_codegen_turn = state.get("current_codegen_turn")
        current_instruction = state.get("turn_datas")[current_codegen_turn-1].instruction
        current_test = state.get("test")
        current_code = state.get("code")
        if current_codegen_turn > 1:
            previous_turn_datas = state.get("turn_datas")[:current_codegen_turn-1]
            previous_instruction = '\n'.join([turn.instruction for turn in previous_turn_datas])
        else:
            previous_instruction = "None, becauses this is the first turn."
        
        user_prompt = f"""--- Previous Instructions ---
{previous_instruction}
--- Current Instruction ---
{current_instruction}
--- SOLUTION ---
{current_code}
--- TEST FUNCTION ---
{current_test}
"""
        # 执行评估
        eval_result,response = self.evaluator_llm.single_turn_chat_with_structure_output(user_prompt)
        eval_decision, eval_feedback = eval_result.decision, eval_result.feedback
        # 记录日志
        interaction_log = self._generation_interaction_log("codegen_evaluator",response.response_metadata,eval_result)
        token_infos = self._get_updated_usage(state,response)

        if eval_decision == 'RETAIN':
            print(f"decision == {eval_decision}")
            # 轮次成功
            # 1. 记录本轮次生成的数据
            turn_datas = state.get("turn_datas")
            turn_datas[current_codegen_turn-1].solution = state.get("code")
            turn_datas[current_codegen_turn-1].test = state.get("test")
            # 2. 检查轮次信息，决定跳转
            if current_codegen_turn == state.get("total_turn"):
                print("已完成所有轮次，准备前往success_handle")
                # 完成所有轮次
                return Command(
                    goto="success_handle",
                    update={
                        # 更新生成数据
                        "turn_datas": turn_datas,
                        # eval
                        "feedback": eval_feedback,
                        # 日志
                        "interactions": [interaction_log],
                        **token_infos,
                    }
                )
            elif current_codegen_turn < state.get("total_turn"):
                print(f"已完成当前轮次[{current_codegen_turn}/{state.get("total_turn")}]，准备前往coder节点")
                # 进入下一轮的生成
                return Command(
                    goto="coder",
                    update={
                        # 更新生成数据
                        "turn_datas": turn_datas,
                        # 清空feedback
                        "feedback": None,
                        # 参数
                        "iter_number": 0,
                        "current_codegen_turn": current_codegen_turn + 1,
                        # 日志
                        "interactions": [interaction_log],
                        **token_infos,
                    }
                )
            else:
                print("❌执行到错误分支: 当前轮次 > total_turn，请检查代码逻辑")
                return Command(
                    goto="__end__"
                )
        elif eval_decision == 'TEST_REFINE':
            print(f"evaluator状态{eval_decision}，准备返回coder节点.\n先前测试:\n {current_test}\n先前代码:\n{current_code}\n反馈:\n{eval_feedback}\n规则符合情况:\n{eval_result.rule_results}")
            # 测试改进，返回coder节点
            return Command(
                goto="coder",
                update={
                    "feedback": eval_feedback,
                    # 日志
                    "interactions": [interaction_log],
                    **token_infos,
                }
            )
        elif eval_decision == 'QUESTION_REFINE':
            print(f"evaluator状态{eval_decision}，准备进入人类干预节点.\n 代码:{current_code}\n 测试:{current_test}\n 反馈:\n{eval_feedback}, ")
            # 指令改进，进入Human节点
            return Command(
                goto="human_in_the_loop",
                update={
                    "feedback": eval_feedback,
                    # 日志
                    "interactions": [interaction_log],
                    **token_infos,
                }
            )

    def _success_handle_node(self,state:CodegenState) -> Command[Literal["__end__"]]:
        """样本生成成功，更新数据库"""
        new_mt_instance = MultiTurnDataInstance(
            hash_id=state.get("hash_id"),
            total_turn=state.get("total_turn"),
            turn_datas=state.get("turn_datas"),
            metadata=state.get("metadata"),
            mt_id=state.get("mt_id")
        )
        
        # 更新数据库
        status = self.data_manager.update(new_mt_instance)
        if status:
            print(f"一条数据生成完毕！更新数据库成功，数据ID:{new_mt_instance.mt_id}")
        else:
            print("❌更新数据库失败,数据信息如下:")
            print(new_mt_instance)
        
        # 更新日志
        log = LogItem(
            status="success",
            task="codegen",
            fail_reason=None,
            mt_id=new_mt_instance.mt_id,
            hash_id=new_mt_instance.hash_id,
            source=new_mt_instance.metadata["source"],
            interactions=state.get("interactions"),
            total_prompt_tokens=state.get("total_prompt_tokens"),
            total_completion_tokens=state.get("total_completion_tokens"),
            total_tokens=state.get("total_tokens"),
            interaction_number=len(state.get("interactions")),
        )
        self.log_manager.add(log)
        print_info = f"""✅成功制作一条多轮数据，汇总信息如下:
- 数据ID:{new_mt_instance.mt_id}, 指令轮数:{new_mt_instance.total_turn}
- Token总消耗:{log.total_tokens}, 输入Token总消耗:{log.total_prompt_tokens}
- 评估迭代次数: {state.get("iter_number")}, 交互次数:{log.interaction_number}"""
        print(print_info)

    def _failure_handle_node(self,state:CodegenState) -> Command[Literal["__end__"]]:
        """样本生成失败，记录日志"""
        fail_reason = state.get("fail_reason")
        print(f"⚠️ID={state.get("mt_id")}进入失败节点,正在记录日志,失败原因:{fail_reason}")
        log = LogItem(
            status="fail",
            task="codegen",
            fail_reason=fail_reason,
            mt_id=state.get("mt_id"),
            hash_id=state.get("hash_id"),
            source=state.get("metadata").get("source"),
            interactions=state.get("interactions"),
            total_prompt_tokens=state.get("total_prompt_tokens"),
            total_completion_tokens=state.get("total_completion_tokens"),
            total_tokens=state.get("total_tokens"),
            interaction_number=len(state.get("interactions"))
        )
        self.log_manager.add(log)

    def _human_in_the_loop_node(self,state:CodegenState)->Command[Literal["coder"]]:
        """更新指令数据，人类标注者可以决定是直接通过 或者 重新进入coder节点"""
        # wait for human feedback
        human_feedback = interrupt({
            "current_codegen_turn": state.get("current_codegen_turn"),
            "current_instruction": state.get("turn_datas")[state.get("current_codegen_turn")-1].instruction,
            "feedback": state.get("feedback")
        })
        # 更新指令
        print(f"得到用户反馈: {human_feedback}")
        turn_datas = state.get("turn_datas")
        current_codegen_turn = state.get("current_codegen_turn")
        turn_datas[current_codegen_turn-1].instruction = human_feedback["new_instruction"]
        # 日志
        interaction_log = self._generation_interaction_log("human_in_the_loop",None,human_feedback)
        # 此时应该回到Evaluator
        return Command(
            goto="coder",
            update={
                "turn_datas": turn_datas,
                # 保存人类反馈，清空原来的反馈
                "human_feedback": human_feedback,
                "feedback": None,
                # 日志
                "interactions": [interaction_log]
            }
        )

    def _build_codegen_graph(self) -> CompiledStateGraph:
        graph_builder = StateGraph(CodegenState)
        # node
        graph_builder.add_node("coder",self._coder_node)
        graph_builder.add_node("codegen_evaluator",self._codegen_evaluator_node)
        graph_builder.add_node("check_correctness",self._check_correctness_node)
        graph_builder.add_node("check_distinctiveness",self._check_distinctiveness_node)
        graph_builder.add_node("success_handle",self._success_handle_node)
        graph_builder.add_node("failure_handle",self._failure_handle_node)
        graph_builder.add_node("human_in_the_loop",self._human_in_the_loop_node)
        # edge
        graph_builder.add_edge(START,"coder")
        graph_builder.add_edge("success_handle",END)
        graph_builder.add_edge("failure_handle",END)
        # checkpoint
        memory = InMemorySaver()
        return graph_builder.compile(checkpointer=memory)
    
    # def execute_codegen_loop_multi_thread(self) -> None:
    #     """读取数据库中的指令并开始进行代码和测试的循环（多线程并发版本）"""
    #     import threading
    #     from concurrent.futures import ThreadPoolExecutor, as_completed

    #     print("准备执行codegen_loop（多线程模式）")
        
    #     # 获取数据
    #     instances = self.data_manager.get_incomplete_instances(source=self.dataset)
    #     print(f"从数据库中获取到未完成的数据: {len(instances)}条.")

    #     # 线程安全的共享结果收集（如果需要收集返回值）
    #     # 这里我们只是处理，不需要返回太多东西，但可以记录失败/成功
    #     success_count = 0
    #     failure_count = 0
    #     lock = threading.Lock()  # 用于保护共享计数器和打印

    #     def process_instance(instance) -> bool:
    #         nonlocal success_count, failure_count
    #         if instance.total_turn > 5:
    #             with lock:
    #                 print(f"跳过: MT_ID={instance.mt_id}, total_turn > 5")
    #             return False

    #         graph_input_init: dict = {
    #             'mt_id': instance.mt_id,
    #             'hash_id': instance.hash_id,
    #             'turn_datas': instance.turn_datas,
    #             'metadata': instance.metadata,
    #             'total_turn': instance.total_turn,
    #             'iter_number': 0,
    #             'current_codegen_turn': 1,
    #             'fail_reason': None,
    #             'total_tokens': 0,
    #             'total_prompt_tokens': 0,
    #             'total_completion_tokens': 0,
    #             'interactions': []
    #         }

    #         thread_config = {"configurable": {"thread_id": f"instance_{instance.mt_id}"}, "recursion_limit": 80}
    #         graph_input = graph_input_init
    #         idx = 1
    #         try:
    #             with lock:
    #                 print(f"正在处理数据: MT_ID={instance.mt_id}, TASK_ID={instance.metadata['task_id']}, 第{idx}次调用graph.")
    #             self.graph.invoke(graph_input, thread_config)
    #             with lock:
    #                 success_count += 1
    #             return True

    #         except Exception as e:
    #             with lock:
    #                 print(f"发生异常: {e}，已跳过 instance: MT_ID={instance.mt_id}, TASK_ID={instance.metadata['task_id']}")
    #                 failure_count += 1
    #             return False

    #     # 并行执行（可根据 CPU/IO 情况调整 max_workers）
    #     max_workers = 10  # 建议根据你的模型推理吞吐能力调整（如 API 限速则调小）
    #     with ThreadPoolExecutor(max_workers=max_workers) as executor:
    #         futures = [executor.submit(process_instance, instance) for instance in instances]
    #         # 可选：实时查看进度
    #         for future in as_completed(futures):
    #             try:
    #                 future.result()  # 可捕获异常（已在函数内处理）
    #             except Exception as e:
    #                 with lock:
    #                     print(f"线程执行出错: {e}")

    #     # 所有任务完成，导出结果（必须在主线程）
    #     timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     output_path = (Path(__file__).parent / f"codegen_output_{timestamp_str}.jsonl").resolve()
    #     self.data_manager.export_jsonl(output_path)

    #     print(f"整个代码生成过程完毕，成功处理 {success_count} 条，失败 {failure_count} 条。")
    #     print(f"已导出为 jsonl 输出到: {output_path}")

    def execute_codegen_loop(self) -> None:
        """读取数据库中的指令并开始进行代码和测试的循环"""
        print("准备执行codegen_loop")
        # 获取数据
        instances = self.data_manager.get_incomplete_instances(source=self.dataset)
        print(f"从数据库中获取到未完成的数据: {len(instances)}条.")

        for instance in instances:
            if instance.total_turn > 5:
                print("total_turn > 5，已经跳过本样本")
                continue
            graph_input_init:CodegenState = {
                # 输入数据
                'mt_id': instance.mt_id,
                'hash_id': instance.hash_id,
                'turn_datas': instance.turn_datas,
                'metadata': instance.metadata,
                'total_turn': instance.total_turn,
                # 配置参数
                'iter_number': 0,
                'current_codegen_turn': 1,
                # 日志数据
                'fail_reason': None,
                "total_tokens": 0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                'interactions': []
            }
            thread = {"configurable": {"thread_id": f"instance_{instance.mt_id}"},"recursion_limit":80}

            graph_input = graph_input_init
            idx = 1
            try:
                while True:
                    print(f"正在处理数据: MT_ID={instance.mt_id}, ,第{idx}次调用graph.")
                    graph_state = self.graph.invoke(graph_input,thread)
                    # 检测中断
                    if self.graph.get_state(thread).interrupts:
                        print(f"检测到中断，等待用户输入⌛️.")
                        # 存在中断，获取用户输入
                        human_feedback = self._get_human_input(graph_state)
                        # 重新执行
                        graph_input = Command(resume={"new_instruction": human_feedback})
                        idx += 1
                    else:
                        break
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"发生异常: {e}\n, 已跳过instance: MT_ID={instance.mt_id}")
                continue
        
        # 最终输出jsonl
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = (Path(__file__).parent / f"codegen_output_{timestamp_str}.jsonl").resolve()
        self.data_manager.export_jsonl(output_path)

        print(f"整个代码生成过程完毕，已导出为jsonl输出到: {output_path}")

    def _get_human_input(self, state:Dict) -> str:
        """
        获取人类用户的新指令输入
        
        Args:
            interrupt_data: 中断事件的数据
            
        Returns:
            str: 用户输入的新指令
        """
        print("\n" + "="*60)
        print("⚠️ codegen_evaluator认为指令存在不对齐情况，需要手动处理!")
        print("🤖 进入HUMAN-IN-THE-LOOP 模式")
        print("="*60)
        print(f"当前轮次: {state.get('current_codegen_turn')}")
        print(f"原始指令: {state.get("turn_datas")[state.get("current_codegen_turn")-1].instruction}")
        print(f"evaluator的建议: {state.get('feedback')}")
        print("\n请根据评估反馈，输入改进后的新指令:")
        print("(输入 'skip' 表示不处理)")
        print("-"*60)
        
        while True:
            if self.auto_skip:
                print("⏭️ 已启动autoskip，自动跳过Human In the Loop。返回原始指令")
                return state.get("turn_datas")[state.get("current_codegen_turn")-1].instruction
                
            user_input = input("请输入新指令: ").strip()

            if user_input.lower() == 'skip':
                print("⏭️ 已跳过本实例，返回原始指令")
                return state.get("turn_datas")[state.get("current_codegen_turn")-1].instruction
            elif user_input:
                print(f"✅ 已接收新指令: {user_input[:100]}...")
                return user_input
            else:
                print("❌ 输入不能为空，请重新输入")
    
    def _generation_interaction_log(self, agent: str, response_metadata: Any, content: BaseModel | Dict) -> InteractionItem:
        """返回一个交互日志对象，content可以是BaseModel或Dict"""
        if isinstance(content, dict):
            content_data = content
        else:
            content_data = content.model_dump()
        return InteractionItem(
            agent=agent,
            response_metadata=response_metadata,
            content=content_data
        )
    
    def _get_updated_usage(self,state:CodegenState,response:Any) ->Dict[str,Any]:
        """
        Params:
            state: 状态
            response: 模型回复, 一般是BaseMessage类型
        
        Returns: 更新了使用情况的字典
        """
        total_tokens = state.get("total_tokens",0) + response.response_metadata['token_usage']['total_tokens']
        total_prompt_tokens = state.get("total_prompt_tokens",0) + response.response_metadata['token_usage']['prompt_tokens']
        total_completion_tokens = state.get("total_completion_tokens",0)+ response.response_metadata['token_usage']['completion_tokens']
        return {
            "total_tokens": total_tokens,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens
        }
