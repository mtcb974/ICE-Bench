from data_construction.data.data_reader import MRGBenchReader
from ..agent.decomposer import DecomposerAgent,InstructionEvaluatorAgent,DecompositionOutput,InstructionAnalyzerAgent,AnalyzerOutputs
from ..data import BigCodeBenchReader,AutoCodeBenchReader,MTDataManager,MultiTurnDataInstance,PerTurnDataInstance
from ..data.log_manager import LogManager,LogItem,InteractionItem

from typing import Annotated,TypedDict,List,Dict,Any,Literal,Optional
from langgraph.graph import StateGraph,START,END
from langgraph.types import Command
from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph.message import add_messages
from pydantic import BaseModel,Field
from operator import add
from pathlib import Path
from datetime import datetime

class DecompositionState(TypedDict):
    # input state
    seed_instruction: str
    seed_code: str
    hash_id: str
    task_id: str
    source: str
    metadata: Any
    # output state
    decomposed_result: List[DecompositionOutput] # decomposer
    feedback: str # evaluator
    decision: Literal["RETAIN","REJECT"] # evaluator
    analysis_result: AnalyzerOutputs # analyzer
    # log state
    interactions: Annotated[List[InteractionItem],add]
    fail_reason: str = None
    total_tokens: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    # config state
    iter_number: int = 0

class DecompositionModule:
    def __init__(
        self,
        dataset: str,
        seed_data_path: Optional[str] = None,
        max_iter_number: int = 5
    ) -> None:
        """
        初始化分解模块
        params:
            dataset: 'bigcodebench' or 'autocodebench'
            seed_data_path: 会根据dataset自动获取
        """
        self.dataset = dataset
        if dataset not in ['bigcodebench','autocodebench','mrgbench']:
            raise ValueError("Dataset not support, try bigcodebench and autocodebench.")
        # 种子数据加载模块
        if dataset == 'bigcodebench':
            self.seed_loader = BigCodeBenchReader(seed_data_path)
        elif dataset == 'autocodebench':
            self.seed_loader = AutoCodeBenchReader(seed_data_path)
        elif dataset == 'mrgbench':
            self.seed_loader = MRGBenchReader(seed_data_path)

        # 数据管理器
        self.data_manager = MTDataManager()

        # 日志管理器
        self.log_manager = LogManager()
        
        # 配置agent
        if dataset == 'bigcodebench':
            self.decomposer_llm = DecomposerAgent()
        elif dataset == 'autocodebench':
            self.decomposer_llm = DecomposerAgent(system_prompt_name="decomposer_autocodebench")
        elif dataset == 'mrgbench':
            self.decomposer_llm = DecomposerAgent(system_prompt_name="decomposer_mrgbench")
            
        self.evaluator_llm = InstructionEvaluatorAgent()
        self.analyzer_llm = InstructionAnalyzerAgent()

        # 配置graph
        self.graph = self._build_decomposition_graph()

        # 配置参数
        self.max_iter_number = max_iter_number
        
    def _decompose_node(self,state:DecompositionState) -> Command[Literal["evaluator","__end__"]]:
        """ LLM performs a decomposition on seed_instruction"""
        print("进入节点: decomposer")

        seed_instruction = state.get("seed_instruction")
        seed_code = state.get("seed_code")
        if not seed_instruction:
            print("出错: 请提供分解指令seed_instruction, 即将退出")
            return Command(
                goto="__end__"
            )
        
        # 检查是否有feedback
        feedback = state.get("feedback")
        previous_decomposition = state.get("decomposed_result")
        if not feedback:
            # user prompt without feedback
            user_prompt = f"""--- ORIGINAL COMPLEX INSTRUCTION ---
{seed_instruction}
--- CORRESPONDING CODE ANSWER ---
{seed_code}
"""
        else:
            print(f"evaluator提供了feedback:\n {feedback}")
            # user prompt with feedback
            user_prompt = f"""Refine your decomposition in the previous chat with my feedback.
--- ORIGINAL COMPLEX INSTRUCTION ---
{seed_instruction}
--- CORRESPONDING CODE ANSWER ---
{seed_code}
--- PREVIOUS DECOMPOSITIONS ---
{previous_decomposition}
--- FEEDBACK ---
{feedback}
"""
        
        # 执行分解，记录过程数据
        decomposed_result,response = self.decomposer_llm.single_turn_chat_with_structure_output(user_prompt=user_prompt)
        # 记录日志数据
        interaction_log = self._generation_interaction_log("decomposer",response.response_metadata,decomposed_result)
        token_infos = self._get_updated_usage(state,response)
        return Command(
            goto="evaluator",
            update={
                'decomposed_result': decomposed_result.results,
                # 日志
                'interactions': [interaction_log],
                **token_infos
            }
        )
    
    def _evaluator_node(self,state:DecompositionState) -> Command[Literal["__end__","decomposer","analyzer","failure_handle"]]:
        """ LLM performs an evaluation on the decomposed instructions"""
        print("进入Evaluator节点")

        # 获取状态
        seed_instruction = state.get("seed_instruction")
        seed_code = state.get("seed_code")
        decomposed_instructions = state.get("decomposed_result")
        iter_number = state.get("iter_number")
        user_prompt = f"""--- ORIGINAL COMPLEX INSTRUCTION ---
{seed_instruction}
--- CORRESPONDING CODE ANSWER ---
{seed_code}
--- DECOMPOSED INSTRUCTIONS ---
{decomposed_instructions}
"""
        # 检查是否超过最大迭代次数
        if iter_number >= self.max_iter_number:
            return Command(
                goto="failure_handle",
                update={
                    "fail_reason": f"运行Evaluator超过最大迭代次数:{self.max_iter_number}次"
                }
            )

        # 执行评估
        evaluate_result,response = self.evaluator_llm.single_turn_chat_with_structure_output(user_prompt=user_prompt)
        # 跳转与记录日志
        interaction = self._generation_interaction_log("evaluator",response.response_metadata,evaluate_result)
        token_infos = self._get_updated_usage(state,response)
        if evaluate_result.decision == 'RETAIN':
            return Command(
                update={
                    "feedback": evaluate_result.feedback,
                    "decision": evaluate_result.decision,
                    "interactions": [interaction],
                    **token_infos
                },
                goto="analyzer"
            )
        elif evaluate_result.decision == 'REJECT':
            return Command(
                update={
                    "feedback": evaluate_result.feedback,
                    "decision": evaluate_result.decision,
                    "iter_number": iter_number +1,
                    "interactions": [interaction],
                    **token_infos
                },
                goto="decomposer"
            )
        else:
            print("Evaluator LLM出现幻觉: decision非预期. 已退出Graph")
            return Command(
                goto="__end__"
            )
            
    def _analyzer_node(self,state:DecompositionState) -> Command[Literal["success_handle"]]:
        """ LLM perform an analysis on instructions, model their dependencies, and determine optimal execution sequences through dependency graph analysis.  """
        print("进入Analyzer节点")

        seed_instruction = state.get("seed_instruction")
        seed_code = state.get("seed_code")
        decomposed_instructions = self._get_decomposed_instructions_str(state)

        user_prompt = f"""--- ORIGINAL COMPLEX INSTRUCTION ---
{seed_instruction}
--- CORRESPONDING CODE ANSWER ---
{seed_code}
--- DECOMPOSED INSTRUCTIONS ---
{decomposed_instructions}
"""
        analysis_result,response = self.analyzer_llm.single_turn_chat_with_structure_output(user_prompt=user_prompt)
        interaction = self._generation_interaction_log("analyzer",response.response_metadata,analysis_result)
        token_infos = self._get_updated_usage(state,response)
        return Command(
            goto="success_handle",
            update={
                "analysis_result":analysis_result,
                "interactions": [interaction],
                **token_infos
            }
        )

    def _success_handle_node(self,state:DecompositionState) -> None:
        """
            数据生成成功。
            1. 根据给定的顺序添加到数据库
            2. 添加日志
        """
        decomposed_result = state.get("decomposed_result")
        analysis_result = state.get("analysis_result")

        # 按照分析顺序去生成PerTurnData，此时的solution和test都还是None
        turn_datas = []
        for idx,seq_num_str in enumerate(analysis_result.execution_sequence):
            seq_num = self._safe_convert_to_int(seq_num_str)
            turn_data = PerTurnDataInstance(
                turn_num=idx+1,
                instruction=decomposed_result[seq_num-1].instruction,
                solution=None,
                test=None
            )
            turn_datas.append(turn_data)
        
        mt_instance = MultiTurnDataInstance(
            hash_id=state.get("hash_id"),
            total_turn=len(decomposed_result),
            turn_datas=turn_datas,
            metadata=state.get("metadata")
        )
        # 插入数据库
        self.data_manager.add(mt_instance)
        # 重新查询一遍，拿到mt_id，这里以后可以优化
        item = self.data_manager.get(state.get("hash_id"))

        # 记录日志
        log = LogItem(
            status="success",
            task="instruction decomposition",
            fail_reason=None,
            mt_id=item.mt_id,
            hash_id=state.get("hash_id"),
            source=state.get("source"),
            interactions=state.get("interactions"),
            total_prompt_tokens=state.get("total_prompt_tokens"),
            total_completion_tokens=state.get("total_completion_tokens"),
            total_tokens=state.get("total_tokens"),
            interaction_number=len(state.get("interactions")),
        )
        self.log_manager.add(log)

        print_info = f"""成功制作一条多轮数据，汇总信息如下:
- 数据ID:{item.mt_id}, 指令轮数:{item.total_turn}
- Token总消耗:{log.total_tokens}, 输入Token总消耗:{log.total_prompt_tokens}
- 评估迭代次数: {state.get("iter_number")}, 交互次数:{log.interaction_number}"""
        print(print_info)

    def _failure_handle_node(self,state:DecompositionState) -> None:
        """任务失败节点，需要记录日志"""
        fail_reason=state.get("fail_reason")
        print(f"数据ID:{state.get("task_id")},⚠️进入失败节点,失败原因:{fail_reason}")
        log = LogItem(
            status="fail",
            task="instruction decomposition",
            fail_reason=fail_reason,
            mt_id=-1,
            hash_id=state.get("hash_id"),
            source=state.get("source"),
            interactions=state.get("interactions"),
            total_prompt_tokens=state.get("total_prompt_tokens"),
            total_completion_tokens=state.get("total_completion_tokens"),
            total_tokens=state.get("total_tokens"),
            interaction_number=len(state.get("interactions"))
        )
        self.log_manager.add(log)

    def _build_decomposition_graph(self) -> Runnable:
        graph_builder = StateGraph(DecompositionState)
        # node
        graph_builder.add_node("decomposer",self._decompose_node)
        graph_builder.add_node("evaluator",self._evaluator_node)
        graph_builder.add_node("analyzer",self._analyzer_node)
        graph_builder.add_node("success_handle",self._success_handle_node)
        graph_builder.add_node("failure_handle",self._failure_handle_node)
        # edge
        graph_builder.add_edge(START,"decomposer")
        graph_builder.add_edge("success_handle",END)
        graph_builder.add_edge("failure_handle",END)
        # 在evaluator内部使用条件边
        return graph_builder.compile()

    def execute_decompostion(self) -> None:
        """
        在指定数据集上，利用大语言模型多智能体进行分解指令，并自动进行验证以提高数据质量
        """
        # 获取数据，暂时用随机采样的接口
        # data_iterator = self.seed_loader.read_from_jsonl_random_sample(sample_size=10,random_seed=57)
        ids = self.data_manager.get_all_hashids()
        data_iterator = self.seed_loader.read_from_jsonl_with_filter(ids=ids)
        
        # 适配一下autocodebench
        for sample in data_iterator:
            graph_input = {
                # input
                "seed_instruction": sample.instruction,
                "seed_code": sample.solution,
                "hash_id": sample.hash_id,
                "task_id": sample.metadata["task_id"] if self.dataset == 'bigcodebench' else None,
                "source": sample.metadata["source"],
                "metadata": sample.metadata,
                # log state
                "fail_reason": None,
                "total_tokens": 0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                # config
                "iter_number": 0
            }
            self.graph.invoke(graph_input)

        # 最终输出jsonl
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = (Path(__file__).parent / f"decomposition_output_{timestamp_str}.jsonl").resolve()
        self.data_manager.export_jsonl(output_path)

        print(f"整个分解过程完毕，已导出为jsonl输出到: {output_path}")

    ##### 辅助函数 #####
    def _get_decomposed_instructions_str(self, state: DecompositionState) -> str:
        """将分解后的指令组装成为字符串，用于提供给Analyzer Agent"""
        result = []
        decomposed_res = state.get("decomposed_result")
        for idx, ins in enumerate(decomposed_res):
            result.append(f"{idx+1}. {ins.type}: {ins.instruction}")
        return "\n".join(result)
    
    def _safe_convert_to_int(self,str_value):
        """将字符串的数字转换为int类型，如果无法转成成功则抛出异常"""
        try:
            return int(str_value)
        except ValueError:
            print(f"无法将 '{str_value}' 转换为整数")
            return None  # 或者根据需求返回其他默认值
    
    def _generation_interaction_log(self,agent:str,response_metadata:Any,content:BaseModel) -> InteractionItem:
        """返回一个交互日志对象"""
        return InteractionItem(
            agent=agent,
            response_metadata=response_metadata,
            content=content.model_dump()
        )
    
    def _get_updated_usage(self,state:DecompositionState,response:Any) ->Dict[str,Any]:
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