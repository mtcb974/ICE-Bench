from typing import Any, Dict, List, Type, Literal,Optional,Tuple
from langchain_core.messages import BaseMessage
from .base_agent import BaseAgent,T
from pathlib import Path
from pydantic import BaseModel,Field

class DecompositionOutput(BaseModel):
    """Decomposition Result"""
    type :Literal["basic","restrictive"] = Field(description="Instruction type, 'basic' for the basic instruction, 'restrictive' for the restrictive instruction")
    instruction: str = Field(description="Instruction content. For basic type, conclude with: 'You should write self-contained code starting with: ```<language>\n<code>```' ,where the <language> refer to python or java and the '<code>' is the function signature and necessary code for the basic round. For restrictive: A single, clear restrictive instruction that builds upon turn 1 , with no prefixes or suffixes. ")

class DecompositionOutputs(BaseModel):
    results: List[DecompositionOutput]

class InstructionEvalOutput(BaseModel):
    """Evaluation Result"""
    decision: Literal["RETAIN","REJECT"] = Field(description="'RETAIN' if all criteria are met, 'REJECT' if any criterion is violated.")
    feedback: str = Field(description="Your concise reasoning here, including actionable suggestions if rejected")

class DependencyEdge(BaseModel):
    """Represents a directed dependency relationship between two instructions."""
    src: str = Field(description="Source instruction ID (dependency prerequisite) that must be executed earlier")
    dst: str = Field(description="Destination instruction ID (dependent instruction) that requires the source to be executed first")

class AnalyzerOutputs(BaseModel):
    """Comprehensive analysis output containing dependency relationships and execution plan"""
    dependency_graph: List[DependencyEdge] = Field(description="List of directed edges representing all dependency relationships between instructions")
    execution_sequence: List[str] = Field(description="Optimal execution order derived from topological sort, providing a clear roadmap for incremental implementation")
    analysis_summary: str = Field(description="Concise explanation of key dependencies, critical paths, and sequencing rationale")
    
class InstructionGenerationOutput(BaseModel):
    """Instruction output"""
    instruction: str = Field(description="Natural language instruction that could have led a programmer to write this function")

class InstructionGeneratorAgent(BaseAgent):
    """将函数签名和函数体合成自然语言指令的智能体"""
    def __init__(self,
                name: str = "instruction_generator",
                model_provider: str | None = "deepinfra",
                model_name: str | None = "Qwen/Qwen3-235B-A22B-Instruct-2507",
                system_prompt_name: str | None = None,
                **kwargs) -> None:
        super().__init__(name, model_provider, model_name, system_prompt_name, **kwargs)

    def single_turn_chat_with_structure_output(self, user_prompt: str, output_type: Optional[type[T]]=None) -> Tuple[T,BaseMessage]:
        return super().single_turn_chat_with_structure_output(user_prompt, InstructionGenerationOutput)


class DecomposerAgent(BaseAgent):
    """
    指令分解智能体
    """

    def __init__(
        self,
        name: str="decomposer",
        model_provider: str | None = "deepinfra",
        model_name: str | None = "Qwen/Qwen3-235B-A22B-Instruct-2507",
        system_prompt_name: str | None = None,
        **kwargs
    ) -> None:
        kwargs.setdefault('temperature',0.5)
        super().__init__(name, model_provider, model_name, **kwargs)
    
    def single_turn_chat_with_structure_output(self, user_prompt: str, output_type: Optional[type[T]]=None) -> Tuple[T,BaseMessage]:
        return super().single_turn_chat_with_structure_output(user_prompt, DecompositionOutputs)

class InstructionEvaluatorAgent(BaseAgent):
    """
    指令分解质量评估智能体
    """
    def __init__(
        self,
        name: str="instruction_evaluator",
        model_provider: str | None = "deepinfra",
        model_name: str | None = "Qwen/Qwen3-235B-A22B-Instruct-2507",
        system_prompt_name: str | None = None,
        **kwargs
    ) -> None:
        kwargs.setdefault('temperature',0.5)
        super().__init__(name, model_provider, model_name, **kwargs)
    
    def single_turn_chat_with_structure_output(self, user_prompt: str, output_type: type[T] | None = None) -> Tuple[T | BaseMessage]:
        return super().single_turn_chat_with_structure_output(user_prompt, InstructionEvalOutput)

class InstructionAnalyzerAgent(BaseAgent): 
    """
    指令依赖分析智能体
    """
    def __init__(
        self,
        name: str="instruction_analyzer",
        model_provider: str | None = "deepinfra",
        model_name: str | None = "Qwen/Qwen3-235B-A22B-Instruct-2507",
        **kwargs
    ) -> None:
        kwargs.setdefault('temperature',0.5)
        super().__init__(name, model_provider, model_name, **kwargs)
    
    def single_turn_chat_with_structure_output(self, user_prompt: str, output_type: type[T] | None = None) -> Tuple[T | BaseMessage]:
        return super().single_turn_chat_with_structure_output(user_prompt, AnalyzerOutputs)