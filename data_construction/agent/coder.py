from typing import Any, Dict, List, Type, Literal,Optional,Tuple
from langchain_core.messages import BaseMessage
from .base_agent import BaseAgent,T
from pathlib import Path
from pydantic import BaseModel,Field

### Coder Agent 输出Schema

class CoderOutput(BaseModel):
    solution: str = Field(description="def task_func(...):\n    ... # Your modified/implemented code here")
    test: str = Field(description="import unittest\n\nclass TestCases(unittest.TestCase):\n    ... # Your test methods here")
    reason: Optional[str] = Field(description="[If feedback is provided] A concise analysis of the root cause of the failure based on the feedback.",default=None)
    plan: Optional[str] = Field(description="[If feedback is provided] A specific description of the changes you will make to the solution and/or tests to address the failure.",default=None)

class CoderOutputForAutocodebench(BaseModel):
    solution: str = Field(description="import java.util.*;\n    class <class_name>... # Your modified/implemented code here")
    test: str = Field(description="import static org.junit.jupiter.api.Assertions.*;\n    class Test<class_name>... # Your test methods here")
    reason: Optional[str] = Field(description="[If feedback is provided] A concise analysis of the root cause of the failure based on the feedback.",default=None)
    plan: Optional[str] = Field(description="[If feedback is provided] A specific description of the changes you will make to the solution and/or tests to address the failure.",default=None)

class CoderOutputForMrgbench(BaseModel):
    solution: str = Field(description="Your modified/implemented Code Snippet here")
    test: str = Field(description="Your test methods here")
    reason: Optional[str] = Field(description="[If feedback is provided] A concise analysis of the root cause of the failure based on the feedback.",default=None)
    plan: Optional[str] = Field(description="[If feedback is provided] A specific description of the changes you will make to the solution and/or tests to address the failure.",default=None)


### Evaluator Agent 输出Schema

class RuleResult(BaseModel):
    """Evaluation result on test function and question against the given rules"""
    rule:str = Field(description="The rule name mentioned above. Example - Naming/Signature Inconsistency")
    result: Literal["pass","fail"] = Field(description="Whether it conform to the rule above")
    reason: str = Field(description="Brief explanation following the rule. Example - function name 'add' doesn't match problem's 'sum'")

class CodegenEvalOutput(BaseModel):
    """Evaluation result"""
    rule_results: List[RuleResult] = Field(description="Evaluate whether the test cases comply with the rules")
    decision: Literal["RETAIN","TEST_REFINE","QUESTION_REFINE"] = Field(description="Provide a definitive decision following the workflows")
    feedback: str = Field(description="If decision is `TEST_REFINE` or `INSTRUCTION_REFINE`, please provide detailed and actionable feedback to support subsequent improvements. For `INSTRUCTION_REFINE`, additionally provide one you consider reasonable.")
    

class CoderAgent(BaseAgent):
    """编码智能体，负责撰写代码方案和测试用例"""
    
    def __init__(
        self,
        name: str = "coder",
        model_provider: str | None = "openrouter",
        model_name: str | None = "anthropic/claude-sonnet-4",
        dataset: str | None = "bigcodebench",
        system_prompt_name: str | None = None,
        **kwargs
    ) -> None:
        self.dataset = dataset
        kwargs.setdefault('temperature',0)
        super().__init__(name, model_provider, model_name, **kwargs)

    def single_turn_chat_with_structure_output(self, user_prompt: str, output_type: type[T] | None = None) -> Tuple[T | BaseMessage]:
        if self.dataset == 'bigcodebench':
            return super().single_turn_chat_with_structure_output(user_prompt, CoderOutput)
        elif self.dataset == 'autocodebench':
            return super().single_turn_chat_with_structure_output(user_prompt, CoderOutputForAutocodebench)
        elif self.dataset == 'mrgbench':
            return super().single_turn_chat_with_structure_output(user_prompt, CoderOutputForMrgbench)
    
        
class CodegenEvaluatorAgent(BaseAgent):
    """评估智能体，检查测试用例是否与指令需求对齐"""

    def __init__(
        self,
        name: str = "codegen_evaluator",
        model_provider: str | None = "deepinfra",
        model_name: str | None = "Qwen/Qwen3-235B-A22B-Instruct-2507",
        system_prompt_name: str | None = None,
        **kwargs
    ) -> None:
        kwargs.setdefault('temperature',0.5)
        super().__init__(name, model_provider, model_name, **kwargs)

    def single_turn_chat_with_structure_output(self, user_prompt: str, output_type: type[T] | None = None) -> Tuple[T | BaseMessage]:
        return super().single_turn_chat_with_structure_output(user_prompt, CodegenEvalOutput)
