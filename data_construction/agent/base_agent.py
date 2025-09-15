from ..llms import LLMManager,ModelConfig,ModelInferParam

from abc import ABC,abstractmethod
from typing import List,Dict,Any,Optional,ClassVar,Type,TypeVar,Tuple
from pathlib import Path
from langchain_core.messages import SystemMessage,BaseMessage,HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts.chat import ChatPromptTemplate
from pydantic import BaseModel
from dataclasses import dataclass

T = TypeVar('T',bound=BaseModel)

@dataclass
class InteractionLog:
    agent_name: str
    result: str
    response_meta: Optional[Dict]


class BaseAgent(ABC):
    """
    Agent抽象基类，维护提示词和执行逻辑
    """
    default_model_config: ClassVar[ModelConfig] = {
        'model_name': "deepseek-chat",
        'model_provider': "deepseek"
    }
    default_infer_params: ClassVar[ModelInferParam] = ModelInferParam(
        temperature=0.0,
        max_retries=3,
        timeout=480,
        model_kwargs={}
    )
    
    def __init__(
        self,
        name: str,
        model_provider:Optional[str] = None,
        model_name:Optional[str] = None,
        system_prompt_name: str | None = None,
        **kwargs # 用于覆盖 infer params
    ) -> None:
        self.name = name

        # 使用传入值覆盖类默认值
        self.model_config = ModelConfig(
            model_name=model_name or self.default_model_config.model_name,
            model_provider=model_provider or self.default_model_config.model_provider
        )
        # 合并默认 infer 参数与传入的 kwargs
        infer_params_dict = self.default_infer_params.to_dict()
        infer_params_dict.update({k: v for k, v in kwargs.items() if k in infer_params_dict})
        self.model_infer_param = ModelInferParam(**infer_params_dict)
        print(self.model_infer_param)
        # 初始化 LLM
        self.llm = self._setup_llm()
        self._system_message: Optional[SystemMessage] = None

        # 初始化系统提示词路径
        if system_prompt_name is None:
            # 默认按照agent的名字来找提示词
            self.system_prompt_path = Path(__file__).parent / ".." / "prompts" / f"{self.name}.md"
        else:
            self.system_prompt_path = Path(__file__).parent / ".." / "prompts" / f"{system_prompt_name}.md"

    def _setup_llm(self):
        """初始化大模型"""
        return LLMManager.get_model(
            model_name=self.model_config.model_name,
            model_provider=self.model_config.model_provider,
            params=self.model_infer_param
        )
    
    def system_prompt(self) -> str:
        """由具体的Agent实现系统提示词"""
        if not self.system_prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {self.system_prompt_path}")

        with open(self.system_prompt_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        
        return content

    def single_turn_chat_with_structure_output(self,user_prompt:str,output_type:Optional[Type[T]]=None)->Tuple[T, BaseMessage]:
        """
        执行单步聊天并将类型绑定到大模型的输出里面，返回用户自定义的类型。
        
        内部完成的操作：
            - 托管system_prompt
            - 构造完整的messages
            - 调用LLM
            - 解析输出
        """
        parser = PydanticOutputParser(pydantic_object=output_type)
        system_prompt = self.system_prompt() + '\n' + """
Wrap the output in ```json ... ``` tags
{format_instructions}"""
        prompt = ChatPromptTemplate.from_messages([
                ("system",system_prompt),
                ("human","{user_prompt}")
        ]).partial(format_instructions=parser.get_format_instructions())
        
        # 分布调用
        # 等价于：chain = prompt | self.llm | parser
        formatted_prompt = prompt.invoke({"user_prompt": user_prompt})
        llm_response = self.llm.invoke(formatted_prompt)
        structured_output = parser.invoke(llm_response)
        return structured_output,llm_response


    def _call_llm(self, messages: List[BaseMessage]) -> str:
        """调用 LLM 并返回字符串结果"""
        response = self.llm.invoke(messages)
        if hasattr(response, "content"):
            return str(response.content)
        return str(response)