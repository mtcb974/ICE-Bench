from dataclasses import dataclass,asdict
from typing import Optional,TypedDict,Dict,Any
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain_deepinfra import ChatDeepInfra
import os


@dataclass
class ModelConfig:
    model_provider: str
    model_name: str

@dataclass
class ModelInferParam:
    temperature: int
    max_retries: Optional[int]
    max_tokens: Optional[int] = None
    timeout: Optional[int] = None
    model_kwargs: Optional[Dict[str,Any]] = None
    def to_dict(self):
        return {k: v for k, v in asdict(self).items() if v is not None}

class LLMManager:
    def __init__(self) -> None:
        pass

    @classmethod
    def _get_env_key(cls,model_provider:str):
        """
        根据模型提供商，返回环境变量中的Key
        Return:
            api_key
            base_url
        """
        return f"{model_provider.upper()}_API_KEY",f"{model_provider.upper()}_BASE_URL"

    @classmethod
    def get_model(cls,model_provider:str,model_name:str,params:ModelInferParam):
        api_key, base_url = cls._get_env_key(model_provider)
        api_key = os.getenv(api_key)
        base_url = os.getenv(base_url)

        if model_provider == 'openai' or 'openrouter' or 'volcengine':
            return ChatOpenAI(
                model=model_name,
                api_key=api_key,
                base_url=base_url,
                **(params.to_dict())
            )
        elif model_provider == 'deepseek':
            return ChatDeepSeek(
                model=model_name,
                api_key=api_key,
                base_url=base_url,
                **(params.to_dict())
            )
        elif model_provider == 'deepinfra':
            return ChatDeepInfra(
                model=model_name,
                api_key=api_key,
                base_url=base_url,
                **(params.to_dict())
            )