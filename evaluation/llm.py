from typing import Any, Dict, List, Type, Literal,Optional,Tuple
from data_construction.agent.base_agent import BaseAgent,T
from pathlib import Path
import re
import openai

class CoderAgent(BaseAgent):
    """被测试的大语言模型"""
    
    def __init__(
        self,
        name: str = "coder",
        model_provider: str | None = "",
        model_name: str | None = "", 
        think_flag: Literal["oss-think","qwen3-think","None"] = "None",
        **kwargs
    ) -> None:
        # 设置推理参数
        kwargs.setdefault('temperature',0)
        self.think_flag = think_flag
        # 处理qwen3的思考模式：关闭
        if  ("Qwen3-30B-A3B" in model_name or "Qwen3-32B" in model_name or "Qwen3-14B" in model_name)  and think_flag != 'qwen3-think':
            model_kwargs: Dict[str,Any] = {}
            model_kwargs.setdefault('reasoning_effort','none')
            kwargs.setdefault('model_kwargs',model_kwargs)
        # 保存模型名称
        self.model_name = model_name
        super().__init__(name, model_provider, model_name,None, **kwargs)

    def system_prompt(self,language:str) -> str:
        if  "gpt-oss" in self.model_name and "oss-think" in self.think_flag:
            # 处理gpt-oss的思考强度
            return f"""Reasoning: Medium\n\nYou are an expert programmer. Your task is to provide a code solution within a single Markdown code block for the given programming problem. Do not include any direct execution commands, test cases, or usage examples within the code block.

Please write code in {language.upper()} and wrap it in a markdown code block:\n```{language}\n<your_code_here>\n```
"""
        elif "gpt-oss" in self.model_name and "oss-think" not in self.think_flag:
            # 处理gpt-oss的非思考强度
            return f"""Reasoning: Low\n\nYou are an expert programmer. Your task is to provide a code solution within a single Markdown code block for the given programming problem. Do not include any direct execution commands, test cases, or usage examples within the code block.

Please write code in {language.upper()} and wrap it in a markdown code block:\n```{language}\n<your_code_here>\n```
"""
        else:
           # 正常处理
           return f"""You are an expert programmer. Your task is to provide a code solution within a single Markdown code block for the given programming problem. Do not include any direct execution commands, test cases, or usage examples within the code block.

Please write code in {language.upper()} and wrap it in a markdown code block:\n```{language}\n<your_code_here>\n```
""" 



    def edit_prompt(self,language:str,previous_code:str,instruction:str, context:Optional[str] = None) -> str:
        lang = "java" if "java" in language else "python"
        
        if language == 'java-repo':
            return f"""I will provide you with a code snippet and an edit instruction. Your task is to edit the code to suit the needs.

## The contexts above the function:
```java
{context}
```
## Previous Code:
```java
{previous_code}
```
## New edit instruction:
{instruction}
"""
        else:
            return f"""I will provide you with a code snippet and an edit instruction. Your task is to edit the code to suit the needs.

Previous Code:
```{lang}
{previous_code}
```

New edit instruction:
{instruction}
"""

    def chat_for_eval(self, messages:List[Any],language: str):
        """
        评估专用的方法

        Return: 
            solution: 格式化后的结果
            raw_solution: LLM的回复
            usage_dict: token使用情况
        """
        try:
            llm_response = self.llm.invoke(messages)
            usage_dict = llm_response.response_metadata.get("token_usage",{})
            raw_solution = llm_response.content if hasattr(llm_response, "content") else str(llm_response)
            # 寻找</think>之后的字符串
            split_raw_solution = raw_solution.split('</think>')[-1]
            # 构建匹配指定语言的代码块的正则表达式
            pattern = rf"```{re.escape(language)}\s*\n(.*?)```"
            match = re.search(pattern, split_raw_solution, re.DOTALL | re.IGNORECASE)
            if match:
                solution = match.group(1).strip()
            else:
                # 如果没找到指定语言的代码块，尝试找任意代码块
                fallback_match = re.search(r"```[a-zA-Z]*\s*\n(.*?)```", split_raw_solution, re.DOTALL)
                solution = fallback_match.group(1).strip() if fallback_match else split_raw_solution.strip()
        except openai.APITimeoutError as e:
            print("⚠️⚠️⚠️Api发生超时，已将solution和raw_solution标记为超时！⚠️⚠️⚠️")
            solution = "timeout"
            raw_solution = "timeout"
            usage_dict = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
        return solution, raw_solution, usage_dict
