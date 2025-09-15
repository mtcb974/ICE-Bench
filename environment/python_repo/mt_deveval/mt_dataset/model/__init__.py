import openai
import time
from typing import List,Dict,Any
import traceback
import os

LLM_CONFIG_DATA_PIPELINE = {
    "model": "gpt-4.1",
    "temperature": 0.6,
    "n": 1,
}

LLM_CONFIG_INFERENCE = {
    "model": "deepseek-v3",
    "temperature": 0,
    "n": 1
}


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

def get_gpt_client(src):
    if src == 'openai':
        return openai.OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    elif src == 'deepinfra':
        return openai.OpenAI(api_key=OPENAI_API_KEY, base_url="https://api.deepinfra.com/v1/openai")
    elif src == 'deepseek':
        return openai.OpenAI(api_key=OPENAI_API_KEY, base_url="https://api.deepseek.com/v1")
    elif src == 'openrouter':
        return openai.OpenAI(api_key=OPENAI_API_KEY,base_url="https://openrouter.ai/api/v1")
    
def request_chatgpt_engine(config,backend:str, base_url=None, max_retries=5, timeout=100):
    ret = None
    retries = 0

    client = get_gpt_client(backend)

    while ret is None and retries < max_retries:
        try:
            print("Creating API request")
            ret = client.chat.completions.create(**config)
        except openai.OpenAIError as e:
            if isinstance(e, openai.BadRequestError):
                print("Request invalid")
                traceback.print_exc()
                raise Exception("Invalid API Request")
            elif isinstance(e, openai.RateLimitError):
                print("Rate limit exceeded. Waiting...")
                print(e)
                time.sleep(5)
            elif isinstance(e, openai.APIConnectionError):
                print("API connection error. Waiting...")
                print(e)
                traceback.print_exc()
                time.sleep(5)
            else:
                print("Unknown error. Waiting...")
                print(e)
                time.sleep(1)

        retries += 1

    print(f"Successfully get API response.")
    return ret

def call_llm(messages: List[Dict[str, str]]) -> str:
    request_config = LLM_CONFIG_DATA_PIPELINE.copy()
    request_config["messages"] = messages
    ret = request_chatgpt_engine(request_config,backend="openrouter")
    return ret.choices[0].message.content

def call_llm_inference(llm:str,messages: List[Dict[str,str]],backend: str) -> str:
    request_config = LLM_CONFIG_INFERENCE.copy()
    request_config["model"] = llm
    request_config["messages"] = messages
    if llm == 'o4-mini':
        request_config["reasoning_effort"] = "medium"
    ret = request_chatgpt_engine(request_config,backend=backend)
    return ret.choices[0].message.content