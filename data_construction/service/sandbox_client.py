import requests
from typing import Dict, Any, Optional, List
from abc import ABC,abstractmethod
import time
import aiohttp
import asyncio
import threading

from data_construction.data.seed.mrgbench.java_repo_client import JavaRepoClient,JavaRepoClientPool

class BaseSandboxClient(ABC):

    def __init__(self,
                dataset:str,
                base_url:Optional[str] = None,
                verify_entrypoint: Optional[str] = None,
        ) -> None:
        """初始化沙盒客户端"""
        self.dataset = dataset
        self.base_url = base_url
        self.verify_entrypoint = verify_entrypoint or  f"{self.base_url}/verify/"

    @abstractmethod
    def execute_code_with_test(
        self,
        code: str,
        test: str,
    ) -> Dict[str, Any]:
        """
        将测试和代码拼接送入沙盒，验证代码的正确性。
        
        :param code: The code to test
        :param test: The test code
        :return: A dictionary containing the execution result, including status and details fields
        :raises: If the request fails or the server returns an error, an exception will be raised
        """
        ...
    
    def check_health(self) -> bool:
        """
        Check if the service is running normally
        
        :return: If the service is normal, return True, otherwise return False
        """
        try:
            response = requests.get(f"{self.base_url}/")
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

class BigCodeSandboxClient(BaseSandboxClient):

    def __init__(self,
                dataset:Optional[str] = "bigcodebench",
                base_url:Optional[str] = "http://localhost:8199",
                verify_entrypoint: Optional[str] = None,
        ) -> None:
        self.timeout = 180
        self.session = None
        super().__init__(dataset, base_url,verify_entrypoint)
    
    async def init_session(self):
        if self.session is None:
            # 可配置连接池大小
            connector = aiohttp.TCPConnector(limit=100, limit_per_host=10)
            self.session = aiohttp.ClientSession(connector=connector)

    async def close(self):
        if self.session:
            await self.session.close()
    
    async def async_execute_code_with_test(self, code: str, test: str) -> Dict[str, Any]:
        if self.session is None:
            await self.init_session()

        payload = {
            "code": code,
            "test": test,
            "entry_point": "task_func"
        }

        # 定义重试的等待时间（秒）
        retry_delays = [1, 2, 4]
        max_retries = len(retry_delays)
        attempt = 0

        while True:
            try:
                async with self.session.post(self.verify_entrypoint, json=payload, timeout=self.timeout) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        print(f"❌沙盒请求错误: {resp.status}: {text[:500]}.")
                        return {
                            "status": "error",
                            "detail": f"HTTP {resp.status}: {text[:500]}"
                        }

                    result = await resp.json()
                    return {
                        "status": result.get("status", "error"),
                        "detail": result.get("details", "") or result.get("detail", "")
                    }

            except (asyncio.TimeoutError, aiohttp.ClientConnectionError, aiohttp.ClientPayloadError, Exception) as e:
                # 打印异常信息
                if isinstance(e, asyncio.TimeoutError):
                    print("请求超时 (120s)")
                    err_detail = "Request timed out (120s)"
                elif isinstance(e, aiohttp.ClientConnectionError):
                    print(f"连接错误: {str(e)}")
                    err_detail = f"Connection error: {str(e)}"
                elif isinstance(e, aiohttp.ClientPayloadError):
                    print(f"Payload解码错误: {str(e)}")
                    err_detail = f"Payload decode error: {str(e)}"
                else:
                    print(f"未知错误 [{type(e).__name__}]: {str(e) or repr(e)}")
                    err_detail = f"Unknown error [{type(e).__name__}]: {str(e) or repr(e)}"

                if attempt < max_retries:
                    delay = retry_delays[attempt]
                    print(f"第{attempt+1}次重试，{delay}秒后重试...")
                    await asyncio.sleep(delay)
                    attempt += 1
                    continue
                else:
                    return {
                        "status": "error",
                        "detail": err_detail
                    }


    def execute_code_with_test(
        self,
        code: str,
        test: str,
    ) -> Dict[str, Any]:
        """
        将测试和代码拼接送入沙盒，验证代码的正确性。
        
        :param code: The code to test
        :param test: The test code
        :return: A dictionary containing the execution result, including status and details fields
        :raises: If the request fails or the server returns an error, an exception will be raised
        """

        if not self.check_health():
            raise ValueError("The service is not available, please check if the server is running")
        
        payload = {
            "code": code,
            "test": test,
            "entry_point": "task_func"
        }
        
        try:
            response = requests.post(self.verify_entrypoint, json=payload)
            response.raise_for_status()  # If the response status code is not 200, throw an HTTPError
            return {
                "status": response.json().get("status"),
                "detail": response.json().get("details")
            }
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request failed: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error during execute_code_with_test {e}")

class MrgBenchSandboxClient(BaseSandboxClient):
    """MrgBench的沙盒客户端(支持并行版)"""

    def __init__(self,dataset: str = "mrgbench",  container_names: List[str] = None) -> None:
        # self.java_repo_Client = JavaRepoClient()
        if container_names is None:
            # 默认创建12个容器连接
            container_names = [f"mrgbench_container_{i}" for i in range(1, 13)]
        
        self.client_pool = JavaRepoClientPool(container_names)
        self.lock = threading.Lock()

    def execute_code_with_test(self, code: str, test: str, metadata:Dict[str,Any]) -> Dict[str, Any]:
        # 从连接池获取客户端
        client = self.client_pool.acquire()
        if client is None:
            return {
                "status": "error",
                "detail": "没有可用的Docker容器"
            }
        
        try:
            status, detail = client.safe_replace_and_test(
                repo=metadata.get("repo",""),
                file_path=metadata.get("file_path",""),
                func_start=metadata.get("func_start",""),
                func_end=metadata.get("func_end",""),
                new_code=code,
                test_file=metadata.get("test_file",""),
                test_start=metadata.get("test_start",""),
                test_end=metadata.get("test_end",""),
                new_test=test,
                test_instruction=metadata.get("test_instruction",""),
            )
            if status:
                return {
                    "status": "pass",
                    "detail": detail,
                }
            else:
                return {
                    "status": "fail",
                    "detail": detail,
                }
        except Exception as e:
            return {
                "status": "error",
                "detail": f"执行异常: {str(e)}"
            }
        finally:
            # 确保客户端被释放回连接池
            self.client_pool.release(client)

    def check_health(self) -> bool:
        return self.client_pool.get_pool_size() > 0

    def get_available_clients_count(self) -> int:
        return self.client_pool.get_pool_size()


class AutoCodeBenchSandboxClient(BaseSandboxClient):
    """AutoCodeBench的沙盒客户端"""
    def __init__(self,
                dataset: str = "autocodebench",
                base_url: str | None = "http://127.0.0.1:7887",
                concurrency: int | None = 32,
                language: str | None = "java",
                verify_entrypoint: str | None = None) -> None:
        self.concurrency = concurrency
        self.language = language
        self.verify_entrypoint = f"{base_url}/submit"
        self.headers = {
            "Content-Type": "application/json"
        }
        self.timeout = 180
        self.session = None
        super().__init__(dataset, base_url, self.verify_entrypoint)
    
    async def init_session(self):
        if self.session is None:
            # 可配置连接池大小
            connector = aiohttp.TCPConnector(limit=100, limit_per_host=10)
            self.session = aiohttp.ClientSession(connector=connector)

    async def close(self):
        if self.session:
            await self.session.close()

    async def async_execute_code_with_test(self, code: str, test: str) -> Dict[str, Any]:
        """
        异步调用Autocodebench的沙盒客户端
        """
        if self.session is None:
            await self.init_session()

        payload = {
            "src_uid": f"0710_bench_test_full_{int(time.time())}",
            "func_code": code,  # code solution
            "main_code": test,  # test function
            "lang": self.language,
            "show_log": "true",
            "request_extensions": {"timeout": 30, "debug": str(False).lower()}
        }

        retry_delays = [1, 2, 4]
        max_retries = len(retry_delays)
        attempt = 0

        while True:
            try:
                async with self.session.post(self.verify_entrypoint, headers=self.headers, json=payload, timeout=self.timeout) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        print(f"❌沙盒请求错误: {resp.status}: {text[:500]}.")
                        return {
                            "status": "fail",
                            "detail": f"HTTP {resp.status}: {text[:500]}",
                            "status_code": resp.status
                        }
                    result = await resp.json()
                    response_extension = result.get("response_extensions",{})
                    exec_outcome = result.get("exec_outcome")
                    stdout = response_extension.get("stdout","")
                    stderr = response_extension.get("stderr","")
                    exec_runtime_message = response_extension.get("exec_runtime_message")
                    if exec_outcome == 'PASSED':
                        return {
                            "status": "pass",
                            "detail": f"===exec_outcome===\n{exec_outcome}\n===exec_runtime_message===\n{exec_runtime_message}\n===stdout:===\n{stdout}\n===stderr:===\n{stderr}",
                        }
                    else:
                        return {
                            "status": "fail",
                            "detail": f"===exec_outcome===\n{exec_outcome}\n===exec_runtime_message===\n{exec_runtime_message}\n===stdout:===\n{stdout}\n===stderr:===\n{stderr}",
                        }
            except (asyncio.TimeoutError, aiohttp.ClientConnectionError, aiohttp.ClientPayloadError, Exception) as e:
                if isinstance(e, asyncio.TimeoutError):
                    print("请求超时 (180s)")
                    err_detail = "Request timed out (180s)"
                elif isinstance(e, aiohttp.ClientConnectionError):
                    print(f"连接错误: {str(e)}")
                    err_detail = f"Connection error: {str(e)}"
                elif isinstance(e, aiohttp.ClientPayloadError):
                    print(f"Payload解码错误: {str(e)}")
                    err_detail = f"Payload decode error: {str(e)}"
                else:
                    print(f"未知错误 [{type(e).__name__}]: {str(e) or repr(e)}")
                    err_detail = f"Unknown error [{type(e).__name__}]: {str(e) or repr(e)}"

                if attempt < max_retries:
                    delay = retry_delays[attempt]
                    print(f"第{attempt+1}次重试，{delay}秒后重试...")
                    await asyncio.sleep(delay)
                    attempt += 1
                    continue
                else:
                    return {
                        "status": "fail",
                        "detail": err_detail,
                        "status_code": None
                    }

    def execute_code_with_test(self, code: str, test: str) -> Dict[str, Any]:
        """调用Autocodebench的沙盒客户端"""
        if not self.check_health():
            raise ValueError("The service is not available, please check if the server is running")
        
        try:
            payload = {
                "src_uid": f"0710_bench_test_full_{int(time.time())}",
                "func_code": code,  # code solution
                "main_code": test,  # test function
                "lang": self.language,
                "show_log": "true",
                "request_extensions": {"timeout": 30, "debug": str(False).lower()}
            }

            response = requests.post(self.verify_entrypoint, headers=self.headers, json=payload, timeout=60)

            if response.status_code == 200:
                result = response.json()
                response_extension = result.get("response_extensions",{})
                exec_outcome = result.get("exec_outcome")
                stdout = response_extension.get("stdout","")
                stderr = response_extension.get("stderr","")
                exec_runtime_message = response_extension.get("exec_runtime_message")
                if exec_outcome == 'PASSED':
                    # 通过
                    return {
                        "status": "pass",
                        "detail": f"===exec_outcome===\n{exec_outcome}\n===exec_runtime_message===\n{exec_runtime_message}\n===stdout:===\n{stdout}\n===stderr:===\n{stderr}",
                    }
                else:
                    # 不通过
                    return {
                        "status": "fail",
                        "detail": f"===exec_outcome===\n{exec_outcome}\n===exec_runtime_message===\n{exec_runtime_message}\n===stdout:===\n{stdout}\n===stderr:===\n{stderr}",
                    }
            else:
                print(f"API调用失败，状态码: {response.status_code}, 响应: {response.text}")
                return {
                    "status": "fail",
                    "detail": f"HTTP {response.status_code}: {response.text}",
                    "status_code": response.status_code
                }
        except Exception as e:
            print(f"沙盒发生异常: {e}")
            return {
                "status": "fail",
                "error": "沙盒发生异常:" + str(e),
                "status_code": None
            }
        
    def check_health(self) -> bool:
        """
        检查服务是否正常

        :return: 如果服务正常返回True，否则返回False
        """
        url = "http://127.0.0.1:7887/submit"
        headers = {
            "Content-Type": "application/json"
        }
        payload = {
            "src_uid": "test-001",
            "lang": "python",
            "source_code": "print(\"Hello World\")"
        }

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=5)
            # 检查 HTTP 状态码是否为 200
            if response.status_code != 200:
                return False

            # 解析 JSON 响应
            data = response.json()

            # 检查 exec_outcome 是否存在且等于 "PASSED"
            return data.get("exec_outcome") == "PASSED"

        except (requests.RequestException, ValueError):
            # 捕获网络异常、超时、JSON 解析错误等
            return False