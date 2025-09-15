import docker
import tempfile
import os
import traceback
import threading
from typing import Tuple,Optional,List
import re
from queue import Queue
import time

class JavaRepoClient:
    """
    Java仓库客户端
    支持功能：
        1. 操作docker容器内部的Java仓库。
        2. 对docker容器内部的java仓库进行文件替换
        3. 给定一个mvn test的命令，对docker容器内部的java仓库运行测试，并返回测试状态和测试运行时信息
    
    容器默认：java_env
    容器内部目录：/langchain4j, /mybatis-flex, /spring-ai
    """

    def __init__(self,container_name:str) -> None:
        # determined_fermi
        self.container_name = container_name
        self.docker_client = docker.from_env()
        try:
            self.container = self.docker_client.containers.get(self.container_name)
        except docker.errors.NotFound:
            raise RuntimeError(f"容器 {container_name} 不存在")
        except docker.errors.APIError as e:
            raise RuntimeError(f"获取容器失败: {e}")

    def _exec_in_container(self, cmd: str, workdir: str = "/") -> Tuple[int, str]:
        """
        在容器内部执行命令
        返回 (exit_code, output)
        """
        exec_id = self.docker_client.api.exec_create(
            self.container.id, cmd, workdir=workdir
        )
        output = self.docker_client.api.exec_start(exec_id).decode("utf-8", errors="ignore")
        exit_code = self.docker_client.api.exec_inspect(exec_id)["ExitCode"]
        return exit_code, output

    def _read_file(self, path: str) -> list[str]:
        code, content = self._exec_in_container(f"cat {path}")
        if code != 0:
            raise RuntimeError(f"读取文件失败: {path}\n{content}")
        return content.splitlines()

    def _write_file(self, container_path: str, lines: list[str]) -> None:
        """
        将本地临时文件写入容器对应路径
        """
        text = "\n".join(lines) + "\n"
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            tmp.write(text)
            tmp_path = tmp.name
        try:
            self.docker_client.api.put_archive(
                self.container.id,
                path=os.path.dirname(container_path),
                data=self._make_tar(tmp_path, os.path.basename(container_path))
            )
        finally:
            os.remove(tmp_path)

    def _make_tar(self, src: str, arcname: str):
        """
        打包一个文件为 tar，用于 docker cp
        """
        import io, tarfile
        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            tar.add(src, arcname=arcname)
        tar_stream.seek(0)
        return tar_stream

    def backup_file(self, container_path: str) -> str:
        bak_path = container_path + ".bak"
        self._exec_in_container(f"cp {container_path} {bak_path}")
        return bak_path

    def restore_file(self, container_path: str) -> None:
        bak_path = container_path + ".bak"
        self._exec_in_container(f"mv {bak_path} {container_path}")

    def replace_code_block(self, repo: str, file_path: str, start_line: int, end_line: int, new_code: str) -> None:
        """
        替换 repo/{file_path} 文件 [start_line, end_line] 区间的内容为 new_code
        注意：start_line, end_line 从 1 开始计数
        """
        container_path = f"/{repo}/{file_path}"
        lines = self._read_file(container_path)
        new_lines = new_code.splitlines()
        updated = lines[: start_line - 1] + new_lines + lines[end_line:]
        self._write_file(container_path, updated)

    def filter_maven_output(self,output: str) -> str:
        """
        过滤 Maven 命令输出，只保留对模型有用的错误和测试失败信息。
        """
        # 分行处理
        lines = output.splitlines()
        
        # 关键结果容器
        filtered_lines = []
        
        # 状态标记
        in_error_section = False
        in_stack_trace = False
        capture_next_lines = 0  # 用于捕获错误后的几行上下文

        # 正则：匹配编译错误行（包含具体文件、行号、错误描述）
        compile_error_pattern = re.compile(r'\[ERROR\]\s+.+:\[?\d+,\d+\]?\s')
        # 匹配测试失败（JUnit）
        test_failure_pattern = re.compile(r'(?i)^\s*Tests? (run|failed)|^\[ERROR\].*test')
        # 匹配堆栈跟踪
        stack_trace_pattern = re.compile(r'\s*at \w+')
        # 匹配关键错误摘要
        critical_error_pattern = re.compile(r'\[ERROR\].*(failure|exception|error)', re.IGNORECASE)

        for line in lines:
            stripped = line.strip()

            # 跳过空行和纯 INFO 行（但保留可能包含关键信息的 ERROR）
            if stripped.startswith('[INFO]'):
                continue

            # 保留所有 ERROR 行
            if '[ERROR]' in line:
                filtered_lines.append(line)
                in_error_section = True
                capture_next_lines = 3  # 捕获后续几行上下文
                continue

            # 捕获测试失败信息
            if re.search(test_failure_pattern, stripped):
                filtered_lines.append(line)
                capture_next_lines = 5
                continue

            # 捕获堆栈跟踪（at xxx.xxx）
            if re.match(stack_trace_pattern, stripped):
                if len(filtered_lines) > 0 or in_error_section:
                    filtered_lines.append(line)
                    in_stack_trace = True
                    capture_next_lines = 2
                continue

            # 捕获编译错误细节（非 ERROR 前缀但实际是错误内容，如 method not applicable）
            if in_error_section or capture_next_lines > 0:
                filtered_lines.append(line)
                capture_next_lines -= 1
                continue

            # 特别捕获：编译错误中的详细解释（如 argument mismatch）
            if re.search(compile_error_pattern, line):
                filtered_lines.append(line)
                in_error_section = True
                capture_next_lines = 4
                continue

            # 捕获关键异常类型
            if re.search(critical_error_pattern, line):
                filtered_lines.append(line)
                capture_next_lines = 3
                continue

        # 合并结果
        result = '\n'.join(filtered_lines).strip()

        # 如果没有捕获到任何错误，返回 "SUCCESS" 表示通过
        if not result:
            return "SUCCESS"

        # 去重连续空行
        result = re.sub(r'\n+', '\n', result)
        return result.strip()

    def run_mvn_test(self, repo: str, test_cmd: str) -> Tuple[bool, str]:
        """
        在容器中进入 repo 执行 mvn test
        返回 (是否成功, 输出日志)
        """
        code, output = self._exec_in_container(test_cmd, workdir=f"/{repo}")
        return code == 0, self.filter_maven_output(output)

    def safe_replace_and_test(
        self,
        repo: str,
        file_path: str,
        func_start: int,
        func_end: int,
        new_code: str,
        test_file: str,
        test_start: int,
        test_end: int,
        new_test: str,
        test_instruction: str,
    ) -> Tuple[bool, str]:
        """
        完整流程：
        1. 备份代码文件 & 测试文件
        2. 替换对应行
        3. 执行 mvn test
        4. 无论成功失败，恢复备份
        """
        container_file = f"/{repo}/{file_path}"
        container_test = f"/{repo}/{test_file}"

        try:
            # 1. 备份
            self.backup_file(container_file)
            self.backup_file(container_test)

            # 2. 替换
            self.replace_code_block(repo, file_path, func_start, func_end, new_code)
            self.replace_code_block(repo, test_file, test_start, test_end, new_test)

            if repo == 'spring-ai':
                print("遇到spring-ai，准备执行代码格式化工具")
                code,output = self._exec_in_container("mvn spring-javaformat:apply",workdir=f"/{repo}")
                if code == 0:
                    print("代码格式化成功")
                else:
                    print(f"代码格式化失败: \n {self.filter_maven_output(output)}")
            print(f"替换成功: {file_path}")
            print(f"替换成功: {test_file}, 正在执行测试... ")

            # 3. 执行测试
            success, output = self.run_mvn_test(repo, test_instruction)
            return success, output
        except Exception as e:
            tb = traceback.format_exc()
            return False, f"异常发生: {e}\n{tb}"
        finally:
            # 4. 回滚
            try:
                self.restore_file(container_file)
                self.restore_file(container_test)
                print(f"回滚成功: {file_path}")
                print(f"回滚成功: {test_file}")
            except Exception as e:
                print(f"回滚失败: {e}")

    def read_code_block(self, repo: str, file_path: str, start_line: int, end_line: int) -> str:
        """
        从容器中读取 repo/{file_path} 的指定行区间 [start_line, end_line] 的代码
        行号从 1 开始
        """
        container_path = f"/{repo}/{file_path}"
        lines = self._read_file(container_path)
        # 提取区间
        selected = lines[start_line - 1:end_line]
        return "\n".join(selected)


class JavaRepoClientPool:
    """JavaRepoClient 连接池"""
    
    def __init__(self, container_names: List[str]):
        self.available_clients = Queue()
        self.lock = threading.Lock()
        self.clients = {}
        
        # 初始化所有客户端
        for container_name in container_names:
            try:
                client = JavaRepoClient(container_name)
                self.available_clients.put(client)
                self.clients[container_name] = client
                print(f"✅ 成功连接容器: {container_name}")
            except Exception as e:
                print(f"❌ 连接容器 {container_name} 失败: {e}")
    
    def acquire(self, timeout: Optional[float] = 30.0) -> Optional[JavaRepoClient]:
        """获取一个可用的客户端"""
        try:
            return self.available_clients.get(timeout=timeout)
        except:
            return None
    
    def release(self, client: JavaRepoClient):
        """释放客户端回连接池"""
        self.available_clients.put(client)
    
    def get_pool_size(self) -> int:
        """获取连接池大小"""
        return self.available_clients.qsize()
    
    def shutdown(self):
        """关闭连接池"""
        while not self.available_clients.empty():
            try:
                self.available_clients.get_nowait()
            except:
                break



if __name__ == '__main__':
    from pathlib import Path
    import json
    from tqdm import tqdm

    client = JavaRepoClient()

    # 读取数据集
    json_path = Path(__file__).parent / 'v3-MRGBench.json'
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    success_count = 0
    
    for idx, item in tqdm(enumerate(data), total=len(data)):
        repo = item.get("project", "")
        func_start = item.get("func_start", 0)
        func_end = item.get("func_end", 0)
        func = item.get("func","")
        file_path = item.get("file_path","")
        test_start = item.get("test_start", 0)
        test_end = item.get("test_end", 0)
        test_code = item.get("test_code","")
        test_file = item.get("test_file", "")
        test_instruction = item.get("test_instruction","")
    
        success, log = client.safe_replace_and_test(
            repo=repo,
            file_path=file_path,
            func_start=func_start,
            func_end=func_end,
            new_code=func,
            test_file=test_file,
            test_start=test_start,
            test_end=test_end,
            new_test=test_code,
            test_instruction=test_instruction
        )
        if success:
            success_count += 1
        print(f"idx = {idx},是否成功:", success)
        print("执行日志:\n", log)
    
    # 如果是46说明正常
    print(f"最终执行成功的样本: {success_count}")
