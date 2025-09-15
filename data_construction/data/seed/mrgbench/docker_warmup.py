#!/usr/bin/env python3
import json
import time
from pathlib import Path
from typing import List
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# 假设你的JavaRepoClient在这个路径
from java_repo_client import JavaRepoClient


class ContainerWarmUp:
    """容器预热管理器"""
    
    def __init__(self, container_names: List[str], dataset_path: Path):
        self.container_names = container_names
        self.dataset_path = dataset_path
        self.clients = {}
        
        # 初始化所有客户端
        print("正在连接Docker容器...")
        for container_name in container_names:
            try:
                client = JavaRepoClient(container_name)
                self.clients[container_name] = client
                print(f"✅ 成功连接容器: {container_name}")
            except Exception as e:
                print(f"❌ 连接容器 {container_name} 失败: {e}")
        
        if not self.clients:
            raise RuntimeError("没有可用的Docker容器")
    
    def load_dataset(self) -> List[dict]:
        """加载数据集"""
        print(f"正在加载数据集: {self.dataset_path}")
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"数据集加载完成，共 {len(data)} 条样本")
        return data
    
    def warm_up_container(self, container_name: str, data: List[dict]) -> int:
        """为单个容器预热"""
        client = self.clients[container_name]
        success_count = 0
        
        print(f"🔥 开始预热容器: {container_name}")
        
        for idx, item in enumerate(data):
            try:
                repo = item.get("project", "")
                func_start = item.get("func_start", 0)
                func_end = item.get("func_end", 0)
                func = item.get("func", "")
                file_path = item.get("file_path", "")
                test_start = item.get("test_start", 0)
                test_end = item.get("test_end", 0)
                test_code = item.get("test_code", "")
                test_file = item.get("test_file", "")
                test_instruction = item.get("test_instruction", "")

                # 执行测试来触发依赖下载
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
                
                # 打印进度（每10个样本打印一次）
                if (idx + 1) % 10 == 0:
                    print(f"  容器 {container_name}: 已处理 {idx + 1}/{len(data)}, 成功: {success_count}")
                    
            except Exception as e:
                print(f"❌ 容器 {container_name} 处理样本 {idx} 时出错: {e}")
                continue
        
        print(f"✅ 容器 {container_name} 预热完成，成功: {success_count}/{len(data)}")
        return success_count
    
    def warm_up_sequential(self):
        """顺序预热所有容器"""
        data = self.load_dataset()
        total_success = 0
        
        for container_name in self.container_names:
            if container_name in self.clients:
                success_count = self.warm_up_container(container_name, data)
                total_success += success_count
            else:
                print(f"⚠️  跳过未连接的容器: {container_name}")
        
        print(f"\n🎉 所有容器预热完成！总成功样本数: {total_success}")
        return total_success
    
    def warm_up_parallel(self, max_workers: int = None):
        """并行预热所有容器"""
        data = self.load_dataset()
        
        if max_workers is None:
            max_workers = len(self.clients)
        
        print(f"使用 {max_workers} 个线程并行预热...")
        
        total_success = 0
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 为每个容器提交预热任务
            future_to_container = {
                executor.submit(self.warm_up_container, container_name, data): container_name
                for container_name in self.clients.keys()
            }
            
            # 收集结果
            for future in tqdm(as_completed(future_to_container), total=len(future_to_container), desc="预热进度"):
                container_name = future_to_container[future]
                try:
                    success_count = future.result()
                    results.append((container_name, success_count))
                    total_success += success_count
                except Exception as e:
                    print(f"❌ 容器 {container_name} 预热失败: {e}")
                    results.append((container_name, 0))
        
        # 打印每个容器的结果
        print("\n📊 各容器预热结果:")
        for container_name, success_count in results:
            print(f"  {container_name}: {success_count}/{len(data)}")
        
        print(f"\n🎉 并行预热完成！总成功样本数: {total_success}")
        return total_success


def main():
    """主函数"""
    # 配置参数
    CONTAINER_NAMES = [f"mrgbench_container_{i}" for i in range(1, 13)]  # 12个容器
    DATASET_PATH = Path(__file__).parent / 'v3-MRGBench.json'
    
    # 创建预热管理器
    warm_up = ContainerWarmUp(CONTAINER_NAMES, DATASET_PATH)
    
    print("=" * 60)
    print("Docker容器冷启动预热工具")
    print("=" * 60)
    print(f"容器数量: {len(CONTAINER_NAMES)}")
    print(f"数据集: {DATASET_PATH.name}")
    print("=" * 60)
    
    # 选择预热模式
    print("请选择预热模式:")
    print("1. 完整预热（所有样本）")
    print("2. 并行预热（所有样本）")
    
    choice = "2"
    
    start_time = time.time()
    
    if choice == "2":
        print("\n🚀 开始并行预热...")
        total_success = warm_up.warm_up_parallel()
    else:
        print("\n🐢 开始顺序预热...")
        total_success = warm_up.warm_up_sequential()
    
    elapsed_time = time.time() - start_time
    print(f"\n⏰ 总耗时: {elapsed_time:.2f} 秒")
    print(f"📈 平均每个容器耗时: {elapsed_time / len(warm_up.clients):.2f} 秒")
    
    # 预期结果检查
    expected_success = len(warm_up.load_dataset()) * len(warm_up.clients)
    print(f"📊 预期成功数: {expected_success}, 实际成功数: {total_success}")
    
    if total_success >= expected_success * 0.8:  # 80%成功率视为正常
        print("✅ 预热成功！容器已准备好进行评测")
    else:
        print("⚠️  预热结果不理想，请检查容器状态")


if __name__ == "__main__":
    main()