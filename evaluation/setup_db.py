import os
from pathlib import Path
from data_construction.data.data_manager import MTDataManager

# 获取当前文件的目录
current_dir = Path(__file__).parent
# 获取dataset目录的路径（相对于当前文件的上级目录）
dataset_dir = current_dir.parent / "dataset"

# 四个数据集文件
dataset_files = [
    "java_function.jsonl",
    "java_repository.jsonl",
    "python_function.jsonl",
    "python_repository.jsonl"
]

# 创建一个临时合并文件
merged_file = current_dir / "merged_dataset.jsonl"

# 合并所有数据集到一个文件
with open(merged_file, 'w', encoding='utf-8') as outfile:
    for dataset_file in dataset_files:
        dataset_path = dataset_dir / dataset_file
        print(f"正在读取: {dataset_path}")
        with open(dataset_path, 'r', encoding='utf-8') as infile:
            for line in infile:
                if line.strip():  # 跳过空行
                    outfile.write(line)

print(f"已合并所有数据集到: {merged_file}")

# 初始化数据库
manager = MTDataManager()
print("正在初始化数据库...")
manager.setup_with_jsonl(merged_file)
print(f"数据库初始化完成！共导入 {manager.count()} 条数据")

# 清理临时文件
if merged_file.exists():
    os.remove(merged_file)
    print(f"已删除临时文件: {merged_file}")
