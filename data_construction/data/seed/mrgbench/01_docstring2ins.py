import json
from pathlib import Path
from data_construction.agent.decomposer import InstructionGeneratorAgent, InstructionGenerationOutput
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

if __name__ == '__main__':
    # 1. 读取JSON
    json_path = Path(__file__).parent / 'MRGBench.json'
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. 只处理 body_len >= 10 的样本
    filtered_data = [item for item in data if item.get('body_len', 0) >= 10]
    print(f"原始样本数: {len(data)}，过滤后样本数: {len(filtered_data)}")

    # 3. 初始化大模型Agent
    agent = InstructionGeneratorAgent()

    # 4. 遍历每一条样本，生成prompt，调用大模型
    for idx, item in tqdm(enumerate(filtered_data), total=len(filtered_data)):
        docstring = item.get('comment', "")
        code = item.get('func', "")

        user_prompt = f"""--- Docstring ---
{docstring}
--- Code ---
{code}
"""

        try:
            output, _ = agent.single_turn_chat_with_structure_output(user_prompt=user_prompt)
            instruction = output.instruction
        except Exception as e:
            print(f"第{idx}个样本生成instruction失败: {e}")
            instruction = ""

        item['instruction'] = instruction

    # 5. 保存到新的JSON文件
    output_path = Path(__file__).parent / 'v2-MRGBench.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)
    print(f"已保存带有instruction的新文件到: {output_path}")
