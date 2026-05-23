"""
step1_generate_data.py
======================
功能：调用API，自动生成题目难度分类数据集，保存为 Alpaca JSON 格式。
运行：python step1_generate_data.py
输出：data/raw_dataset.json

依赖安装：pip install openai
"""

import os
import json
import time
import random
from openai import OpenAI

# ── 配置区（只需改这里）─────────────────────────────────────────────────────

API_KEY  = ""   # 你的key
BASE_URL = " "# 你的key的网址
MODEL    = "qwen-plus"  # 可选: qwen-turbo, qwen-max, qwen-long
OUTPUT_PATH = "data/raw_dataset.json"

# 每个学科、每个难度生成多少题（总量 = 科目数 × 难度数 × 每批数量）
SUBJECTS = ["数学", "语文", "英语", "物理", "化学", "历史"]
LEVELS   = ["简单", "中等", "困难", "竞赛"]
PER_BATCH = 8     # 每次 API 调用生成几题，建议 5-10
SLEEP_SEC = 1.5   # 每次请求间隔（避免触发频率限制）

# ── 提示词模板 ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """你是一位专业的教育题目生成专家。
请严格按照 JSON 格式输出，不要有任何多余文字。"""

def build_user_prompt(subject: str, level: str, n: int) -> str:
    level_desc = {
        "简单": "小学或初中基础题，概念直接考查，步骤不超过2步",
        "中等": "初中或高中常规题，需要2-3个知识点组合",
        "困难": "高中难题或竞赛入门题，需要综合多个知识点",
        "竞赛": "全国竞赛或奥林匹克级别，需要创造性思维",
    }
    return f"""请生成 {n} 道{subject}学科的{level}难度题目。
难度标准：{level_desc[level]}

输出格式（纯 JSON 数组，不要加任何说明）：
[
  {{
    "instruction": "判断下面这道题目的难度等级，从【简单、中等、困难、竞赛】中选一个输出。",
    "input": "【题目内容写在这里，要完整，包含题干和选项（如有）】",
    "output": "{level}"
  }}
]

注意：
- 题目要真实、完整，不能是假题目
- instruction 固定为上面那句话
- output 只能是：简单、中等、困难、竞赛 之一
- 生成 {n} 道，放在同一个 JSON 数组里"""


# ── API 调用 ─────────────────────────────────────────────────────────────────

def call_api(client: OpenAI, subject: str, level: str, n: int) -> list[dict]:
    """调用API 生成一批题目，返回解析后的列表"""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_user_prompt(subject, level, n)},
            ],
            temperature=0.8,
            max_tokens=2000,
        )
        raw = response.choices[0].message.content.strip()

        # 清理可能的 markdown 代码块标记
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        items = json.loads(raw)
        # 验证格式
        valid = []
        for item in items:
            if all(k in item for k in ["instruction", "input", "output"]):
                if item["output"] in LEVELS:
                    valid.append(item)
        return valid

    except json.JSONDecodeError as e:
        print(f"  ⚠️  JSON 解析失败（{subject}/{level}）：{e}")
        return []
    except Exception as e:
        print(f"  ⚠️  API 调用失败（{subject}/{level}）：{e}")
        return []


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("data", exist_ok=True)

    if API_KEY == "your_api_key_here":
        print(" 请先在脚本顶部填入你的 Key！")

        return

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    all_data = []
    total_calls = len(SUBJECTS) * len(LEVELS)
    call_count  = 0

    print("=" * 55)
    print("  Step 1：生成题目数据集")
    print(f"  计划：{len(SUBJECTS)} 科目 × {len(LEVELS)} 难度 × {PER_BATCH} 题")
    print(f"  预计总量：{total_calls * PER_BATCH} 条")
    print("=" * 55)

    for subject in SUBJECTS:
        for level in LEVELS:
            call_count += 1
            print(f"[{call_count}/{total_calls}] 生成 {subject} - {level} ...", end=" ", flush=True)

            items = call_api(client, subject, level, PER_BATCH)
            all_data.extend(items)
            print(f"✓ 获得 {len(items)} 条（累计 {len(all_data)} 条）")

            time.sleep(SLEEP_SEC)

    # 打乱顺序
    random.shuffle(all_data)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print(f"\n完成！共生成 {len(all_data)} 条数据")
    print(f"   已保存到：{OUTPUT_PATH}")
    print(f"\n   下一步：运行 python step2_kmeans_filter.py")


if __name__ == "__main__":
    main()
