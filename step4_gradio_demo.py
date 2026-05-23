"""
step5_gradio_demo.py
====================
功能：启动 Gradio 可视化演示界面，输入题目文本，输出难度等级预测。
      支持调用本地 LoRA 模型（有 GPU）或 API（无 GPU）两种模式。
运行：python step4_gradio_demo.py
界面：自动在浏览器打开 http://localhost:7860

依赖安装：pip install gradio openai
"""

import json
import gradio as gr
from datetime import datetime

# ── 配置 ──────────────────────────────────────────────────────────────────────

# 模式选择：
#   "api"   → 调用 阿里云百炼 API（本地无 GPU 时用，需要 API Key）
#   "local" → 调用本地 LoRA 模型（有 GPU 时用，需先下载权重）
MODE = "local"

API_KEY = ""
BASE_URL = ""

LOCAL_MODEL_PATH   = "Qwen/Qwen2.5-0.5B-Instruct"
LOCAL_ADAPTER_PATH = "saves/lora_weights/checkpoint-45"

LABELS = ["简单", "中等", "困难", "竞赛"]
LABEL_COLORS = {
    "简单": "#4CAF50",
    "中等": "#2196F3",
    "困难": "#FF9800",
    "竞赛": "#F44336",
}
LABEL_EMOJI = {"简单": "🟢", "中等": "🔵", "困难": "🟠", "竞赛": "🔴"}

# 历史记录（内存存储，重启清空）
history_log = []


# ── 模型加载（local 模式）────────────────────────────────────────────────────

_local_model = None
_local_tokenizer = None

def load_local_model():
    global _local_model, _local_tokenizer
    if _local_model is not None:
        return
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    import torch
    print("[模型] 加载本地 LoRA 模型...")
    _local_tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_PATH, trust_remote_code=True,
        torch_dtype=torch.float16, device_map="auto"
    )
    _local_model = PeftModel.from_pretrained(base, LOCAL_ADAPTER_PATH)
    _local_model.eval()
    print("[模型] 加载完成")


# ── 推理函数 ──────────────────────────────────────────────────────────────────

def predict_api(question: str) -> str:
    """调用 API 预测难度（无 GPU 时用）"""
    from openai import OpenAI
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    system = (
        "你是一个题目难度判断专家。"
        "用户输入一道题目，你只需输出难度等级，"
        "只能从【简单、中等、困难、竞赛】中选一个词输出，不要有任何其他内容。"
    )
    resp = client.chat.completions.create(
        model="-chat",
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": f"请判断这道题的难度：\n{question}"},
        ],
        temperature=0.1,
        max_tokens=10,
    )
    result = resp.choices[0].message.content.strip()
    for label in LABELS:
        if label in result:
            return label
    return "中等"


def predict_local(question: str) -> str:
    """调用本地 LoRA 模型预测难度（有 GPU 时用）"""
    import torch
    load_local_model()
    instruction = "判断下面这道题目的难度等级，从【简单、中等、困难、竞赛】中选一个输出。"
    prompt = instruction + "\n" + question
    inputs = _local_tokenizer(prompt, return_tensors="pt").to(_local_model.device)
    with torch.no_grad():
        output = _local_model.generate(**inputs, max_new_tokens=10, do_sample=False)
    pred = _local_tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()
    for label in LABELS:
        if label in pred:
            return label
    return "中等"


def predict(question: str) -> str:
    if MODE == "api":
        return predict_api(question)
    else:
        return predict_local(question)


# ── 示例题目 ──────────────────────────────────────────────────────────────────

EXAMPLES = [
    ["计算 15 + 27 = ？"],
    ["解方程：2x + 5 = 13，求 x 的值。"],
    ["已知函数 f(x) = x³ - 3x，求 f(x) 在 [-2, 2] 上的最大值和最小值。"],
    ["证明：对任意正整数 n，n(n+1)(n+2) 能被 6 整除。"],
    ["The cat sat on the mat. 这句话用了什么时态？"],
    ["分析《红楼梦》中贾宝玉的人物形象。"],
    ["用辗转相除法求 gcd(252, 105)，写出每步计算过程。"],
]


# ── Gradio 界面逻辑 ────────────────────────────────────────────────────────────

def classify_question(question: str):
    """主预测函数，返回 Gradio 组件更新"""
    if not question.strip():
        return (
            gr.update(value="请输入题目内容", visible=True),
            gr.update(value="", visible=False),
            gr.update(value=""),
        )

    try:
        label = predict(question.strip())
    except Exception as e:
        return (
            gr.update(value=f"预测失败：{str(e)}", visible=True),
            gr.update(visible=False),
            gr.update(value=""),
        )

    color = LABEL_COLORS.get(label, "#9E9E9E")
    emoji = LABEL_EMOJI.get(label, "⚪")

    # 记录历史
    history_log.append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "question": question[:40] + "..." if len(question) > 40 else question,
        "label": label,
    })

    result_html = f"""
    <div style="text-align:center; padding:20px; background:#f8f9fa; border-radius:12px; margin:10px 0;">
        <div style="font-size:48px; margin-bottom:8px;">{emoji}</div>
        <div style="font-size:32px; font-weight:bold; color:{color}; margin-bottom:6px;">{label}</div>
        <div style="font-size:14px; color:#666;">难度等级</div>
    </div>
    """

    tips = {
        "简单": "✅ 适合：小学/初中基础练习、课堂随堂测验",
        "中等": "📘 适合：初高中单元测试、期中期末考试",
        "困难": "📙 适合：高中综合题、高考压轴题",
        "竞赛": "🏆 适合：全国数学/物理/化学竞赛、奥赛选拔",
    }

    return (
        gr.update(visible=False),
        gr.update(value=result_html, visible=True),
        gr.update(value=tips.get(label, "")),
    )


def get_history_table():
    if not history_log:
        return "暂无记录"
    rows = "\n".join(
        f"| {r['time']} | {r['question']} | {LABEL_EMOJI.get(r['label'],'')} {r['label']} |"
        for r in reversed(history_log[-10:])
    )
    return f"| 时间 | 题目 | 难度 |\n|------|------|------|\n{rows}"


# ── 构建界面 ──────────────────────────────────────────────────────────────────

with gr.Blocks(title="题目难度智能分类系统", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 📚 题目难度智能分类系统
    基于 **LLaMA-Factory + QLoRA 微调 + K-means 数据筛选** 构建
    输入任意题目，自动判断难度等级：🟢 简单 / 🔵 中等 / 🟠 困难 / 🔴 竞赛
    """)

    with gr.Row():
        with gr.Column(scale=3):
            question_input = gr.Textbox(
                label="输入题目",
                placeholder="在这里粘贴题目内容...",
                lines=5,
                max_lines=10,
            )
            with gr.Row():
                submit_btn = gr.Button("🔍 判断难度", variant="primary", scale=3)
                clear_btn  = gr.Button("🗑️ 清空", scale=1)

            gr.Examples(
                examples=EXAMPLES,
                inputs=question_input,
                label="📝 示例题目（点击填入）",
            )

        with gr.Column(scale=2):
            error_output  = gr.Markdown(visible=False)
            result_output = gr.HTML(visible=False, label="预测结果")
            tips_output   = gr.Textbox(label="适用场景", interactive=False, lines=2)

    with gr.Accordion("📊 历史记录（最近10条）", open=False):
        history_btn    = gr.Button("刷新记录")
        history_output = gr.Markdown("暂无记录")

    # 事件绑定
    submit_btn.click(
        fn=classify_question,
        inputs=question_input,
        outputs=[error_output, result_output, tips_output],
    )
    clear_btn.click(
        fn=lambda: ("", gr.update(visible=False), gr.update(visible=False), ""),
        outputs=[question_input, error_output, result_output, tips_output],
    )
    history_btn.click(fn=get_history_table, outputs=history_output)

    gr.Markdown("""
    ---
    **项目说明**：本系统基于 [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory) 开源框架二次开发，
    引入 K-means 聚类对训练数据进行质量筛选，使用 QLoRA 在 Qwen2.5 上进行 SFT 微调。
    """)


if __name__ == "__main__":
    print("=" * 55)
    print("  Step 5：启动 Gradio 演示界面")
    print(f"  模式：{'API 调用 ' if MODE == 'api' else '本地 LoRA 模型'}")
    print("=" * 55)

    if MODE == "local" and API_KEY == "":
        print("⚠️  请先在脚本顶部填入 API_KEY！")
    else:
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,       # 改为 True 可生成公网链接（方便演示）
            inbrowser=True,    # 自动打开浏览器
        )
