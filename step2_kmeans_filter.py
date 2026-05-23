"""
step2_kmeans_filter.py
======================
功能：对原始数据做 K-means 聚类筛选，去除重复/相似样本，
      保留多样性代表样本，生成更高质量的训练集。
运行：python step2_kmeans_filter.py
输入：data/raw_dataset.json
输出：data/filtered_dataset.json + cluster_visualization.png

依赖安装：pip install sentence-transformers scikit-learn matplotlib
"""

import json
import argparse
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

# ── 默认配置 ──────────────────────────────────────────────────────────────────

INPUT_PATH  = "data/raw_dataset.json"
OUTPUT_PATH = "data/filtered_dataset.json"
K           = 20    # 聚类数量（建议：数据量 / 25 左右）
PER_K       = 5     # 每簇保留代表样本数
MODEL_NAME  = "paraphrase-multilingual-MiniLM-L12-v2"  # 支持中文，首次运行自动下载


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def load_data(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[加载] 共读取 {len(data)} 条数据")
    return data


def save_data(data: list[dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[保存] 已保存到 {path}")


def print_distribution(data: list[dict], title: str):
    counter = Counter(item.get("output", "?").strip() for item in data)
    print(f"\n  {'─'*40}")
    print(f"  {title} 标签分布（共 {len(data)} 条）")
    print(f"  {'─'*40}")
    for label in ["简单", "中等", "困难", "竞赛"]:
        count = counter.get(label, 0)
        bar   = "█" * (count // 2)
        print(f"  {label:4s}：{count:4d} 条  {bar}")
    print(f"  {'─'*40}\n")


# ── 核心流程 ──────────────────────────────────────────────────────────────────

def encode(data: list[dict]) -> np.ndarray:
    """把每条数据的 input 字段转成语义向量"""
    print(f"[向量化] 加载模型：{MODEL_NAME}（首次运行会自动下载，约 400MB）")
    model = SentenceTransformer(MODEL_NAME)
    texts = [item.get("input", item.get("instruction", "")) for item in data]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    print(f"[向量化] 完成，向量维度：{embeddings.shape[1]}")
    return embeddings


def cluster(embeddings: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """K-means 聚类，返回每条数据的簇编号和簇中心"""
    print(f"[聚类] K-means 开始，K={k} ...")
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels  = km.fit_predict(embeddings)
    centers = km.cluster_centers_
    print(f"[聚类] 完成")
    return labels, centers


def select_representatives(
    data: list[dict],
    embeddings: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    per_k: int,
) -> tuple[list[dict], list[int]]:
    """每个簇选距离中心最近的 per_k 条样本"""
    selected, selected_idx = [], []
    for cid in range(len(centers)):
        idx_in_cluster = np.where(labels == cid)[0]
        if len(idx_in_cluster) == 0:
            continue
        dists    = np.linalg.norm(embeddings[idx_in_cluster] - centers[cid], axis=1)
        n_pick   = min(per_k, len(idx_in_cluster))
        picked   = idx_in_cluster[np.argsort(dists)[:n_pick]]
        selected_idx.extend(picked.tolist())
        selected.extend([data[i] for i in picked])

    print(f"[筛选] {len(data)} 条 → {len(selected)} 条"
          f"（压缩到 {len(selected)/len(data)*100:.1f}%）")
    return selected, selected_idx


# ── 可视化 ────────────────────────────────────────────────────────────────────

def visualize(
    data: list[dict],
    embeddings: np.ndarray,
    labels: np.ndarray,
    selected_idx: list[int],
    save_path: str = "cluster_visualization.png",
):
    print("[可视化] 生成聚类散点图 ...")
    pca    = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(embeddings)
    sel_set = set(selected_idx)

    # 难度颜色映射
    level_colors = {"简单": "#4CAF50", "中等": "#2196F3", "困难": "#FF9800", "竞赛": "#F44336"}
    default_color = "#9E9E9E"

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 左图：按难度着色
    ax = axes[0]
    for item, coord in zip(data, coords):
        level = item.get("output", "").strip()
        color = level_colors.get(level, default_color)
        ax.scatter(coord[0], coord[1], c=color, s=18, alpha=0.6)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=l) for l, c in level_colors.items()],
              fontsize=10, loc="upper right")
    ax.set_title("按难度等级分布", fontsize=13)
    ax.set_xlabel("PCA 第一主成分")
    ax.set_ylabel("PCA 第二主成分")
    ax.grid(True, alpha=0.3)

    # 右图：K-means 筛选结果
    ax = axes[1]
    for i, coord in enumerate(coords):
        if i in sel_set:
            ax.scatter(coord[0], coord[1], c="#E53935", marker="*", s=120, alpha=0.9, zorder=3)
        else:
            ax.scatter(coord[0], coord[1], c="#90CAF9", s=15, alpha=0.4)
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#90CAF9", markersize=8, label="未选中"),
        Line2D([0],[0], marker="*", color="w", markerfacecolor="#E53935", markersize=12, label="K-means 代表样本"),
    ], fontsize=10, loc="upper right")
    ax.set_title("K-means 筛选结果", fontsize=13)
    ax.set_xlabel("PCA 第一主成分")
    ax.set_ylabel("PCA 第二主成分")
    ax.grid(True, alpha=0.3)

    plt.suptitle("题目难度分类数据集 — K-means 聚类分析", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[可视化] 图已保存到 {save_path}")


# ── 主函数 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",   default=INPUT_PATH)
    parser.add_argument("--output",  default=OUTPUT_PATH)
    parser.add_argument("--k",       type=int, default=K)
    parser.add_argument("--per_k",   type=int, default=PER_K)
    parser.add_argument("--no_plot", action="store_true")
    args = parser.parse_args()

    print("=" * 55)
    print("  Step 2：K-means 数据筛选")
    print("=" * 55)

    data = load_data(args.input)
    print_distribution(data, "筛选前")

    embeddings              = encode(data)
    labels, centers         = cluster(embeddings, args.k)
    filtered, selected_idx  = select_representatives(data, embeddings, labels, centers, args.per_k)

    print_distribution(filtered, "筛选后")
    save_data(filtered, args.output)

    if not args.no_plot:
        visualize(data, embeddings, labels, selected_idx)

    print("\n Step 2 完成！")
    print(f"   下一步：把 {args.output} 上传到 Colab，运行 step3_train_colab.ipynb")


if __name__ == "__main__":
    main()
