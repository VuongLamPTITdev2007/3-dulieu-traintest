import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = r"D:\Combined_Datasets"
RESULTS_FILE = os.path.join(BASE_DIR, "evaluation_results_all.csv")
CHARTS_DIR = os.path.join(BASE_DIR, "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

def visualize_results():
    if not os.path.exists(RESULTS_FILE):
        print(f"Error: Could not find {RESULTS_FILE}")
        return

    df = pd.read_csv(RESULTS_FILE)
    
    # Tạo biến nhóm Model Category + Algorithm cho nhãn rõ ràng hơn
    df["Model_Alg"] = df["Category"] + " - " + df["Algorithm"]

    sns.set_theme(style="whitegrid")
    
    datasets = df["Dataset"].unique()
    
    for ds in datasets:
        df_ds = df[df["Dataset"] == ds].copy()
        df_ds = df_ds.sort_values(by="Category")
        
        # 1. Vẽ Accuracy và F1-Score
        plt.figure(figsize=(14, 6))
        
        # Melt data cho dễ vẽ grouped bar chart
        df_melt = df_ds.melt(id_vars=["Model_Alg"], value_vars=["Accuracy", "F1", "Precision", "Recall"], 
                             var_name="Metric", value_name="Score")
        
        ax = sns.barplot(data=df_melt, x="Model_Alg", y="Score", hue="Metric", palette="viridis")
        plt.title(f"[{ds}] Comparison of AI Metrics (Accuracy, Precision, Recall, F1)", fontsize=14, fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.ylim(0, 1.1)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(CHARTS_DIR, f"{ds}_AI_Metrics.png"), dpi=150)
        plt.close()

        # 2. Vẽ System Metrics (CPU và RAM)
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        ax1.bar(df_ds["Model_Alg"], df_ds["CPU_pct"], color='salmon', alpha=0.7, label='CPU Usage (%)')
        ax1.set_xlabel("Algorithms", fontsize=12)
        ax1.set_ylabel("CPU Usage (%)", color='red', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='red')
        plt.xticks(rotation=45, ha="right")
        
        ax2 = ax1.twinx()
        ax2.plot(df_ds["Model_Alg"], df_ds["RAM_MB"], color='blue', marker='o', linewidth=2, label='Peak RAM (MB)')
        ax2.set_ylabel("RAM Usage (MB)", color='blue', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='blue')
        
        plt.title(f"[{ds}] System Profiling: CPU & RAM Usage on Edge", fontsize=14, fontweight="bold")
        fig.tight_layout()
        plt.savefig(os.path.join(CHARTS_DIR, f"{ds}_System_Metrics.png"), dpi=150)
        plt.close()
        
        # 3. Vẽ Latency (Thời gian suy luận)
        plt.figure(figsize=(12, 5))
        sns.barplot(data=df_ds, x="Model_Alg", y="Latency_ms", palette="magma")
        plt.title(f"[{ds}] Inference Latency (ms / sample)", fontsize=14, fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Latency (ms)")
        
        # Thêm value label
        for i, val in enumerate(df_ds["Latency_ms"]):
            plt.text(i, val + 0.1, f"{val:.2f}", ha="center", fontsize=9)
            
        plt.tight_layout()
        plt.savefig(os.path.join(CHARTS_DIR, f"{ds}_Latency.png"), dpi=150)
        plt.close()
        
    print("Visualizations successfully generated in 'charts' directory.")

if __name__ == "__main__":
    visualize_results()
