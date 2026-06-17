import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Paths
WORKSPACE = Path(__file__).resolve().parent
RESULTS_DIR = WORKSPACE / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
PLOTS_DIR = RESULTS_DIR / "plots"

# Create directories
METRICS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "model_a": "Model A (ElasticNet-PAC)",
    "model_b": "Model B (GRU)",
    "model_c": "Model C (Ridge+RF)",
    "model_d": "Model D (HybridRidge+GRU)"
}

def compile_results():
    print("=== Compiling Results ===")
    
    all_summaries = []
    
    for folder, name in MODELS.items():
        summary_path = WORKSPACE / folder / "results" / "summary.csv"
        if not summary_path.exists():
            print(f"Warning: Summary not found for {folder}")
            continue
            
        df = pd.read_csv(summary_path)
        df.insert(0, "model", name)
        all_summaries.append(df)
        
        # Copy and rename actual vs predicted plots
        for scenario in ["80_20", "70_30", "60_40"]:
            src_plot = WORKSPACE / folder / "results" / f"actual_vs_predicted_{scenario}.png"
            if src_plot.exists():
                model_letter = folder.split("_")[1].upper() # e.g. 'a' -> 'A'
                dest_plot = PLOTS_DIR / f"model_{model_letter}_{scenario}.png"
                shutil.copy(src_plot, dest_plot)
                print(f"Copied {src_plot.name} -> {dest_plot.name}")
                
    if not all_summaries:
        print("Error: No model summaries found.")
        return
        
    combined_df = pd.concat(all_summaries, ignore_index=True)
    
    # Save results summary CSV
    csv_out = METRICS_DIR / "results_summary.csv"
    combined_df.to_csv(csv_out, index=False)
    print(f"Saved consolidated metrics to {csv_out}")
    
    # Pivot tables for heatmaps
    rmse_pivot = combined_df.pivot(index="model", columns="scenario", values="RMSE")
    mape_pivot = combined_df.pivot(index="model", columns="scenario", values="MAPE")
    
    # Reorder columns to 80/20, 70/30, 60/40
    cols_order = ["80/20", "70/30", "60/40"]
    rmse_pivot = rmse_pivot[cols_order]
    mape_pivot = mape_pivot[cols_order]
    
    # 1. Heatmap RMSE
    plt.figure(figsize=(10, 6))
    sns.heatmap(rmse_pivot, annot=True, fmt=".2f", cmap="Blues_r", cbar=True, linewidths=0.5)
    plt.title("Model Comparison - Test RMSE (Lower is Better)")
    plt.ylabel("Model")
    plt.xlabel("Scenario Split")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "heatmap_RMSE.png", dpi=150)
    plt.close()
    print("Saved heatmap_RMSE.png")
    
    # 2. Heatmap MAPE
    plt.figure(figsize=(10, 6))
    sns.heatmap(mape_pivot, annot=True, fmt=".2f", cmap="Oranges_r", cbar=True, linewidths=0.5)
    plt.title("Model Comparison - Test MAPE % (Lower is Better)")
    plt.ylabel("Model")
    plt.xlabel("Scenario Split")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "heatmap_MAPE.png", dpi=150)
    plt.close()
    print("Saved heatmap_MAPE.png")
    
    # 3. Results Table Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('off')
    ax.axis('tight')
    
    # Format values for display
    display_df = combined_df.copy()
    display_df["RMSE"] = display_df["RMSE"].round(2)
    display_df["MAPE"] = display_df["MAPE"].round(2).astype(str) + "%"
    display_df["MAE"] = display_df["MAE"].round(2)
    display_df["R2"] = display_df["R2"].round(4)
    display_df["SMAPE"] = display_df["SMAPE"].round(2).astype(str) + "%"
    
    # Rename columns for presentation
    display_df = display_df.rename(columns={
        "model": "Model Name",
        "scenario": "Scenario Split",
        "train_ratio": "Train Ratio"
    })
    
    table = ax.table(cellText=display_df.values, colLabels=display_df.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.2)
    
    # Style table headers
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#1F77B4')
            
    plt.title("Model Scenario Results Summary Table", fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "results_table.png", dpi=150)
    plt.close()
    print("Saved results_table.png")
    
    print("\n=== Markdown Summary of Results ===")
    print(combined_df.to_markdown(index=False))

if __name__ == "__main__":
    compile_results()
