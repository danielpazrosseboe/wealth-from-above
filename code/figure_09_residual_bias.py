import matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import analysis_results as A, figure_common as F

def figure_residual_bias(d):
    _, quint = A.bias_analysis(d)
    if not quint: print("[SKIP] no data"); return None
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name, by_q in quint.items(): ax.plot(range(len(by_q)), by_q.values, marker="o", label=name)
    ax.axhline(0, color="k", lw=0.8); ax.set_xticks(range(5)); ax.set_xticklabels([f"Q{i+1}" for i in range(5)])
    ax.set_xlabel("True wealth quintile (Q1 = poorest)"); ax.set_ylabel("Mean residual (pred - true)")
    ax.set_title("Prediction bias across the wealth distribution"); ax.legend(); fig.tight_layout()
    return F.save(fig, "figure_residual_by_quintile.png")

if __name__ == "__main__":
    figure_residual_bias(A.load_predictions())
