import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load all mom_*.csv logs
log_dir = "logs"
logs = [f for f in os.listdir(log_dir) if f.startswith("mom_") and f.endswith(".csv")]

results = []

for file in logs:
    parts = file.replace("mom_", "").replace(".csv", "").split("_")
    batch = int(parts[0])
    lr = float(parts[1])

    with open(os.path.join(log_dir, file)) as f:
        lines = f.readlines()
        for line in reversed(lines):
            if line.startswith("EPOCH_LOG"):
                parts = line.strip().split(",")
                acc = float(parts[-1])
                break
        else:
            acc = 0.0

    results.append({"batch": batch, "lr": lr, "accuracy": acc})

results_df = pd.DataFrame(results)
pivot = results_df.pivot(index="batch", columns="lr", values="accuracy")

plt.figure(figsize=(8, 6))
sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis")
plt.title("Final Test Accuracy: Momentum Grid Search")
plt.xlabel("Learning Rate")
plt.ylabel("Batch Size")
plt.tight_layout()

os.makedirs("figures", exist_ok=True)
plt.savefig("figures/compare_partII.png")
print("[✓] Saved: figures/compare_partII.png")
