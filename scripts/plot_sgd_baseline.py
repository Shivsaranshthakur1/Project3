import os
import matplotlib.pyplot as plt

def parse_log(filepath):
    epochs = []
    accs = []
    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("EPOCH_LOG"):
                parts = line.strip().split(",")
                epoch = int(parts[1])
                acc = float(parts[3])
                epochs.append(epoch)
                accs.append(acc)
    return epochs, accs

def main():
    os.makedirs("figures", exist_ok=True)
    log_file = "logs/run_sgd_fixed.csv"

    if not os.path.exists(log_file):
        raise FileNotFoundError(f"Expected log file not found: {log_file}")

    epochs, accs = parse_log(log_file)

    plt.figure()
    plt.plot(epochs, accs, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.title("SGD Baseline Accuracy per Epoch")
    plt.grid(True)
    plt.savefig("figures/sgd_baseline.png")

if __name__ == "__main__":
    main()
