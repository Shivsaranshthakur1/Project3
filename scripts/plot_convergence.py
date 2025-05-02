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
    log_files = sorted([f for f in os.listdir("logs") if f.startswith("mom_") and f.endswith(".csv")])

    plt.figure()
    for log_file in log_files:
        fullpath = os.path.join("logs", log_file)
        epochs, accs = parse_log(fullpath)
        label = log_file.replace("mom_", "").replace(".csv", "")
        plt.plot(epochs, accs, marker='o', label=label)

    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.title("Momentum Accuracy Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig("figures/compare_partII.png")

if __name__ == "__main__":
    main()
