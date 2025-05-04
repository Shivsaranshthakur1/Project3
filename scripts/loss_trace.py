import subprocess
import re

losses = []

for i in range(100):
    cmd = [
        "./mnist_optimiser.out",
        "mnist_data",
        "0.1", "0.1",  # lr0 = lrN
        "1",           # batch size
        "1",           # epochs
        "1",           # opt_flag (1 = momentum)
        "0.0",         # momentum = 0.0 (pure SGD)
        "0.0", "1e-8"  # dummy b2, eps (ignored for momentum)
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    match = re.search(r"Mean Loss:\s*([\d\.eE+-]+)", result.stdout)
    if match:
        losses.append(float(match.group(1)))
    else:
        print("Failed to parse loss on run", i)
        exit(1)

for i in range(len(losses) - 1):
    assert losses[i] > losses[i + 1], f"Loss did not decrease: {losses[i]} → {losses[i+1]}"

print("Validation passed: loss consistently decreases.")
