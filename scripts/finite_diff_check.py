import numpy as np
import sys
import os
import random

sys.path.append(os.path.join(os.path.dirname(__file__), "../COMP36212_EX3_Code"))
from neural_network import *  # expects init, forward, backward, and weight arrays
from mnist_helper import training_data, training_labels

EPSILON = 1e-5
TOL = 1e-3
SAMPLE_INDEX = 0  # fixed sample

def finite_difference(w, i, eps):
    original = w[i]
    w[i] = original + eps
    L_plus = compute_loss(SAMPLE_INDEX)
    w[i] = original - eps
    L_minus = compute_loss(SAMPLE_INDEX)
    w[i] = original  # restore
    return (L_plus - L_minus) / (2 * eps)

def compute_loss(sample):
    evaluate_forward_pass(training_data, sample)
    return compute_xent_loss(training_labels[sample])

def main():
    evaluate_forward_pass(training_data, SAMPLE_INDEX)
    evaluate_backward_pass_sparse(training_labels[SAMPLE_INDEX], SAMPLE_INDEX)
    store_gradient_contributions()

    tested = 0
    failed = 0
    results = []

    layers = [
        (w_LI_L1, "LI_L1"),
        (w_L1_L2, "L1_L2"),
        (w_L2_L3, "L2_L3"),
        (w_L3_LO, "L3_LO")
    ]

    for w_mat, label in layers:
        flat_w = w_mat.ravel()
        indices = random.sample(range(flat_w.size), min(3, flat_w.size))
        for idx in indices:
            numeric = finite_difference(flat_w, idx, EPSILON)
            analytic = flat_w[idx].dw
            rel_error = abs(numeric - analytic) / max(1, abs(numeric))
            ok = rel_error < TOL
            results.append((label, idx, numeric, analytic, rel_error, ok))
            tested += 1
            if not ok:
                failed += 1

    print("LAYER,INDEX,NUMERIC,ANALYTIC,REL_ERROR,PASS")
    for r in results:
        print(",".join(map(str, r)))

    sys.exit(1 if failed > 0 else 0)

if __name__ == "__main__":
    main()
