/* main.c – entry-point & CLI presets + built-in finite-difference grad-check */

#include <math.h> /* fabs, fmax – grad-check helpers */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mnist_helper.h"
#include "neural_network.h"
#include "optimiser.h"

/* ------------ usage helper ------------ */
static void print_usage_and_exit(char** argv) {
    printf("Usage:\n");
    printf(
        "  %s <dataset> <lr0> <lrN> <batch> <epochs> "
        "<opt_flag> <momentum/b1> <b2> <eps>\n",
        argv[0]);
    printf("    opt_flag: 0 = SGD, 1 = Momentum, 2 = Adam\n");
    printf("  %s --safe-sgd\n", argv[0]);
    printf("  %s --safe-momentum\n", argv[0]);
    printf("  %s --safe-adam\n", argv[0]);
    printf("  %s --gradcheck         # finite-difference test\n", argv[0]);
    exit(1);
}

/* ============== main ============== */
int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IOLBF, 0); /* line-buffered logs */

    const char* dataset = NULL;
    double lr0 = 0, lrN = 0, b1 = 0, b2 = 0, eps = 0;
    unsigned batch = 0, epochs = 0;
    int flag = 0;

    /* ---------- finite-difference checker ---------- */
    if (argc == 2 && strcmp(argv[1], "--gradcheck") == 0) {
        initialise_dataset("mnist_data", 0);
        initialise_nn();
        initialise_optimiser(0.1, 0.1, 1, 1, 0, 0, 1e-8, 0);

        /* one forward/backward to populate .dw */
        evaluate_forward_pass(training_data, 0);
        evaluate_backward_pass_sparse(training_labels[0], 0);
        store_gradient_contributions();

        srand(0); /* deterministic picks */
        const double eps_fd = 1e-5;
        const double tol = 5e-1; /* generous – covers ReLU discontinuities */
        int pick[3];

#define CK(mat, R, C, tag)                                          \
    do {                                                            \
        size_t SZ = (size_t)(R) * (C);                              \
        for (int k = 0; k < 3; ++k) pick[k] = rand() % SZ;          \
        for (int k = 0; k < 3; ++k) {                               \
            size_t idx = pick[k], r = idx / (C), c = idx % (C);     \
            weight_t* w = &mat[r][c];                               \
            double ana = w->dw;                                     \
            w->dw = 0.0;                                            \
            double orig = w->w;                                     \
            w->w = orig + eps_fd;                                   \
            double Lp = evaluate_objective_function(0);             \
            w->w = orig - eps_fd;                                   \
            double Lm = evaluate_objective_function(0);             \
            w->w = orig;                                            \
            double num = (Lp - Lm) / (2 * eps_fd);                  \
            double rel = fabs(num - ana) / fmax(1.0, fabs(num));    \
            /* treat near-zero pairs as OK */                       \
            if (fabs(num) < 1e-6 && fabs(ana) < 1e-6) rel = 0.0;    \
            printf("%s,%zu,%e,%e,%e,%s\n", tag, idx, num, ana, rel, \
                   (rel < tol ? "PASS" : "FAIL"));                  \
        }                                                           \
    } while (0)

        CK(w_LI_L1, N_NEURONS_LI, N_NEURONS_L1, "LI_L1");
        CK(w_L1_L2, N_NEURONS_L1, N_NEURONS_L2, "L1_L2");
        CK(w_L2_L3, N_NEURONS_L2, N_NEURONS_L3, "L2_L3");
        CK(w_L3_LO, N_NEURONS_L3, N_NEURONS_LO, "L3_LO");
        return 0;
    }
    /* ------------------------------------------------ */

    /* ---------- convenience presets ---------- */
    if (argc == 2) {
        if (strcmp(argv[1], "--safe-sgd") == 0) {
            dataset = "mnist_data";
            lr0 = lrN = 0.1;
            batch = 10;
            epochs = 10;
            flag = 0;
            b1 = b2 = eps = 0.0;
            printf("▶ Running preset: SGD baseline\n");

        } else if (strcmp(argv[1], "--safe-momentum") == 0) {
            dataset = "mnist_data";
            lr0 = 0.01;
            lrN = 0.0001;
            batch = 10;
            epochs = 10;
            flag = 1;
            b1 = 0.9;
            b2 = eps = 0.0;
            printf("▶ Running preset: Momentum\n");

        } else if (strcmp(argv[1], "--safe-adam") == 0) {
            dataset = "mnist_data";
            lr0 = 0.001;
            lrN = 0.0001;
            batch = 10;
            epochs = 10;
            flag = 2;
            b1 = 0.9;
            b2 = 0.999;
            eps = 1e-8;
            printf("▶ Running preset: Adam\n");

        } else {
            print_usage_and_exit(argv);
        }

    } else if (argc == 10) { /* full CLI */
        dataset = argv[1];
        lr0 = atof(argv[2]);
        lrN = atof(argv[3]);
        batch = (unsigned)atoi(argv[4]);
        epochs = (unsigned)atoi(argv[5]);
        flag = atoi(argv[6]);
        b1 = atof(argv[7]);
        b2 = atof(argv[8]);
        eps = atof(argv[9]);

    } else {
        print_usage_and_exit(argv);
    }
    /* ----------------------------------------- */

    if (flag < 0 || flag > 2) {
        fprintf(stderr, "ERROR: invalid opt_flag %d (must be 0,1,2)\n", flag);
        return 1;
    }

    initialise_dataset(dataset, 0);
    initialise_nn();
    initialise_optimiser(lr0, lrN, batch, epochs, b1, b2, eps, flag);
    run_optimisation();
    free_dataset_data_structures();
    return 0;
}