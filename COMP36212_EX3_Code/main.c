#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mnist_helper.h"
#include "neural_network.h"
#include "optimiser.h"

void print_usage_and_exit(char** argv) {
    printf("Usage:\n");
    printf("  %s <dataset> <lr0> <lrN> <batch> <epochs> <opt_flag> <momentum/b1> <b2> <eps>\n",
           argv[0]);
    printf("    opt_flag: 0 = SGD, 1 = Momentum, 2 = Adam\n");
    printf("  %s --safe-sgd\n", argv[0]);
    printf("  %s --safe-momentum\n", argv[0]);
    printf("  %s --safe-adam\n", argv[0]);
    exit(1);
}

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IOLBF, 0);  // line-buffered output for logs

    const char* dataset = NULL;
    double lr0 = 0.0, lrN = 0.0, b1 = 0.0, b2 = 0.0, eps = 0.0;
    unsigned batch = 0, epochs = 0;
    int flag = 0;

    if (argc == 2) {
        if (strcmp(argv[1], "--safe-sgd") == 0) {
            dataset = "mnist_data";
            lr0 = 0.1;
            lrN = 0.1;
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
    } else if (argc == 10) {
        dataset = argv[1];
        lr0 = atof(argv[2]);
        lrN = atof(argv[3]);
        batch = atoi(argv[4]);
        epochs = atoi(argv[5]);
        flag = atoi(argv[6]);
        b1 = atof(argv[7]);
        b2 = atof(argv[8]);
        eps = atof(argv[9]);
    } else {
        print_usage_and_exit(argv);
    }

    if (flag < 0 || flag > 2) {
        fprintf(stderr, "ERROR: Invalid optimiser flag %d. Must be 0, 1, or 2.\n", flag);
        print_usage_and_exit(argv);
    }

    initialise_dataset(dataset, 0);
    initialise_nn();
    initialise_optimiser(lr0, lrN, batch, epochs, b1, b2, eps, flag);
    run_optimisation();
    free_dataset_data_structures();
    return 0;
}