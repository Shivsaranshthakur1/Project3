#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mnist_helper.h"
#include "neural_network.h"
#include "optimiser.h"

void print_help_and_exit(char** argv) {
    printf(
        "usage: %s <path_to_dataset> <lr0> <lrN> <batch_size> <total_epochs> <opt_flag> "
        "<momentum|beta1> <beta2> <eps>\n",
        argv[0]);
    exit(0);
}

int main(int argc, char** argv) {
    if (argc != 10) {
        printf(
            "usage: %s <dataset> <lr0> <lrN> <batch> <epochs> <opt_flag> <momentum/b1> <b2> "
            "<eps>\n",
            argv[0]);
        exit(1);
    }

    const char* dataset = argv[1];
    double lr0 = atof(argv[2]);
    double lrN = atof(argv[3]);
    unsigned batch = atoi(argv[4]);
    unsigned epochs = atoi(argv[5]);
    int flag = atoi(argv[6]);
    double b1_mom = atof(argv[7]);
    double b2 = atof(argv[8]);
    double eps = atof(argv[9]);

    initialise_dataset(dataset, 0);
    initialise_nn();
    initialise_optimiser(lr0, lrN, batch, epochs, b1_mom, b2, eps, flag);
    run_optimisation();
    free_dataset_data_structures();
}