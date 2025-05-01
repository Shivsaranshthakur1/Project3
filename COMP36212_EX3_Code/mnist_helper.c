// FORCE LINE TO TRIGGER COMMIT
#include "mnist_helper.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TRAINING_DATA_BYTES 47040016
#define TRAINING_LABELS_BYTES 60008
#define TESTING_DATA_BYTES 7840016
#define TESTING_LABELS_BYTES 10008
#define PATH_SIZE_MAX 200

uint8_t** training_data;
uint8_t training_labels[N_TRAINING_SET];
uint8_t** testing_data;
uint8_t testing_labels[N_TESTING_SET];

uint8_t training_data_buffer[TRAINING_DATA_BYTES];
uint8_t training_labels_buffer[TRAINING_LABELS_BYTES];
uint8_t testing_data_buffer[TESTING_DATA_BYTES];
uint8_t testing_labels_buffer[TESTING_LABELS_BYTES];

void initialise_dataset(const char* path_to_dataset, unsigned int print_samples) {
    training_data = (uint8_t**)malloc(N_TRAINING_SET * sizeof(uint8_t*));
    for (int i = 0; i < N_TRAINING_SET; i++) {
        training_data[i] = (uint8_t*)malloc(PIXEL_DIM_FLAT * sizeof(uint8_t));
    }
    testing_data = (uint8_t**)malloc(N_TESTING_SET * sizeof(uint8_t*));
    for (int i = 0; i < N_TESTING_SET; i++) {
        testing_data[i] = (uint8_t*)malloc(PIXEL_DIM_FLAT * sizeof(uint8_t));
    }

    char* training_set_loc = malloc(PATH_SIZE_MAX);
    char* training_labels_loc = malloc(PATH_SIZE_MAX);
    char* testing_set_loc = malloc(PATH_SIZE_MAX);
    char* testing_labels_loc = malloc(PATH_SIZE_MAX);

    strcat(strcpy(training_set_loc, path_to_dataset), "/train-images-idx3-ubyte");
    strcat(strcpy(training_labels_loc, path_to_dataset), "/train-labels-idx1-ubyte");
    strcat(strcpy(testing_set_loc, path_to_dataset), "/t10k-images-idx3-ubyte");
    strcat(strcpy(testing_labels_loc, path_to_dataset), "/t10k-labels-idx1-ubyte");

    load_mnist_training_set(training_set_loc);
    load_mnist_training_labels(training_labels_loc);
    load_mnist_testing_set(testing_set_loc);
    load_mnist_testing_labels(testing_labels_loc);

    free(training_set_loc);
    free(training_labels_loc);
    free(testing_set_loc);
    free(testing_labels_loc);

    if (print_samples) {
        for (int i = 0; i < 3; i++) {
            printf("label: %u \n", training_labels[i]);
            print_single_example(training_data, i);
        }
        for (int i = 0; i < 3; i++) {
            printf("label: %u \n", testing_labels[i]);
            print_single_example(testing_data, i);
        }
    }
}

void load_mnist_training_set(char* path) {
    printf("Loading training set...\n");
    FILE* in_file = fopen(path, "rb");
    if (!in_file) {
        fprintf(stderr, "Failed to open training set\n");
        exit(1);
    }
    if (fread(training_data_buffer, sizeof(training_data_buffer[0]), TRAINING_DATA_BYTES,
              in_file) != TRAINING_DATA_BYTES) {
        fprintf(stderr, "fread failed in load_mnist_training_set\n");
        exit(1);
    }
    fclose(in_file);

    for (int i = 0; i < N_TRAINING_SET; i++) {
        for (int j = 0; j < PIXEL_DIM_FLAT; j++) {
            training_data[i][j] = training_data_buffer[i * PIXEL_DIM_FLAT + j + 16];
        }
    }
    printf("Training set loaded successfully...\n");
}

void load_mnist_training_labels(char* path) {
    printf("\nLoading training set labels...\n");
    FILE* in_file = fopen(path, "rb");
    if (!in_file) {
        fprintf(stderr, "Failed to open training labels\n");
        exit(1);
    }
    if (fread(training_labels_buffer, sizeof(training_labels_buffer[0]), TRAINING_LABELS_BYTES,
              in_file) != TRAINING_LABELS_BYTES) {
        fprintf(stderr, "fread failed in load_mnist_training_labels\n");
        exit(1);
    }
    fclose(in_file);

    for (int i = 0; i < N_TRAINING_SET; i++) {
        training_labels[i] = training_labels_buffer[i + 8];
    }
    printf("Training set labels loaded successfully...\n");
}

void load_mnist_testing_set(char* path) {
    printf("\nLoading testing set...\n");
    FILE* in_file = fopen(path, "rb");
    if (!in_file) {
        fprintf(stderr, "Failed to open testing set\n");
        exit(1);
    }
    if (fread(testing_data_buffer, sizeof(testing_data_buffer[0]), TESTING_DATA_BYTES, in_file) !=
        TESTING_DATA_BYTES) {
        fprintf(stderr, "fread failed in load_mnist_testing_set\n");
        exit(1);
    }
    fclose(in_file);

    for (int i = 0; i < N_TESTING_SET; i++) {
        for (int j = 0; j < PIXEL_DIM_FLAT; j++) {
            testing_data[i][j] = testing_data_buffer[i * PIXEL_DIM_FLAT + j + 16];
        }
    }
    printf("Testing set loaded successfully...\n");
}

void load_mnist_testing_labels(char* path) {
    printf("\nLoading testing set labels...\n");
    FILE* in_file = fopen(path, "rb");
    if (!in_file) {
        fprintf(stderr, "Failed to open testing labels\n");
        exit(1);
    }
    if (fread(testing_labels_buffer, sizeof(testing_labels_buffer[0]), TESTING_LABELS_BYTES,
              in_file) != TESTING_LABELS_BYTES) {
        fprintf(stderr, "fread failed in load_mnist_testing_labels\n");
        exit(1);
    }
    fclose(in_file);

    for (int i = 0; i < N_TESTING_SET; i++) {
        testing_labels[i] = testing_labels_buffer[i + 8];
    }
    printf("Testing set labels loaded successfully...\n");
}

void free_dataset_data_structures(void) {
    for (int i = 0; i < N_TRAINING_SET; i++) {
        free(training_data[i]);
    }
    for (int i = 0; i < N_TESTING_SET; i++) {
        free(testing_data[i]);
    }
    free(training_data);
    free(testing_data);
}

void print_single_example(uint8_t** dataset, int n) {
    for (int i = 0; i < PIXEL_DIM; i++) {
        for (int j = 0; j < PIXEL_DIM; j++) {
            printf("%3i ", dataset[n][i * PIXEL_DIM + j]);
        }
        printf("\n");
    }
    printf("\n");
}
// force diff
// FORCE PUSH TRIGGER
// FINAL PUSH TRIGGER
