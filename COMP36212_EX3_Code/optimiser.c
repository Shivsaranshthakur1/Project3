#include "optimiser.h"

#include "math.h"
#include "mnist_helper.h"
#include "neural_network.h"

// Function declarations
void update_parameters(unsigned int batch_size);
void print_training_stats(unsigned int epoch_counter, unsigned int total_iter, double mean_loss,
                          double test_accuracy);

// Optimisation parameters
unsigned int log_freq = 30000;  // Compute and print accuracy every log_freq iterations

// Parameters passed from command line arguments
unsigned int num_batches;
unsigned int batch_size;
unsigned int total_epochs;
double learning_rate;

void print_training_stats(unsigned int epoch_counter, unsigned int total_iter, double mean_loss,
                          double test_accuracy) {
    printf("Epoch: %u,  Total iter: %u,  Mean Loss: %0.12f,  Test Acc: %f\n", epoch_counter,
           total_iter, mean_loss, test_accuracy);
}

void initialise_optimiser(double cmd_line_learning_rate, int cmd_line_batch_size,
                          int cmd_line_total_epochs) {
    batch_size = cmd_line_batch_size;
    learning_rate = cmd_line_learning_rate;
    total_epochs = cmd_line_total_epochs;

    num_batches = total_epochs * (N_TRAINING_SET / batch_size);
    printf(
        "Optimising with parameters: \n\tepochs = %u \n\tbatch_size = %u \n\tnum_batches = "
        "%u\n\tlearning_rate = %f\n\n",
        total_epochs, batch_size, num_batches, learning_rate);
}

void run_optimisation(void) {
    unsigned int training_sample = 0;
    unsigned int total_iter = 0;
    double obj_func = 0.0;
    unsigned int epoch_counter = 0;
    double test_accuracy = 0.0;
    double mean_loss = 0.0;

    for (int i = 0; i < num_batches; i++) {
        for (int j = 0; j < batch_size; j++) {
            if (total_iter % log_freq == 0 || total_iter == 0) {
                if (total_iter > 0) {
                    mean_loss = mean_loss / ((double)log_freq);
                }
                test_accuracy = evaluate_testing_accuracy();
                print_training_stats(epoch_counter, total_iter, mean_loss, test_accuracy);
                mean_loss = 0.0;
            }

            obj_func = evaluate_objective_function(training_sample);
            mean_loss += obj_func;

            total_iter++;
            training_sample++;
            if (training_sample == N_TRAINING_SET) {
                training_sample = 0;
                epoch_counter++;
            }
        }

        update_parameters(batch_size);
    }

    test_accuracy = evaluate_testing_accuracy();
    print_training_stats(epoch_counter, total_iter, (mean_loss / ((double)log_freq)),
                         test_accuracy);
}

double evaluate_objective_function(unsigned int sample) {
    evaluate_forward_pass(training_data, sample);
    double loss = compute_xent_loss(training_labels[sample]);
    evaluate_backward_pass_sparse(training_labels[sample], sample);
    store_gradient_contributions();
    return loss;
}

// SGD batch update implementation
static inline void apply_and_zero(weight_struct_t* w, unsigned int n_rows, unsigned int n_cols,
                                  double scale) {
    for (unsigned int i = 0; i < n_rows; ++i) {
        for (unsigned int j = 0; j < n_cols; ++j) {
            weight_struct_t* p = &w[i * n_cols + j];
            p->w -= scale * p->dw;
            p->dw = 0.0;
        }
    }
}

void update_parameters(unsigned int batch_sz) {
    const double scale = learning_rate / (double)batch_sz;
    apply_and_zero(&w_L3_LO[0][0], N_NEURONS_L3, N_NEURONS_LO, scale);
    apply_and_zero(&w_L2_L3[0][0], N_NEURONS_L2, N_NEURONS_L3, scale);
    apply_and_zero(&w_L1_L2[0][0], N_NEURONS_L1, N_NEURONS_L2, scale);
    apply_and_zero(&w_LI_L1[0][0], N_NEURONS_LI, N_NEURONS_L1, scale);
}
