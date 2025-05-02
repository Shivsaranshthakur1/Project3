#include "optimiser.h"

#include <assert.h>
#include <math.h>
#include <stdlib.h>

#include "mnist_helper.h"
#include "neural_network.h"

unsigned int log_freq = 30000;
unsigned int num_batches;
unsigned int batch_size;
unsigned int total_epochs;
double lr0, lrN, momentum, beta1, beta2, epsilon;
int opt_flag;

static inline double lr_epoch(unsigned k, unsigned N) {
    return lr0 * (1.0 - (double)k / N) + lrN * ((double)k / N);
}

void print_training_stats(unsigned int epoch_counter, unsigned int total_iter, double mean_loss,
                          double test_accuracy) {
    printf("Epoch: %u,  Total iter: %u,  Mean Loss: %0.12f,  Test Acc: %f\n", epoch_counter,
           total_iter, mean_loss, test_accuracy);
}

void initialise_optimiser(double lr0_in, double lrN_in, unsigned bs, unsigned epochs,
                          double beta1_or_mom, double beta2_in, double eps_in, int flag) {
    lr0 = lr0_in;
    lrN = lrN_in;
    batch_size = bs;
    total_epochs = epochs;
    momentum = beta1_or_mom;
    beta1 = beta1_or_mom;
    beta2 = beta2_in;
    epsilon = eps_in;
    opt_flag = flag;
    num_batches = epochs * (N_TRAINING_SET / bs);

    printf("Optimising with parameters:\n");
    printf("\tlr0 = %.6f, lrN = %.6f\n", lr0, lrN);
    printf("\tbatch_size = %u, epochs = %u\n", batch_size, total_epochs);
    printf("\tbeta1/momentum = %.6f, beta2 = %.6f, eps = %.6f\n", beta1, beta2, epsilon);
    printf("\topt_flag = %d\n\n", opt_flag);
}

void run_optimisation(void) {
    unsigned int training_sample = 0;
    unsigned int total_iter = 0;
    unsigned int epoch_counter = 0;
    double test_accuracy = 0.0;
    double mean_loss = 0.0;

    // 📍 Step 3: address print (optimiser side)
    printf("Address in optimiser.c: %p\n", (void*)&w_LI_L1[0][0].w);

    for (int i = 0; i < num_batches; i++) {
        for (int j = 0; j < batch_size; j++) {
            if (total_iter % log_freq == 0 || total_iter == 0) {
                if (total_iter > 0) {
                    mean_loss /= log_freq;
                }
                test_accuracy = evaluate_testing_accuracy();
                printf("EPOCH_LOG,%u,%f,%f\n", epoch_counter, mean_loss, test_accuracy);
                print_training_stats(epoch_counter, total_iter, mean_loss, test_accuracy);
                mean_loss = 0.0;
            }

            double loss = evaluate_objective_function(training_sample);
            mean_loss += loss;

            ++total_iter;
            ++training_sample;
            if (training_sample == N_TRAINING_SET) {
                training_sample = 0;
                ++epoch_counter;
            }
        }

        double lr_k = lr_epoch(epoch_counter, total_epochs);

        // ✅ Step 1: probe before update
        if (total_iter == batch_size - 1) {
            printf("before: w=%e dw=%e v=%e\n", w_LI_L1[0][0].w, w_LI_L1[0][0].dw, w_LI_L1[0][0].v);
        }

        switch (opt_flag) {
            case 0:
                update_parameters(batch_size);
                break;
            case 1:
                update_parameters_momentum(batch_size, lr_k, momentum);
                // ✅ Step 2: probe after momentum update
                if (total_iter == batch_size) {
                    printf("after : w=%e dw=%e v=%e\n", w_LI_L1[0][0].w, w_LI_L1[0][0].dw,
                           w_LI_L1[0][0].v);
                }
                break;
            case 2:
                update_parameters_adam(batch_size, lr_k, beta1, beta2, epsilon, total_iter);
                break;
            default:
                fprintf(stderr, "Invalid opt_flag: %d\n", opt_flag);
                exit(1);
        }

        assert(!isnan(w_LI_L1[0][0].w));
    }

    test_accuracy = evaluate_testing_accuracy();
    printf("EPOCH_LOG,%u,%f,%f\n", total_epochs, mean_loss, test_accuracy);
    print_training_stats(total_epochs, total_iter, mean_loss / log_freq, test_accuracy);
}

double evaluate_objective_function(unsigned int sample) {
    evaluate_forward_pass(training_data, sample);
    double loss = compute_xent_loss(training_labels[sample]);
    evaluate_backward_pass_sparse(training_labels[sample], sample);
    store_gradient_contributions();
    return loss;
}

// SGD
static inline void apply_and_zero(weight_t* W, unsigned r, unsigned c, double scale) {
    for (unsigned i = 0; i < r * c; ++i) {
        W[i].w -= scale * W[i].dw;
        W[i].dw = 0.0;
    }
}

void update_parameters(unsigned batch_sz) {
    const double scale = lr0 / (double)batch_sz;
    apply_and_zero(&w_L3_LO[0][0], N_NEURONS_L3, N_NEURONS_LO, scale);
    apply_and_zero(&w_L2_L3[0][0], N_NEURONS_L2, N_NEURONS_L3, scale);
    apply_and_zero(&w_L1_L2[0][0], N_NEURONS_L1, N_NEURONS_L2, scale);
    apply_and_zero(&w_LI_L1[0][0], N_NEURONS_LI, N_NEURONS_L1, scale);
}
// Momentum
static inline void momentum_layer(weight_t* W, unsigned r, unsigned c, double lr, double beta,
                                  unsigned batch) {
    double scale = lr / (double)batch;
    for (unsigned i = 0; i < r * c; ++i) {
        W[i].v = beta * W[i].v - scale * W[i].dw;
        W[i].w += W[i].v;
        W[i].dw = 0.0;
    }
}

void update_parameters_momentum(unsigned batch, double lr, double beta) {
    momentum_layer(&w_L3_LO[0][0], N_NEURONS_L3, N_NEURONS_LO, lr, beta, batch);
    momentum_layer(&w_L2_L3[0][0], N_NEURONS_L2, N_NEURONS_L3, lr, beta, batch);
    momentum_layer(&w_L1_L2[0][0], N_NEURONS_L1, N_NEURONS_L2, lr, beta, batch);
    momentum_layer(&w_LI_L1[0][0], N_NEURONS_LI, N_NEURONS_L1, lr, beta, batch);
}

// Adam
static inline void adam_layer(weight_t* W, unsigned r, unsigned c, double lr, double b1, double b2,
                              double eps, unsigned batch, unsigned t) {
    const double scale = lr / (double)batch;
    const double b1t = 1.0 - pow(b1, t);
    const double b2t = 1.0 - pow(b2, t);
    for (unsigned i = 0; i < r * c; ++i) {
        W[i].v_m = b1 * W[i].v_m + (1.0 - b1) * W[i].dw;
        W[i].v_v = b2 * W[i].v_v + (1.0 - b2) * (W[i].dw * W[i].dw);
        double m_hat = W[i].v_m / b1t;
        double v_hat = W[i].v_v / b2t;
        W[i].w -= scale * m_hat / (sqrt(v_hat) + eps);
        W[i].dw = 0.0;
    }
}

void update_parameters_adam(unsigned batch, double lr, double b1, double b2, double eps,
                            unsigned t) {
    adam_layer(&w_L3_LO[0][0], N_NEURONS_L3, N_NEURONS_LO, lr, b1, b2, eps, batch, t);
    adam_layer(&w_L2_L3[0][0], N_NEURONS_L2, N_NEURONS_L3, lr, b1, b2, eps, batch, t);
    adam_layer(&w_L1_L2[0][0], N_NEURONS_L1, N_NEURONS_L2, lr, b1, b2, eps, batch, t);
    adam_layer(&w_LI_L1[0][0], N_NEURONS_LI, N_NEURONS_L1, lr, b1, b2, eps, batch, t);
}