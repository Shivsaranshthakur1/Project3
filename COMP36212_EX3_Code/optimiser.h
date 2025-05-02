#ifndef OPTIMISER_H
#define OPTIMISER_H

#include <stdio.h>

#ifndef WEIGHT_T_DECLARED
#define WEIGHT_T_DECLARED
typedef struct {
    double w;
    double dw;
    double v;
    double v_m;
    double v_v;
} weight_t;
#endif

void initialise_optimiser(double lr0, double lrN, unsigned bs, unsigned epochs,
                          double beta1_or_momentum, double beta2, double eps, int opt_flag);

void run_optimisation(void);
double evaluate_objective_function(unsigned int sample);

void update_parameters(unsigned int batch_sz);  // SGD
void update_parameters_momentum(unsigned int batch, double lr, double beta);
void update_parameters_adam(unsigned int batch, double lr, double b1, double b2, double eps,
                            unsigned t);

#endif /* OPTIMISER_H */