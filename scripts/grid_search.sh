#!/bin/bash
mkdir -p logs
for bs in 10 100; do
  for lr in 0.01 0.001; do
    echo "Running: batch=1, lr=0.1"
    ./COMP36212_EX3_Code/mnist_optimiser.out mnist_data $lr 0.0001 $bs 10 1 0.9 0.0 1e-8 \
      > logs/mom_${bs}_${lr}.csv 2>&1
  done
done
