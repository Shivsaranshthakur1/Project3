#!/bin/bash

mkdir -p logs

for bs in 1 10 100; do
  for lr in 0.1 0.01 0.001; do
    echo "Running batch size $bs, learning rate $lr"
    ./COMP36212_EX3_Code/mnist_optimiser.out mnist_data $lr 0.0001 $bs 10 0.9 \
      > logs/mom_${bs}_${lr}.csv
  done
done
