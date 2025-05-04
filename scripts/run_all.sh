#!/bin/bash
set -e

echo "▶ Running SGD..."
../COMP36212_EX3_Code/mnist_optimiser.out --safe-sgd

echo "▶ Running Momentum..."
../COMP36212_EX3_Code/mnist_optimiser.out --safe-momentum

echo "▶ Running Adam..."
../COMP36212_EX3_Code/mnist_optimiser.out --safe-adam