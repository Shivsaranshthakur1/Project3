# Existing target
all: mnist_optimiser.out

mnist_optimiser.out:
	gcc -o mnist_optimiser.out main.c mnist_helper.c neural_network.c optimiser.c -lm -O3 -Wall -Werror

# Extended phony targets
.PHONY: run test clean format

run: all
	./mnist_optimiser.out mnist_data/

test:
	@echo "No tests implemented yet."

clean:
	rm -f mnist_optimiser.out

format:
	clang-format -i -style=file *.c *.h

bench:
	time ./mnist_optimiser.out mnist_data/ --epochs 1

gradcheck:
	python3 ../scripts/finite_diff_check.py

plot:
	python3 ../scripts/plot_convergence.py

quickcheck:
	python3 ../scripts/loss_trace.py

run_all:
	../scripts/run_all.sh
	./scripts/run_all.sh
