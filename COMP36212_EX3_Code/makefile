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
test:
	@echo "No tests implemented yet."

bench:
	time ./mnist_optimiser.out mnist_data/ --epochs 1
