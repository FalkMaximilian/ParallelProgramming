CFLAGS=-Wall -Wextra -g -fopenmp
CC=gcc

pmmul: mmul_omp.c
	$(CC) $(CFLAGS) mmul_omp.c -o mmul_omp

.PHONY: clean

clean:
	rm -f mmul_omp
