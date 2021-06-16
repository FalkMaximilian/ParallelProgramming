CFLAGS=-Wall -Wextra -g -fopenmp
CC=gcc-11

pmmul: mmul_omp.c
	$(CC) $(CFLAGS) mmul_omp.c -o mmul_omp

.PHONY: clean

clean:
	rm -f mmul_omp
