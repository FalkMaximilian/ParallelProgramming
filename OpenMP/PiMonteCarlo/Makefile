CFLAGS=-Wall -Wextra -g -fopenmp
CC=gcc-11

pmmul: pi.c
	$(CC) $(CFLAGS) pi.c -o pi

.PHONY: clean

clean:
	rm -f pi
