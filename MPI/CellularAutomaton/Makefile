CC=mpicc
LDFLAGS=-lcrypto

.PHONY: clean

capar: capar.c random.c md5tool.c
	$(CC) $^ $(LDFLAGS) -o $@

clean:
	rm -rf *.o
	rm -rf capar
