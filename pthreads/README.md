# Pthreads

This programm calculates the product of 2 nxn matrices with a given amount of threads.

The prgram takes two files containing the nxn matrices and the threadcount as parameters.
The second matrix is read transposed to make better use of the processors cache.

An example matrix "matrix.txt" is given. The first 2 lines of a matrix define it's dimensions. Technically only one is currently needed as the program only works with nxn matrices. 
