#include <mpi.h>
#include <errno.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define TAG 123

typedef double matrix_elem_t;

/* Matrix can only hold nxn matrices.
 * No need to hold both dimenstions. */
typedef struct {
  int dim;
  matrix_elem_t *data;
} matrix_t;

/* This function intializes a given Matrix by the scheme of
 *  m_ij = (i / j+1) */
void initialize_matrix(matrix_t mat) {
    int i, j;
    for (i = 0; i < mat.dim; ++i) {
      for (j = 0; j < mat.dim; ++j) {
        // Division by 0 cannot happen. No extra check.
        mat.data[(i * mat.dim) + j] = ((double)i / (j + 1));
      }
    }
}

// Prints the matrix to a file
void print_matrix(matrix_t m) {
  int i, j;
  FILE *result;
  result = fopen("c.txt", "w");
  if (result == NULL) {
    perror("Could not create file!");
    exit(EXIT_FAILURE);
  }

  for (i = 0; i < m.dim; ++i) {
    for (j = 0; j < m.dim; ++j) {
      fprintf(result, "%lf\t ", m.data[(i * m.dim) + j]);
    }
    fprintf(result, "\n");
  }
  fclose(result);
}

int main(int argc, char ** argv) {

    if (argc != 2) {
        fprintf(stderr, "Usage: %s <dimension>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int dim = (int) strtol(argv[1], NULL, 0);

    matrix_t A = {dim, NULL};
    matrix_t B = {dim, NULL};
    matrix_t R = {dim, NULL};

    A.data = calloc(dim * dim, sizeof(matrix_elem_t));
    if (A.data == NULL) {
        perror("Could not allocate memory.");
        exit(EXIT_FAILURE);
    }

    B.data = calloc(dim * dim, sizeof(matrix_elem_t));
    if (A.data == NULL) {
        perror("Could not allocate memory.");
        exit(EXIT_FAILURE);
    }

    R.data = calloc(dim * dim, sizeof(matrix_elem_t));
    if (A.data == NULL) {
        perror("Could not allocate memory.");
        exit(EXIT_FAILURE);
    }

    initialize_matrix(A);
    initialize_matrix(B);
    

    MPI_Init(&argc, &argv);

    int nprocs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Bcast(B.data , dim * dim, MPI_DOUBLE , 0, MPI_COMM_WORLD);
    MPI_Bcast(R.data , dim * dim, MPI_DOUBLE , 0, MPI_COMM_WORLD);

    int rows_per_proc = dim / (nprocs - 1); // Number of rows each process shall calculate
    int remainder = dim % (nprocs - 1); // Number of rows left over

    double start, elapsed, time;
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();

    // Main process 
    if (!rank) {

        // Send the corresponding rows to each process
        for (int i = 1; i < nprocs; i++) {
            if (i != (nprocs - 1)) {
                MPI_Send(&A.data[(i - 1) * rows_per_proc * dim], rows_per_proc * dim, MPI_DOUBLE, i, TAG, MPI_COMM_WORLD);
            } else {
                // The last process also receives the leftover rows
                MPI_Send(&A.data[(i - 1) * rows_per_proc * dim], (rows_per_proc + remainder) * dim, MPI_DOUBLE, i, TAG, MPI_COMM_WORLD);
            }
        }

        // Receive the results from the corresponding processes
        MPI_Status status;
        for (int i = 1; i < nprocs; i++) {
            if (i != (nprocs - 1)) {
                MPI_Recv(&R.data[(i - 1) * rows_per_proc * dim], rows_per_proc * dim, MPI_DOUBLE, i, TAG , MPI_COMM_WORLD, &status);
            } else {
                // The last process will send the more data back due to the leftover rows
                MPI_Recv(&R.data[(i - 1) * rows_per_proc * dim], (rows_per_proc + remainder) * dim, MPI_DOUBLE, i, TAG, MPI_COMM_WORLD, &status);
            }
        }
    } else if (rank == (nprocs - 1)) {
        // Last process has some differences in the way the result is calculated due to the remainder.
        MPI_Status status;
        MPI_Recv(&A.data[(rank - 1) * rows_per_proc * dim], (rows_per_proc + remainder) * dim, MPI_DOUBLE, 0, TAG, MPI_COMM_WORLD, &status);

        matrix_elem_t sum;
        for (int i = 0; i < (rows_per_proc + remainder); i++) {
            for(int j = 0; j < dim; j++) {
                sum = 0.0;
                for (int x = 0; x < dim; x++) {
                    sum += A.data[((rank - 1) * rows_per_proc * dim) + (i * dim) + x] * B.data[x * dim + j];
                }
                R.data[((rank - 1) * rows_per_proc * dim) + (i * dim) + j] = sum;
            }
        }

        MPI_Send(&R.data[(rank - 1) * rows_per_proc * dim], (rows_per_proc + remainder) * dim, MPI_DOUBLE, 0, TAG, MPI_COMM_WORLD);
    } else {
        // All the worker processes have to calculate the corresponding rows

        MPI_Status status;
        MPI_Recv(&A.data[(rank - 1) * rows_per_proc * dim], rows_per_proc * dim, MPI_DOUBLE, 0, TAG, MPI_COMM_WORLD, &status);

        matrix_elem_t sum;
        for (int i = 0; i < rows_per_proc; i++) {
            for (int j = 0; j < dim; j++) {
                sum = 0.0;
                for (int x = 0; x < dim; x++) {
                    sum += A.data[((rank - 1) * rows_per_proc * dim) + (i * dim) + x] * B.data[x * dim + j];
                }
                R.data[((rank - 1) * rows_per_proc * dim) + (i * dim) + j] = sum;
            }
        }

        MPI_Send(&R.data[(rank - 1) * rows_per_proc * dim], rows_per_proc * dim, MPI_DOUBLE, 0, TAG, MPI_COMM_WORLD);
    }

    elapsed = MPI_Wtime() - start;
    MPI_Reduce( &elapsed, &time, 1, MPI_DOUBLE, MPI_MAX, 0 , MPI_COMM_WORLD);

    if (!rank) {
        fprintf(stderr, "Time used: %14.8f seconds\n", time);
        print_matrix(R);
    }

    MPI_Finalize();
}