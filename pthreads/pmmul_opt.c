#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <pthread.h>
#include <stdbool.h>
#include <errno.h>
#include <time.h>

#define TIME_GET(timer) struct timespec timer; clock_gettime(CLOCK_MONOTONIC , &timer)
#define TIME_DIFF(timer1 , timer2) ((timer2.tv_sec * 1.0E+9 + timer2.tv_nsec) - (timer1.tv_sec * 1.0E+9 + timer1.tv_nsec)) / 1.0E+9

typedef int64_t matrix_elem_t;

typedef struct {
    int rows, cols;
    matrix_elem_t * data;
} matrix_t;

typedef struct {
    pthread_t thread_id;
    matrix_t * a, * b, *r;
    int i, j;
} thread_info;

void print_matrix(matrix_t m){
    int i, j;
    matrix_elem_t sum;

    sum = 0;
    for (i = 0; i < m.rows; ++i) {
        for (j = 0; j < m.cols; ++j) {
            sum += m.data[(i * m.cols) + j];
            printf("%ld\t ", m.data[(i * m.cols) + j]);
        }
        printf("\n");
    }
    printf("sum: %ld\n", sum);
}

bool read_matrix(char *filepath, matrix_t* matrix, bool read_transposed)
{
    int i, j, x;
    FILE *fp;

    // Try to open the file with read permissions
    if ((fp = fopen(filepath, "r")) == NULL) {
        perror("Can't open file!");
        exit(EXIT_SUCCESS);
    }

    // Read the matrix dimenstions
    fscanf(fp, "%d", &matrix->rows);
    fscanf(fp, "%d", &matrix->cols);    

    // Allocate space for matrix
    matrix->data = calloc(matrix->rows * matrix->cols, sizeof(matrix_elem_t));
    if (matrix->data == NULL) {
        return false;
    }

    // Read the matrix or read and transpose the matrix.
    if (read_transposed) {
        for (i = 0; i < matrix->rows; ++i) {
            for (j = 0; j < matrix->cols; ++j) {
                fscanf(fp, "%d\t", &x);
                matrix->data[(matrix->cols * j) + i] = x;
            }
            fscanf(fp, "\n");   // Newline at the end of each row
        }
    } else {
        for (i = 0; i < matrix->rows; ++i) {
            for (j = 0; j < matrix->cols; ++j) {
                fscanf(fp, "%d\t", &x);
                matrix->data[(matrix->cols * i) + j] = x;
            }
            fscanf(fp, "\n");
        }
    }

    fclose(fp); // Close the file
    return true;
}

/**
* Calculate a given part of the matrix.
* Matrices A and B are both nxn. 
*
* The number of threads passed to the program specifies that
* each thread calculates the results for n/t rows of matrix A.
* The last thread will also calculate the remainder rows of Matrix A.
* 
* I.e. given two 5x5 matrices and 2 Threads the first thread will calculate rows 0 to 1 and
* the second thread will calculate rows 2 to 4.
*/
void * matrix_mult(void * arg) {
    
    thread_info * info = arg;

    int x, i, j;
    matrix_elem_t sum;

    // info->a-cols is used throughout the program because we only calculate nxn matrices so all dimensions are the same.
    // Using only the same value may result in faster performance since the value will probably always be in the cache.
    for (i = info->i; i < info->j; ++i) {
        for (j = 0; j < info->a->cols; ++j) {
            sum = 0;
            for (x = 0; x < info->a->cols; ++x) {
                sum += info->a->data[(i * info->a->cols) + x] * info->b->data[(j * info->a->cols) + x];
            }
            info->r->data[(i * info->a->cols) + j] = sum;
        }
    }

    return NULL;
}

bool matrix_mult_threaded(matrix_t * a, matrix_t * b, matrix_t * r, int t) {

    // Both input matrices have to have the same dimensions
    if ((a->cols != b->rows) || (a->cols != a->rows)) {
        return false;
    }

    // All dimension are the same
    r->rows = a->rows;
    r->cols = a->rows;

    int rows_per_thread = a->rows / t; // Number of rows (of Matrix A) each thread shall calculate
    int remainder = a->rows % t;    // The last thread will also have to calculate the remainder rows

    // Try to allocate memory for the result matrix
    r->data = calloc((r->rows * r->cols), sizeof(matrix_elem_t));
    if (r->data == NULL) {
        perror("Could not allocate memory for result matrix!");
        exit(EXIT_FAILURE);
    }

    int i, s;

    // Try to allocate memory for the thread elements
    thread_info * tinfo = calloc(t, sizeof(thread_info));
    if (tinfo == NULL) {
        perror("Could not allocate memory for thread elements!");
        exit(EXIT_FAILURE);
    }

    for (i = 0; i < t; ++i) {
        tinfo[i].a = a;
        tinfo[i].b = b;
        tinfo[i].r = r;
        tinfo[i].i = i*rows_per_thread;     // First row for this thread
        tinfo[i].j = (i+1)*rows_per_thread; // First row that should *not* be calculate by this thread

        if (i == (t-1)) {
            tinfo[i].j = tinfo[i].j + remainder; // Last thread needs to calculate the remaining rows too.
        }

        // Try to create a new thread
        s = pthread_create(&tinfo[i].thread_id, NULL, &matrix_mult, (void *) &tinfo[i]);
        if (s != 0) {
            perror("Couldnt create thread!");
            exit(EXIT_FAILURE);
        }
    }

    // Join all threads
    for (i = 0; i < t; ++i) {
        pthread_join(tinfo[i].thread_id, NULL);
    }

    free(tinfo);
    return true;
}

int main(int argc, char **argv) {

    TIME_GET(timer_1);

    matrix_t a = {0, 0, NULL};
    matrix_t b = a;
    matrix_t r = a;

    if (argc != 4) {
        fprintf(stderr, "Usage: %s <file1> <file2> <threadcount>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int t = (int) strtol(argv[3], NULL, 0); // Number of threads

    if (!read_matrix(argv[1], &a, false)) {
        fputs("could not read input matrix A", stderr);
        return EXIT_FAILURE;
    }

    if (!read_matrix(argv[2], &b, true)) {
        fputs("could not read input matrix B", stderr);
        return EXIT_FAILURE;
    }

    if (matrix_mult_threaded(&a, &b, &r, t)) {
        print_matrix(r);
    } else {
        fputs("could not multiply: mismatch between number of rows and columns in in put matricess.", stderr);
    }

    free(a.data);
    free(b.data);
    free(r.data);

    TIME_GET(timer_2);
    fprintf(stderr, "%lf\n", TIME_DIFF(timer_1, timer_2));

    return EXIT_SUCCESS;
}
