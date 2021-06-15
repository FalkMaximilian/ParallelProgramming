#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <errno.h>
#include <time.h>

#ifdef _OPENMP
  #include <omp.h>
#endif

typedef int64_t matrix_elem_t;


typedef struct {
    int rows, cols;
    matrix_elem_t *data;
} matrix_t;


bool read_matrix(char *filepath, matrix_t* matrix)
{
    int i, j, x;
    FILE *fp;
    if ((fp = fopen(filepath, "r")) == NULL) {
        perror("Can't open file!");
        exit(EXIT_SUCCESS);
    }

    fscanf(fp, "%d", &matrix->rows);
    fscanf(fp, "%d", &matrix->cols);

    matrix->data = calloc(matrix->rows * matrix->cols, sizeof(matrix_elem_t));
    if (matrix->data == NULL) {
        return false;
    }

    for (i = 0; i < matrix->rows; ++i) {
        for (j = 0; j < matrix->cols; ++j) {
            fscanf(fp, "%d\t", &x);
            matrix->data[(matrix->cols * i) + j] = x;
        }
        fscanf(fp, "\n");
    }

    fclose(fp);

    return true;
}


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


void matrix_mult(matrix_t* a, matrix_t* b, matrix_t* r)
{
    long i, j, k, y;

    #pragma omp parallel for schedule(static) private(i, j, k, y)
    for (i = 0; i < r->rows; ++i) {
        for (j = 0; j < r->cols; ++j) {
            y = 0;
            for (k = 0; k < a->cols; ++k) {
                y += a->data[i * a->cols + k] * b->data[k * b->cols + j];
            }   
            r->data[i * r->cols + j] = y;
        }
    }
}

bool matrix_mult_simple(matrix_t* a, matrix_t* b, matrix_t* r)
{
    if (a->cols != b->rows) {
        return false;
    }

    r->rows = a->rows;
    r->cols = b->cols;
    r->data = calloc(r->rows * r->cols, sizeof(matrix_elem_t));

    if (r->data == NULL) {
        return false;
    }

    matrix_mult(a, b, r);

    return true;
}


int main(int argc, char* argv[])
{
    double time_1 = omp_get_wtime();

    matrix_t a = { 0, 0, NULL };
    matrix_t b = a;
    matrix_t r = a;

    if (argc != 4) {
        fprintf(stderr, "Usage: %s <file1> <file2> <threadcount>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int t = (int) strtol(argv[3], NULL, 0); // Number of threads
    omp_set_num_threads(t);

    if (!read_matrix(argv[1], &a)) {
        fputs("could not read input matrix A", stderr);
        return EXIT_FAILURE;
    }

    if (!read_matrix(argv[2], &b)) {
        fputs("could not read input matrix B", stderr);
        return EXIT_FAILURE;
    }

    if (matrix_mult_simple(&a, &b, &r)) {
        print_matrix(r);
    } else {
        fputs("could not multiply: mismatch between number of rows and columns in input matricess.", stderr);
    }

    free(a.data);
    free(b.data);
    free(r.data);

    double time_2 = omp_get_wtime();
    fprintf(stderr, "%lf\n", time_2 - time_1);

    return EXIT_SUCCESS;
}

