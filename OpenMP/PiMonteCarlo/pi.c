/** 
 *  @file pi.c
 *  @brief Pi approximation with Monte-Carlo-Approach
 *
 *  This Program will approximate pi using the Monte-Carlo
 *  Method. The Program takes 2 parameters. The number of iterations
 *  that shall be performed and the number of Threads to be used.
 *
 *  @author Maximilian Falk (799269)
 */

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

double monte_carlo_pi(int iter, double *r) {

  int i, count = 0;
  unsigned int seedp = 0;

  #pragma omp parallel private(seedp, i)
  {
    // Each thread needs it's own seed for the random number generator
    seedp = omp_get_thread_num() * time(NULL);

    #pragma omp for
    for (i = 0; i < iter; ++i) {
      r[2 * i] = (double)rand_r(&seedp) / RAND_MAX;       // x value
      r[(2 * i) + 1] = (double)rand_r(&seedp) / RAND_MAX; // y value
    }
  }

  // Check if all the random points are within the circle
  #pragma omp parallel for private(i) reduction(+ : count)
  for (i = 0; i < iter; ++i) {
    if (((r[2 * i] * r[2 * i]) + (r[(2 * i) + 1] * r[(2 * i) + 1])) <= 1) {
      count++;
    }
  }

  return 4.0 * (double)count/((double)iter);
}

int main(int argc, char **argv) {

  double time_1 = omp_get_wtime();

  if (argc != 3) {
    fprintf(stderr, "Usage: %s <iterations> <threadcount>\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  // Number of iterations and number of threads
  int iter = (int) strtol(argv[1], NULL, 0);
  int thrc = (int) strtol(argv[2], NULL, 0);

  // Allocate memory for all random numbers
  double *randomNums = calloc(2 * iter, sizeof(double));
  if (randomNums == NULL) {
    perror("Can't allocate memory for random numbers!");
    exit(EXIT_FAILURE);
  }

  // Set the number of threads
  omp_set_num_threads(thrc);

  // Approximate pi
  double res = monte_carlo_pi(iter, randomNums);
  printf("Approximation of pi:  %lf\n", res);

  free(randomNums);

  double time_2 = omp_get_wtime();
  printf("Time elapsed: %lf seconds\n", time_2 - time_1);
}
