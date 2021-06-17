#include <stdlib.h>
#include "mpi.h"
#include <stdio.h>
#include <string.h>
#include "random.h"
#include "md5tool.h"

// Tag for MPI communication
#define TAG 2021

/* horizontal size of the configuration */
#define XSIZE 1024

/* "ADT" State and line of states (plus border) */
typedef char State;
typedef State Line[XSIZE + 2];

/* determine random integer between 0 and n-1 */
#define randInt(n) ((int)(nextRandomLEcuyer() * n))

/* random starting configuration */
static void initConfig(Line *buf, int lines) {  
   int x, y;

   initRandomLEcuyer(424243);
   for (y = 1;  y <= lines;  y++) {
      for (x = 1;  x <= XSIZE;  x++) {
         buf[y][x] = randInt(100) >= 50;
      }
   }
}

static void boundary(Line *buf, int lines, int top, int bot) {  

   for (int y = 0;  y <= lines+1;  y++) {
      /* copy rightmost column to the buffer column 0 */
      buf[y][0      ] = buf[y][XSIZE];

      /* copy leftmost column to the buffer column XSIZE + 1 */
      buf[y][XSIZE+1] = buf[y][1    ];
   }

   MPI_Status status;
   // Send bottommost line and receive the one sent to this process
   MPI_Send(&buf[lines], sizeof(Line), MPI_CHAR, bot, TAG, MPI_COMM_WORLD);
   MPI_Recv(&buf[0], sizeof(Line), MPI_CHAR, top, TAG, MPI_COMM_WORLD, &status);

   // Send topmost line and receive the one sent to this process
   MPI_Send(&buf[1], sizeof(Line), MPI_CHAR, top, TAG, MPI_COMM_WORLD);
   MPI_Recv(&buf[lines + 1], sizeof(Line), MPI_CHAR, bot, TAG, MPI_COMM_WORLD, &status);
}

/* annealing rule from ChoDro96 page 34 
 * the table is used to map the number of nonzero
 * states in the neighborhood to the new state
 */
static State anneal[10] = {0, 0, 0, 0, 1, 0, 1, 1, 1, 1};

/* a: pointer to array; x,y: coordinates; result: n-th element of anneal,
      where n is the number of neighbors */
#define transition(a, x, y) \
   (anneal[(a)[(y)-1][(x)-1] + (a)[(y)][(x)-1] + (a)[(y)+1][(x)-1] +\
           (a)[(y)-1][(x)  ] + (a)[(y)][(x)  ] + (a)[(y)+1][(x)  ] +\
           (a)[(y)-1][(x)+1] + (a)[(y)][(x)+1] + (a)[(y)+1][(x)+1]])

/* make one simulation iteration with lines lines.
 * old configuration is in from, new one is written to to.
 */
static void simulate(Line *from, Line *to, int lines)
{
   int x,y;

   for (y = 1;  y <= lines;  y++) {
      for (x = 1;  x <= XSIZE;  x++) {
         to[y][x  ] = transition(from, x  , y);
      }
   }
}

int main(int argc, char **argv) {

   if (argc != 3) {
      fprintf(stderr, "Usage: %s <height of grid> <iterations>\n", argv[0]);
      exit(EXIT_FAILURE);
   }

   int numberOfLines, its;       // Lines in grid and iterations
   int nprocs, rank, procLines;  // Process relevant values 
   int topRecip, botRecip;       // Neighbors of process i
   Line *current, *next, *temp;  // Sub-grids of process i

   numberOfLines = (int) strtol(argv[1], NULL, 0);
   its = (int) strtol(argv[2], NULL, 0);

   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   // Lines per process
   if (rank != (nprocs - 1)) {
      procLines = (numberOfLines / nprocs);
   } else {
      procLines = (numberOfLines / nprocs) + (numberOfLines % nprocs);
   }

   botRecip = (rank + 1) % nprocs;  // Neighbor that receives bottommost line
   if (!rank) {
      topRecip = nprocs - 1; // Recipient of topmost line of process 0
   } else {
      topRecip = rank - 1; // Recipient of topmost
   }

   // Try to allocate memory
   current = malloc((procLines + 2) * sizeof(Line));
   next = malloc((procLines + 2) * sizeof(Line));

   if (current == NULL || next == NULL) {
      perror("Could not allocate memory in process.\n");
      MPI_Abort(MPI_COMM_WORLD, MPI_ERR_UNKNOWN);
      MPI_Finalize();
   }

   Line *final;
   if (!rank) {

      // Try to allocate memory for the grid
      final = malloc((numberOfLines + 2) * sizeof(Line));
      if (final == NULL) {
         perror("Could not allocate memory for final.");
         MPI_Abort(MPI_COMM_WORLD, MPI_ERR_UNKNOWN);
         MPI_Finalize();
      }

      // Initialize the Grid
      initConfig(final, numberOfLines);

      // Copy the first procLines into current of process 0
      for (int i = 1; i <= procLines; i++) {
         for (int j = 0; j < sizeof(Line); j++) {
            current[i][j] = final[i][j];
         }
      }

      // Send all the other chunks to the other processes
      for (int i = 1; i < nprocs; i++) {
         if (i != nprocs - 1) {
            MPI_Send(&final[i * procLines + 1], procLines * sizeof(Line), MPI_CHAR, i, TAG, MPI_COMM_WORLD);
         } else {
            MPI_Send(&final[i * procLines + 1], (procLines + (numberOfLines % nprocs)) * sizeof(Line), MPI_CHAR, i, TAG, MPI_COMM_WORLD);
         }
      }
   } else {
      MPI_Status status;
      MPI_Recv(current[1], procLines * sizeof(Line), MPI_CHAR, 0, TAG, MPI_COMM_WORLD, &status);
   }

   for (int i = 0; i < its; i++) {
      boundary(current, procLines, topRecip, botRecip);
      simulate(current, next, procLines);

      temp = current;
      current = next;
      next = temp;
   }

   // Alle Prozesse senden ihr finales Gitter an 0
   if (!rank) {
      
      // Copy process 0's data into final
      for (int i = 1; i <= procLines; i++) {
         for (int j = 0; j < sizeof(Line); j++) {
            final[i][j] = current[i][j];
         }
      }

      // Collect all the data from the other processes into final
      MPI_Status status;
      for (int i = 1; i < nprocs; i++) {
         if (i != (nprocs - 1)) {
            MPI_Recv(&final[(i * procLines) + 1], procLines * sizeof(Line), MPI_CHAR, i, TAG, MPI_COMM_WORLD, &status);
         } else {
            MPI_Recv(&final[(i * procLines) + 1], (procLines + (numberOfLines % nprocs)) * sizeof(Line), MPI_CHAR, i, TAG, MPI_COMM_WORLD, &status);
         }
      }

      char *hash;
      hash = getMD5DigestStr(final[1], sizeof(Line) * numberOfLines);
      printf("hash: %s\n", hash);

      free(final);
      free(hash);
   } else {
      MPI_Send(&current[1], sizeof(Line) * procLines, MPI_CHAR, 0, TAG, MPI_COMM_WORLD);
   }

   free(current);
   free(next);
   MPI_Barrier(MPI_COMM_WORLD);
   
   MPI_Finalize();
}
