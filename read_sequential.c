#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <fcntl.h>

int main(int argc, char** argv) {
    int i, j, ii, jj, sub_i, offset, n, sub_n;
    FILE *fptr;
    long *sub_matrix;
    long duration;
    struct timeval h_start, h_end;
    if (argc < 3) {
        printf("usage: %s <path> <matrix size> <submatrix size>\n", argv[0]);
        return 1;
    }

    fptr = fopen(argv[1], "rb");
    n = atoi(argv[2]);
    sub_n = atoi(argv[3]);

    sub_matrix = (long*) malloc(sub_n * sub_n * sizeof(long));

    gettimeofday(&h_start, NULL);
    for (i = 0; i < n; i+= sub_n) {
        for (j = 0; j < n; j+= sub_n) {           
            sub_i = 0;        
            for(ii = i; ii < i+sub_n; ii++) {
                offset = (ii * n + j) * sizeof(long);
                fseek(fptr, offset, SEEK_SET);
                fread((sub_matrix + sub_i * sub_n), sizeof(long), sub_n, fptr);
                sub_i++;
            }

        #ifdef DEBUG
            // debug purpose
            for(ii = 0; ii < sub_n; ii++) {
                for (jj = 0; jj < sub_n; jj++) {
                    printf("%ld ", sub_matrix[ii * sub_n + jj]);
                }
                printf("\n");
            }
            printf("\n");
        #endif
        }
    }
    gettimeofday(&h_end, NULL);
    duration = ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);

    free(sub_matrix);
    fclose(fptr);
    printf("duration: %f s\n", (float) duration / 1000000);

    return 0;
}