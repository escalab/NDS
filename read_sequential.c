#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <fcntl.h>

int main(int argc, char** argv) {
    int i, j, ii, jj, sub_i, offset, n, sub_n;
    FILE *fptr;
    long **sub_matrix;
    
    if (argc < 3) {
        printf("usage: %s <path> <matrix size> <submatrix size>\n", argv[0]);
        return 1;
    }

    fptr = fopen(argv[1], "rb");
    n = atoi(argv[2]);
    sub_n = atoi(argv[3]);

    sub_matrix = (long**) malloc(sub_n * sizeof(long*));
    
    for (i = 0; i < sub_n; i++) {
        sub_matrix[i] = (long *) malloc(sub_n * sizeof(long));
    }

    for (i = 0; i < n; i+= sub_n) {
        for (j = 0; j < n; j+= sub_n) {           
            sub_i = 0;        
            for(ii = i; ii < i+sub_n; ii++) {
                offset = (ii * n + j) * sizeof(long);
                fseek(fptr, offset, SEEK_SET);
                fread(sub_matrix[sub_i], sizeof(long), sub_n, fptr);
                sub_i++;
            }

            // debug purpose
            for(ii = 0; ii < sub_n; ii++) {
                for (jj = 0; jj < sub_n; jj++) {
                    printf("%ld ", sub_matrix[ii][jj]);
                }
                printf("\n");
            }
            printf("\n");
        }
    }

    for (i = 0; i < sub_n; i++) {
        free(sub_matrix[i]);
    }

    free(sub_matrix);
    fclose(fptr);
    return 0;
}