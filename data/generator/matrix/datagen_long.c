#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>

int main(int argc, char** argv) {
    int i, j, ii, jj, n, sub_n;
    FILE *fptr;
    long **output_matrix;
    long *submatrix;
    int idx;
    
    if (argc < 4) {
        printf("usage: %s <normal output path> <block output path> <matrix size> <submatrix size>\n", argv[0]);
        return 1;
    }

    fptr = fopen(argv[1], "w");
    n = atoi(argv[3]);
    sub_n = atoi(argv[4]);

    output_matrix = (long **) malloc(n * sizeof(long *));
    submatrix = (long *) malloc(sub_n * sub_n * sizeof(long));
    
    for (i = 0; i < n; i++) {
        output_matrix[i] = (long *) malloc(n * sizeof(long));
        for (j = 0; j < n; j++) {
            output_matrix[i][j] = (long) i * n + j;
            fwrite(&(output_matrix[i][j]), sizeof(long), 1, fptr);
        }
    }

    fclose(fptr);

    fptr = fopen(argv[2], "w");

    for(i = 0; i < n; i+= sub_n) {
        for(j = 0; j < n; j+= sub_n) {  
            idx = 0;     
            for(ii = i; ii < i+sub_n; ii++) {
                for(jj = j; jj < j+sub_n; jj++) {
                    submatrix[idx] = output_matrix[ii][jj];
                    idx++;
                }
            }
            fwrite(submatrix, sizeof(long), sub_n * sub_n, fptr);
        }
    }  

    fclose(fptr);
    for (i = 0; i < n; i++) {
        free(output_matrix[i]);
    }    
    free(output_matrix);

    free(submatrix);

    return 0;
}