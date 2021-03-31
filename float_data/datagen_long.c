#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <string.h>

int main(int argc, char** argv) {
    int i, j, ii, jj, n, sub_n;
    FILE *fptr;
    int **output_matrix;
    int *submatrix;
    int idx;
    
    if (argc < 4) {
        printf("usage: %s <normal output path> <block output path> <matrix size> <submatrix size>\n", argv[0]);
        return 1;
    }

    fptr = fopen(argv[1], "w");
    n = atoi(argv[3]);
    sub_n = atoi(argv[4]);

    output_matrix = (int **) malloc(n * sizeof(int *));
    submatrix = (int *) malloc(sub_n * sub_n * sizeof(int));
    
    for (i = 0; i < n; i++) {
        output_matrix[i] = (int *) malloc(n * sizeof(int));
        for (j = 0; j < n; j++) {
            output_matrix[i][j] = (int) i * n + j;
            fwrite(&(output_matrix[i][j]), sizeof(int), 1, fptr);
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
            fwrite(submatrix, sizeof(int), sub_n * sub_n, fptr);
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