#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>

int main(int argc, char** argv) {
    int i, j, ii, jj, n, sub_n;
    FILE *fptr;
    double **output_matrix;
    double *submatrix;
    int idx;
    
    if (argc < 4) {
        printf("usage: %s <normal output path> <block output path> <matrix size> <submatrix size>\n", argv[0]);
        return 1;
    }

    srand(5);

    fptr = fopen(argv[1], "w");
    n = atoi(argv[3]);
    sub_n = atoi(argv[4]);

    output_matrix = (double **) malloc(n * sizeof(double *));
    submatrix = (double *) malloc(sub_n * sub_n * sizeof(double));
    
    for (i = 0; i < n; i++) {
        output_matrix[i] = (double *) malloc(n * sizeof(double));
        for (j = 0; j < n; j++) {
            output_matrix[i][j] = (double) rand() / RAND_MAX;
            fwrite(&(output_matrix[i][j]), sizeof(double), 1, fptr);
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
            fwrite(submatrix, sizeof(double), sub_n * sub_n, fptr);
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