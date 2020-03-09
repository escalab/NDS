#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

void sequential_blockmm(double *a, double *b, double *c, int matrix_size, int submatrix_size) {
    int i, j, k, ii, jj, kk;
    for(i = 0; i < matrix_size; i+=submatrix_size) {
        for(j = 0; j < matrix_size; j+=submatrix_size) {
            for(k = 0; k < matrix_size; k+=submatrix_size) {        
                for(ii = i; ii < i+submatrix_size; ii++) {
                    for(jj = j; jj < j+submatrix_size; jj++) {
                        for(kk = k; kk < k+submatrix_size; kk++) {
                            c[ii * matrix_size + jj] += a[ii * matrix_size + kk] * b[kk * matrix_size + jj];
                        }
                    }
                }
            }
        }
    }  
}


int main(int argc, char** argv) {
    int i, j, n, sub_n, count = 0;
    FILE *fptr;
    double *a, *b, *c;
    // double *a_sub, *b_sub, *c_sub;
    long duration;
    struct timeval h_start, h_end;
    
    if (argc < 3) {
        printf("usage: %s <path> <matrix size> <submatrix size>\n", argv[0]);
        return 1;
    }

    fptr = fopen(argv[1], "rb");
    n = atoi(argv[2]);
    sub_n = atoi(argv[3]);

    a = (double *) malloc(n * n * sizeof(double));
    b = (double *) malloc(n * n * sizeof(double));
    c = (double *) malloc(n * n * sizeof(double));

    count = fread(a, sizeof(double), n * n, fptr);
    if (count != n * n) {
        printf("reading a matrix incorrectly\n");
        printf("A: \n");
        for(i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                printf("%f ", a[i * n + j]);
            }
            printf("\n");
        }
        printf("\n");
        exit(1);
    }

    fseek(fptr, 0, SEEK_SET);
    count = fread(b, sizeof(double), n * n, fptr);
    if (count != n * n) {
        printf("reading a matrix incorrectly\n");
        printf("B: \n");
        for(i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                printf("%f ", a[i * n + j]);
            }
            printf("\n");
        }
        printf("\n");
        exit(1);
    }

    memset(c, 0, sizeof(double) * n * n);

    gettimeofday(&h_start, NULL);
    sequential_blockmm(a, b, c, n, sub_n);
    gettimeofday(&h_end, NULL);
    duration = ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);
#ifdef DEBUG
    printf("A: \n");
    for(i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            printf("%f ", a[i * n + j]);
        }
        printf("\n");
    }
    printf("\n");
    printf("B: \n");
    for(i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            printf("%f ", b[i * n + j]);
        }
        printf("\n");
    }
    printf("\n");
    printf("C: \n");
    for(i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            printf("%f ", c[i * n + j]);
        }
        printf("\n");
    }
    printf("\n");
#endif
    free(a);
    free(b);
    free(c);
    fclose(fptr);
    printf("duration: %f s\n", (float) duration / 1000000);
    return 0;
}