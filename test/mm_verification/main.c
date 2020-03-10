#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>

void blockmm(double *a, double *b, double *c, int n, int sub_n) {
    int i, j, k, ii, jj, kk;
    for(i = 0; i < n; i+=sub_n) {
        for(j = 0; j < n; j+=sub_n) {
            for(k = 0; k < n; k+=sub_n) {        
                for(ii = i; ii < i+sub_n; ii++) {
                    for(jj = j; jj < j+sub_n; jj++) {
                        for(kk = k; kk < k+sub_n; kk++) {
                            c[ii * n + jj] += a[ii * n + kk] * b[kk * n + jj];
                        }
                    }
                }
            }
        }
    }  
}

void mm(double *a, double *b, double *c, int n) {
    int i, j, k;
    for(i = 0; i < n; i++) {
        for(j = 0; j < n; j++) {
            for(k = 0; k < n; k++) {  
                c[i * n + j] += a[i * n + k] * b[k * n + j];      
            }
        }
    }  
}

void submatrix_blockmm(double *a, double *b, double *c, double *a_sub, double *b_sub, double *c_sub, int n, int sub_n) {
    int i, j, k;
    int cross_row = n * sub_n, cross_col = sub_n * sub_n;
    for (i = 0; i < (n/sub_n); i++) {
        for (j = 0; j < (n/sub_n); j++) {
            memcpy(c_sub, (c + i * cross_row + j * cross_col), sub_n * sub_n * sizeof(double));
            for (k = 0; k < (n/sub_n); k++) {
                // fill the block
                // printf("i: %d, j: %d, k: %d\n", i, j, k);
                memcpy(a_sub, (a + i * cross_row + k * cross_col), sub_n * sub_n * sizeof(double));
                memcpy(b_sub, (b + k * cross_row + j * cross_col), sub_n * sub_n * sizeof(double));
                blockmm(a_sub, b_sub, c_sub, sub_n, sub_n);
            }
            memcpy((c + i * cross_row + j * cross_col), c_sub, sub_n * sub_n * sizeof(double));
        }
    }
}

int main(int argc, char** argv) {
    int i, j, ii, jj, n, sub_n, count = 0;
    FILE *fptr;
    double *a, *b, *c;
    double *a_block, *b_block, *c_block, *c_reformat;
    double *a_sub, *b_sub, *c_sub;
    long duration;
    struct timeval h_start, h_end;
    
    if (argc < 4) {
        printf("usage: %s <sequential format path> <tensor format path> <matrix size> <submatrix size>\n", argv[0]);
        return 1;
    }

    n = atoi(argv[3]);
    sub_n = atoi(argv[4]);

    a = (double *) malloc(n * n * sizeof(double));
    b = (double *) malloc(n * n * sizeof(double));
    c = (double *) malloc(n * n * sizeof(double));

    a_block = (double *) malloc(n * n * sizeof(double));
    b_block = (double *) malloc(n * n * sizeof(double));
    c_block = (double *) malloc(n * n * sizeof(double));
    c_reformat = (double *) malloc(n * n * sizeof(double));
    
    a_sub = (double *) malloc(sub_n * sub_n * sizeof(double));
    b_sub = (double *) malloc(sub_n * sub_n * sizeof(double));
    c_sub = (double *) malloc(sub_n * sub_n * sizeof(double));

    // read the sequential format
    fptr = fopen(argv[1], "rb");
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
    fclose(fptr);

    memset(c, 0, sizeof(double) * n * n);
    gettimeofday(&h_start, NULL);
    blockmm(a, b, c, n, sub_n);
    gettimeofday(&h_end, NULL);
    duration = ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);
    printf("sequential format duration: %f s\n", (float) duration / 1000000);

    fptr = fopen(argv[2], "rb");
    count = fread(a_block, sizeof(double), n * n, fptr);
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
    count = fread(b_block, sizeof(double), n * n, fptr);
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
    fclose(fptr);
    
    memset(c_block, 0, sizeof(double) * n * n);
    memset(c_sub, 0, sizeof(double) * sub_n * sub_n);
    // read the tensor form
    gettimeofday(&h_start, NULL);
    submatrix_blockmm(a_block, b_block, c_block, a_sub, b_sub, c_sub, n, sub_n);
    gettimeofday(&h_end, NULL);
    duration = ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);
    printf("tensor format duration: %f s\n", (float) duration / 1000000);

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

    printf("A tensor: \n");
    for(i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            printf("%f ", a_block[i * n + j]);
        }
        printf("\n");
    }
    printf("\n");
    printf("B tensor: \n");
    for(i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            printf("%f ", b_block[i * n + j]);
        }
        printf("\n");
    }
    printf("\n");
    printf("C tensor: \n");
    for(i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            printf("%f ", c_block[i * n + j]);
        }
        printf("\n");
    }
    printf("\n");
#endif
    printf("Reformat from tensor to sequential...\n");
    count = 0;
    for (i = 0; i < n; i += sub_n) {
        for (j = 0; j < n; j += sub_n) {  
            for(ii = i; ii < i + sub_n; ii++) {
                for(jj = j; jj < j + sub_n; jj++) {
                    c_reformat[ii * n + jj] = c_block[count];
                    // printf("ii: %d, jj: %d\n", ii, jj);
                    count++;
                }
            }
        }
    }      
    printf("Verifying correctness of the computations...\n");
    count = 0;
    
    for(i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (fabs(c[i * n + j] - c_reformat[i * n + j]) > 0.1f) {
                printf("mismatch at [%d][%d] c=%f c_reformat=%f\n", i, j, c[i * n + j], c_reformat[i * n + j]);
                count++;
            }
        }
    }

    if (count == 0) {
        printf("test passed\n");
    }

    free(a);
    free(b);
    free(c);
    free(a_block);
    free(b_block);
    free(c_block);
    free(c_reformat);
    free(a_sub);
    free(b_sub);
    free(c_sub);
    return 0;
}