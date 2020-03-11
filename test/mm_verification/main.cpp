#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>
#include "half.hpp"

void blockmm(double *a, double *b, float *c, int n, int sub_n) {
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

void blockmm(half_float::half *a, half_float::half *b, float *c, int n, int sub_n) {
    int i, j, k, ii, jj, kk;
    for(i = 0; i < n; i+=sub_n) {
        for(j = 0; j < n; j+=sub_n) {
            for(k = 0; k < n; k+=sub_n) {        
                for(ii = i; ii < i+sub_n; ii++) {
                    for(jj = j; jj < j+sub_n; jj++) {
                        for(kk = k; kk < k+sub_n; kk++) {
                            c[ii * n + jj] += float(a[ii * n + kk] * b[kk * n + jj]);
                        }
                    }
                }
            }
        }
    }  
}

void sequential_blockmm(half_float::half *a, half_float::half *b, float *c, half_float::half *a_sub, half_float::half *b_sub, float *c_sub, int n, int sub_n) {
    int i, j, k, ii, jj, kk, i_idx, j_idx, k_idx;
    for(i = 0; i < n; i += sub_n) {
        for(j = 0; j < n; j += sub_n) {
            for(k = 0; k < n; k += sub_n) {
                for (ii = i, i_idx = 0; ii < (i + sub_n); ii++, i_idx++) {
                    for (jj = j, j_idx = 0; jj < (j + sub_n); jj++, j_idx++) {
                        c_sub[i_idx * sub_n + j_idx] = c[ii * n + jj];
                    }
                }
                for (ii = i, i_idx = 0; ii < (i + sub_n); ii++, i_idx++) {
                    for (kk = k, k_idx = 0; kk < (k + sub_n); kk++, k_idx++) {
                        a_sub[i_idx * sub_n + k_idx] = a[ii*n + kk];         
                    }
                }

                for (jj = j, j_idx = 0; jj < (j + sub_n); jj++, j_idx++) {
                    for (kk = k, k_idx = 0; kk < (k + sub_n); kk++, k_idx++) {
                        b_sub[k_idx * sub_n + j_idx] = b[kk * n + jj];
                    }
                }
                blockmm(a_sub, b_sub, c_sub, sub_n, sub_n);
                for (ii = i, i_idx = 0; ii < (i + sub_n); ii++, i_idx++) {
                    for (jj = j, j_idx = 0; jj < (j + sub_n); jj++, j_idx++) {
                        c[ii * n + jj] = c_sub[i_idx * sub_n + j_idx];
                    }
                }  
            }              
        }
    }  
}

void sequential_blockmm(double *a, double *b, float *c, double *a_sub, double *b_sub, float *c_sub, int n, int sub_n) {
    int i, j, k, ii, jj, kk, i_idx, j_idx, k_idx;
    for(i = 0; i < n; i += sub_n) {
        for(j = 0; j < n; j += sub_n) {
            for(k = 0; k < n; k += sub_n) {
                for (ii = i, i_idx = 0; ii < (i + sub_n); ii++, i_idx++) {
                    for (jj = j, j_idx = 0; jj < (j + sub_n); jj++, j_idx++) {
                        c_sub[i_idx * sub_n + j_idx] = c[ii * n + jj];
                    }
                }
                for (ii = i, i_idx = 0; ii < (i + sub_n); ii++, i_idx++) {
                    for (kk = k, k_idx = 0; kk < (k + sub_n); kk++, k_idx++) {
                        a_sub[i_idx * sub_n + k_idx] = a[ii*n + kk];         
                    }
                }

                for (jj = j, j_idx = 0; jj < (j + sub_n); jj++, j_idx++) {
                    for (kk = k, k_idx = 0; kk < (k + sub_n); kk++, k_idx++) {
                        b_sub[k_idx * sub_n + j_idx] = b[kk * n + jj];
                    }
                }
                blockmm(a_sub, b_sub, c_sub, sub_n, sub_n);
                for (ii = i, i_idx = 0; ii < (i + sub_n); ii++, i_idx++) {
                    for (jj = j, j_idx = 0; jj < (j + sub_n); jj++, j_idx++) {
                        c[ii * n + jj] = c_sub[i_idx * sub_n + j_idx];
                    }
                }  
            }              
        }
    }  
}

void submatrix_blockmm(half_float::half *a, half_float::half *b, float *c, half_float::half *a_sub, half_float::half *b_sub, float *c_sub, int n, int sub_n) {
    int i, j, k;
    int cross_row = n * sub_n, cross_col = sub_n * sub_n;
    for (i = 0; i < (n/sub_n); i++) {
        for (j = 0; j < (n/sub_n); j++) {
            memcpy(c_sub, (c + i * cross_row + j * cross_col), sub_n * sub_n * sizeof(float));
            for (k = 0; k < (n/sub_n); k++) {
                // fill the block
                // printf("i: %d, j: %d, k: %d\n", i, j, k);
                memcpy(a_sub, (a + i * cross_row + k * cross_col), sub_n * sub_n * sizeof(half_float::half));
                memcpy(b_sub, (b + k * cross_row + j * cross_col), sub_n * sub_n * sizeof(half_float::half));
                blockmm(a_sub, b_sub, c_sub, sub_n, sub_n);
            }
            memcpy((c + i * cross_row + j * cross_col), c_sub, sub_n * sub_n * sizeof(float));
        }
    }
}

void submatrix_blockmm(double *a, double *b, float *c, double *a_sub, double *b_sub, float *c_sub, int n, int sub_n) {
    int i, j, k;
    int cross_row = n * sub_n, cross_col = sub_n * sub_n;
    for (i = 0; i < (n/sub_n); i++) {
        for (j = 0; j < (n/sub_n); j++) {
            memcpy(c_sub, (c + i * cross_row + j * cross_col), sub_n * sub_n * sizeof(float));
            for (k = 0; k < (n/sub_n); k++) {
                // fill the block
                // printf("i: %d, j: %d, k: %d\n", i, j, k);
                memcpy(a_sub, (a + i * cross_row + k * cross_col), sub_n * sub_n * sizeof(double));
                memcpy(b_sub, (b + k * cross_row + j * cross_col), sub_n * sub_n * sizeof(double));
                blockmm(a_sub, b_sub, c_sub, sub_n, sub_n);
            }
            memcpy((c + i * cross_row + j * cross_col), c_sub, sub_n * sub_n * sizeof(float));
        }
    }
}

int main(int argc, char** argv) {
    int i, j, ii, jj, n, sub_n, count = 0, need_output;
    FILE *fptr;
    double *tmp_arr;
    half_float::half *a, *b;
    half_float::half *a_block, *b_block;
    half_float::half *a_sub, *b_sub;
    float *c, *c_valid, *c_block, *c_reformat, *c_sub;
    long duration;
    struct timeval h_start, h_end;
    
    if (argc < 6) {
        printf("usage: %s <sequential format path> <tensor format path> <matrix size> <submatrix size> <output?>\n", argv[0]);
        return 1;
    }

    n = atoi(argv[3]);
    sub_n = atoi(argv[4]);
    need_output = atoi(argv[5]);

    tmp_arr = (double *) malloc(n * n * sizeof(double));

    a = (half_float::half *) malloc(n * n * sizeof(half_float::half));
    b = (half_float::half *) malloc(n * n * sizeof(half_float::half));
    c = (float *) malloc(n * n * sizeof(float));
    c_valid = (float *) malloc(n * n * sizeof(float));

    a_block = (half_float::half *) malloc(n * n * sizeof(half_float::half));
    b_block = (half_float::half *) malloc(n * n * sizeof(half_float::half));
    c_block = (float *) malloc(n * n * sizeof(float));
    c_reformat = (float *) malloc(n * n * sizeof(float));
    
    a_sub = (half_float::half *) malloc(sub_n * sub_n * sizeof(half_float::half));
    b_sub = (half_float::half *) malloc(sub_n * sub_n * sizeof(half_float::half));
    c_sub = (float *) malloc(sub_n * sub_n * sizeof(float));

    // read the sequential format
    fptr = fopen(argv[1], "rb");
    count = fread(tmp_arr, sizeof(double), n * n, fptr);
    if (count != n * n) {
        printf("reading a matrix incorrectly\n");
#ifdef DEBUG
        printf("A: \n");
        for(i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                printf("%f ", a[i * n + j]);
            }
            printf("\n");
        }
        printf("\n");
#endif
        exit(1);
    }
    fseek(fptr, 0, SEEK_SET);
    for (i = 0; i < n * n; i++) {
        a[i] = half_float::half(tmp_arr[i]);
    }

    count = fread(tmp_arr, sizeof(double), n * n, fptr);
    if (count != n * n) {
        printf("reading a matrix incorrectly\n");
#ifdef DEBUG
        printf("B: \n");
        for(i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                printf("%f ", a[i * n + j]);
            }
            printf("\n");
        }
        printf("\n");
#endif
        exit(1);
    }
    fclose(fptr);
    for (i = 0; i < n * n; i++) {
        b[i] = half_float::half(tmp_arr[i]);
    }

    memset(c_valid, 0, sizeof(float) * n * n);
    gettimeofday(&h_start, NULL);
    blockmm(a, b, c_valid, n, sub_n);
    gettimeofday(&h_end, NULL);
    duration = ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);
    printf("in memory sequential format duration: %f s\n", (float) duration / 1000000);


    memset(c, 0, sizeof(float) * n * n);
    gettimeofday(&h_start, NULL);
    sequential_blockmm(a, b, c, a_sub, b_sub, c_sub, n, sub_n);
    gettimeofday(&h_end, NULL);
    duration = ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);
    printf("external memory sequential format duration: %f s\n", (float) duration / 1000000);


    fptr = fopen(argv[2], "rb");
    count = fread(tmp_arr, sizeof(double), n * n, fptr);
    if (count != n * n) {
        printf("reading a matrix incorrectly\n");
#ifdef DEBUG
        printf("A: \n");
        for(i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                printf("%f ", a[i * n + j]);
            }
            printf("\n");
        }
        printf("\n");
#endif
        exit(1);
    }
    for (i = 0; i < n * n; i++) {
        a_block[i] = half_float::half(tmp_arr[i]);
    }

    fseek(fptr, 0, SEEK_SET);
    count = fread(tmp_arr, sizeof(double), n * n, fptr);
    if (count != n * n) {
        printf("reading a matrix incorrectly\n");
#ifdef DEBUG
        printf("B: \n");
        for(i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                printf("%f ", a[i * n + j]);
            }
            printf("\n");
        }
        printf("\n");
#endif
        exit(1);
    }
    fclose(fptr);
    
    for (i = 0; i < n * n; i++) {
        b_block[i] = half_float::half(tmp_arr[i]);
    }
    memset(c_block, 0, sizeof(float) * n * n);
    memset(c_sub, 0, sizeof(float) * sub_n * sub_n);

    // read the tensor form
    gettimeofday(&h_start, NULL);
    submatrix_blockmm(a_block, b_block, c_block, a_sub, b_sub, c_sub, n, sub_n);
    gettimeofday(&h_end, NULL);
    duration = ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);
    printf("external memory in tensor format duration: %f s\n", (float) duration / 1000000);

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
            printf("%f ", c_valid[i * n + j]);
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
    gettimeofday(&h_start, NULL);
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
    gettimeofday(&h_end, NULL);
    duration = ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);
    printf("Reformat from tensor to sequential duration: %f s\n", (float) duration / 1000000);  
   
    printf("Verifying correctness of external tensor format...\n");
    count = 0;
    for(i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (fabs(c_valid[i * n + j] - c_reformat[i * n + j]) > 0.1f) {
                printf("mismatch at [%d][%d] c_valid=%f c_reformat=%f\n", i, j, c_valid[i * n + j], c_reformat[i * n + j]);
                count++;
            }
        }
    }

    printf("Verifying correctness of external sequential format...\n");    
    for(i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (fabs(c_valid[i * n + j] - c[i * n + j]) > 0.1f) {
                printf("mismatch at [%d][%d] c_valid=%f c=%f\n", i, j, c_valid[i * n + j], c[i * n + j]);
                count++;
            }
        }
    }

    if (count == 0) {
        printf("test passed\n");
        if (need_output) {
            fptr = fopen("answer.bin", "wb");  
            fwrite(c_reformat, sizeof(float), n * n, fptr);  
            fclose(fptr);
            fptr = fopen("answer_block.bin", "wb");  
            fwrite(c_block, sizeof(float), n * n, fptr);  
            fclose(fptr);
        }
    }

    free(a);
    free(b);
    free(c);
    free(c_valid);
    free(a_block);
    free(b_block);
    free(c_block);
    free(c_reformat);
    free(a_sub);
    free(b_sub);
    free(c_sub);
    return 0;
}