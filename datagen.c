#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>

int main(int argc, char** argv) {
    int i, n;
    FILE *fptr;
    double tmp;

    if (argc < 3) {
        printf("usage: %s <output path> <matrix size>\n", argv[0]);
        return 1;
    }

    srand(5);

    fptr = fopen(argv[1], "w+b");
    n = atoi(argv[2]);
    for (i = 0; i < n * n; i++) {
        tmp = (double) rand() / RAND_MAX;
        fwrite(&tmp, sizeof(double), 1, fptr);
    }

    fclose(fptr);
    return 0;
}