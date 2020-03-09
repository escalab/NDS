#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>

int main(int argc, char** argv) {
    int i, j, n;
    FILE *fptr;
    long *tmp;
    
    if (argc < 2) {
        printf("usage: %s <path> <matrix size>\n", argv[0]);
        return 1;
    }

    fptr = fopen(argv[1], "r");
    n = atoi(argv[2]);

    tmp = malloc(sizeof(long));

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            fread(tmp, sizeof(long), 1, fptr);
            printf("%ld ", *tmp);
        }
        printf("\n");
    }

    free(tmp);
    fclose(fptr);
    return 0;
}