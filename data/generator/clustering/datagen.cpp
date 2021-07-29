#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>

#include <sys/mman.h>

/**
 * Initializes randomly data points.
 *
 * @param data        refence points
 * @param data_nb     number of reference points
 * @param dim        dimension of points
 */
void generate_data(double *data, uint64_t data_nb, uint64_t dim) 
{
    // Generate random data points
    for (uint64_t i=0; i < dim * data_nb; ++i) 
    {
        data[i] = 10. * (double)(rand() / (double)RAND_MAX);
    }
}

int generate_data_file(const char *filename, uint64_t data_nb, uint64_t dim)
{
    uint64_t dsize;
    int ret, fd;
    double *data;

    // generate query data
    dsize = sizeof(double) * data_nb * dim;
    fd = open(filename, O_RDWR | O_CREAT | O_TRUNC, 0644);
    ret = posix_fallocate(fd, 0, dsize);
    if (ret != 0) {
        printf("fallocate error code: %d\n", ret);
        return ret;
    }

    data = (double *) mmap(NULL, dsize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (data == NULL) {
        printf("failed to mmap\n");
        return -1;
    }
    generate_data(data, data_nb, dim);
    msync(data, dsize, MS_SYNC);
    munmap(data, dsize);
    close(fd);
    return 0;
}
/**
 * Create the synthetic data (reference and query points).
 */
int main(int argc, char** argv) {
    // Parameters
    uint64_t query_nb, dim;
    int rc;

    if (argc < 4) {
        printf("usage: %s <output path> <# of points> <# of attributes>\n", argv[0]);
    }

    // ref_nb = (uint64_t) atoll(argv[1]);
    query_nb = (uint64_t) atoll(argv[2]);
    dim = (uint64_t) atoll(argv[3]);

    // Display
    printf("PARAMETERS\n");
    // printf("- Number reference points : %lu\n",   ref_nb);
    printf("- Number query points     : %lu\n",   query_nb);
    printf("- Dimension of points     : %lu\n",   dim);

    // Initialize random number generator
    srand(5);

    // generate query data
    // rc = generate_data_file("query_data.bin", query_nb, dim);
    // if (rc)
    // {
    //     printf("failed to generate file\n");
    //     return rc;
    // }

    rc = generate_data_file(argv[1], query_nb, dim);
    if (rc)
    {
        printf("failed to generate file\n");
        return rc;
    }

    return EXIT_SUCCESS;
}
