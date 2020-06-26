#include "spdkrpc.h"

#include <sys/stat.h>
#include <fcntl.h>

int main(int argc, char **argv) {
    // SPDK RPC part
    char *original_data;
    int fd, rc;
    uint64_t i, j, k, checked_byte, offset, file_size;
    struct stat st;
    int matrix_size, submatrix_size, x, y;
    
    char *hugepage_addr;
    struct JSONRPCClient client;
    size_t return_size; 

    if (argc < 6) {
        printf("usage: %s <verify file path> <matrix size> <submatrix size> <id> <x> <y>\n", argv[0]);
        exit(1);
    }

    matrix_size = atoi(argv[2]);
    submatrix_size = atoi(argv[3]);
    x = atoi(argv[4]);
    y = atoi(argv[5]);

    rc = connect_to_spdkrpc_server(&client);
    if (rc) {
        printf("cannot create conntection to SPDK RPC server");
        return rc;
    }

    hugepage_addr = mmap_to_tensorstore_hugepage();

#ifdef GATHER
    return_size = tensorstore_get_gather_submatrix(&client, 0, x, y, submatrix_size);
#else
    return_size = tensorstore_get_submatrix(&client, 0, x, y);
#endif
    // assume matrix is a square
    fd = open(argv[1], O_RDONLY);
    fstat(fd, &st);
	file_size = st.st_size;
    original_data = (char *) mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);

    file_size = file_size / ((matrix_size / submatrix_size) * (matrix_size / submatrix_size));

    checked_byte = 0;
    offset = 0;
    if (file_size == return_size) {
        printf("compare data\n");
        for (i = y * submatrix_size; i < (y+1) * submatrix_size; i++) {
            for (j = x * submatrix_size; j < (x+1) * submatrix_size; j++) {
                // how many bytes per element needs to be checked
                for (k = 0; k < sizeof(double); k++) {
                    offset = (i*matrix_size+j)*sizeof(double)+k;
                    // printf("i=%lu, j=%lu, offset=%lu\n", i, j, offset);
                    if (hugepage_addr[checked_byte] != original_data[offset]) {
                        printf("data is wrong at byte %lu, data=%x, original_data=%x\n", checked_byte, hugepage_addr[checked_byte], original_data[offset]);
                        rc = -1;
                        abort();
                    }
                    checked_byte++;
                }
            }
        }
    } else {
        printf("return_size (%lu) != file_size (%lu)\n", return_size, file_size);
        rc = -1;
    }

    if (rc) {
        printf("Test Failed\n");
    } else {
        printf("Test Passed\n");
    }

    munmap(original_data, file_size);
    munmap(hugepage_addr, HUGEPAGE_SZ);
    close(fd);
    return rc;
}