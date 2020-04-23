#include "spdkrpc.h"

void free_mem(void *ptr) {
    if (ptr) {
        free(ptr);
    }
}