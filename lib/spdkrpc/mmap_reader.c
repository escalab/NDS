#include "spdkrpc.h"

struct MmapReader {
    uint64_t _BASE_ADDR;
    uint32_t _BLOCK_SIZE;
    char *mmap_data;
    unsigned long mmap_size;
    unsigned int total_block_index;
    unsigned int curr_block_index;
    unsigned long *addr_list;
    unsigned long file_size;
    unsigned long curr_offset_in_block;
    unsigned int size_in_last_block;
};

char *mmap_readall(struct MmapReader *reader, char *buf) {
    char *data, *copy_ptr;
    unsigned long addr;

    if (buf == NULL) {
        data = (char *) malloc(reader->file_size);
    } else {
        data = buf;
    }

    copy_ptr = data;

    // read not the last block
    for (; reader->curr_block_index < reader->total_block_index; reader->curr_block_index++, copy_ptr += reader->_BLOCK_SIZE) {
        addr = reader->addr_list[reader->curr_block_index] - reader->_BASE_ADDR;
        memcpy(copy_ptr, (reader->mmap_data + addr), reader->_BLOCK_SIZE);
    }

    // last block
    DEBUG_PRINT("last block\n");
    addr = reader->addr_list[reader->curr_block_index] - reader->_BASE_ADDR;
    memcpy(copy_ptr, (reader->mmap_data + addr), reader->size_in_last_block);
    reader->curr_block_index++;
    return data;
}

void destroy_mmap_reader(struct MmapReader *reader) {
    if (reader->mmap_data) {
        munmap(reader->mmap_data, reader->mmap_size);
    }
    if (reader->addr_list) {
        free(reader->addr_list);
    }
}

void construct_mmap_reader(struct MmapReader *reader, int pid, unsigned long file_size, int parts) {
    char spdk_mmap_filename[64];
    int fd;
    struct stat sb;
    sprintf(spdk_mmap_filename, "/dev/hugepages/spdk_pid%dmap_0", pid);
    DEBUG_PRINT("going to map %s\n", spdk_mmap_filename);
    fd = open(spdk_mmap_filename, O_RDONLY);
    fstat(fd, &sb);
    reader->mmap_size = sb.st_size;
    reader->mmap_data = mmap(NULL, reader->mmap_size, PROT_READ, MAP_SHARED, fd, 0);
    reader->_BASE_ADDR = 0x200000200000;
    reader->_BLOCK_SIZE = BLOCK_SIZE;
    reader->total_block_index = parts - 1;
    reader->addr_list = malloc(parts * sizeof(unsigned long));
    reader->file_size = file_size;
    reader->size_in_last_block = file_size % reader->_BLOCK_SIZE;
    if (reader->size_in_last_block == 0) {
        reader->size_in_last_block = reader->_BLOCK_SIZE;
    }
    close(fd);
}

char *read_from_spdk(int pid, unsigned long file_size, int parts, unsigned long *addr_list, char *buf) {
    int i;
    struct MmapReader reader;
    char *data;
    memset(&reader, 0, sizeof(struct MmapReader));
    
    // creating a memory map reader to access SPDK's memory region
    construct_mmap_reader(&reader, pid, file_size, parts);

    // assign addresses got from SPDK's RPC respond
    for (i = 0; i < parts; i++) {
        reader.addr_list[i] = addr_list[i];
    }

    data = mmap_readall(&reader, buf);
    destroy_mmap_reader(&reader);
    return data;
}