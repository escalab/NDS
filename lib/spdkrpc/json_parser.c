#include <cjson/cJSON.h>

#include "spdkrpc.h"


cJSON *create_get_tensorstore_matrix_param(int id, int x, int y) {
    cJSON *param = cJSON_CreateObject();
    cJSON *id_json = NULL;
    cJSON *x_json = NULL;
    cJSON *y_json = NULL;

    id_json = cJSON_CreateNumber(id);
    if (id_json == NULL) {
        return param;
    }

    cJSON_AddItemToObject(param, "id", id_json);

    x_json = cJSON_CreateNumber(x);
    if (x_json == NULL) {
        return param;
    }
    cJSON_AddItemToObject(param, "x", x_json);

    y_json = cJSON_CreateNumber(y);
    if (y_json == NULL) {
        return param;
    }
    cJSON_AddItemToObject(param, "y", y_json);

    return param;
}

cJSON *create_rpc_json_object(struct JSONRPCClient *client, char *method_string, cJSON *params) {
    cJSON *request_json = cJSON_CreateObject();
    cJSON *jsonrpc;
    cJSON *method;

    (client->request_id)++;

    jsonrpc = cJSON_CreateString("2.0");
    if (jsonrpc == NULL) {
        return request_json;
    }

    cJSON_AddItemToObject(request_json, "jsonrpc", jsonrpc);


    method = cJSON_CreateString(method_string);
    if (method == NULL) {
        return request_json;
    }

    cJSON_AddItemToObject(request_json, "method", method);
    
    if ((cJSON_AddNumberToObject(request_json, "id", client->request_id)) == NULL) {
        return request_json;
    }

    if (params) {
        cJSON_AddItemToObject(request_json, "params", params);
    }

    return request_json;
}

char *create_get_tensorstore_matrix_json_string(struct JSONRPCClient* client, int id, int x, int y) {
    cJSON *params = create_get_tensorstore_matrix_param(id, x, y);
    cJSON *request = create_rpc_json_object(client, "get_tensorstore_matrix", params);
    char *request_string = cJSON_PrintUnformatted(request);
    
    cJSON_Delete(request);

    return request_string;
}

char *parse_get_tensorstore_matrix_json(const char* respond_string, int pid) {
    cJSON *object_json = NULL;
    const cJSON *result = NULL;
    const cJSON *addrs = NULL;
    const cJSON *addr = NULL;
    const cJSON *size = NULL;

    size_t file_size, parts = 0;

    unsigned long *data_map_addr;
    char *data;

    struct timeval g_start, g_end;
    uint64_t g_duration;
    
	gettimeofday(&g_start, NULL);
    object_json = cJSON_Parse(respond_string);
    if (object_json == NULL)
    {
        const char *error_ptr = cJSON_GetErrorPtr();
        if (error_ptr != NULL)
        {
            fprintf(stderr, "Error before: %s\n", error_ptr);
        }
        cJSON_Delete(object_json);
        return NULL;
    }

    // deconstruct the JSON object
    result = cJSON_GetObjectItemCaseSensitive(object_json, "result");
    addrs = cJSON_GetObjectItemCaseSensitive(result, "addr");
    size = cJSON_GetObjectItemCaseSensitive(result, "size");

    // get the basic info of fetched data
    file_size = (unsigned long) size->valuedouble;
    parts = (file_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    data_map_addr = (unsigned long *) calloc(parts, sizeof(unsigned long));

    cJSON_ArrayForEach(addr, addrs) {
        data_map_addr[atoi(addr->string)] = (unsigned long) addr->valuedouble;
    }

    gettimeofday(&g_end, NULL);
    g_duration = ((g_end.tv_sec * 1000000 + g_end.tv_usec) - (g_start.tv_sec * 1000000 + g_start.tv_usec));
    printf("parse string to JSON elapsed time: %f s\n", (double) g_duration / 1000000);

	gettimeofday(&g_start, NULL);
    data = read_from_spdk(pid, file_size, parts, data_map_addr, NULL);
    gettimeofday(&g_end, NULL);
    g_duration = ((g_end.tv_sec * 1000000 + g_end.tv_usec) - (g_start.tv_sec * 1000000 + g_start.tv_usec));
    printf("copy data from SPDK time: %f s\n", (double) g_duration / 1000000);
    // DEBUG_PRINT("parts: %d\n", parts);
    // DEBUG_PRINT("uint64_t _BASE_ADDR: %lx\n", reader->_BASE_ADDR);
    // DEBUG_PRINT("uint32_t_BLOCK_SIZE: %u\n", reader->_BLOCK_SIZE);
    // DEBUG_PRINT("char *mmap_data: %p\n", reader->mmap_data);
    // DEBUG_PRINT("unsigned int total_block_index: %u\n", reader->total_block_index);
    // DEBUG_PRINT("unsigned int curr_block_index: %u\n", reader->curr_block_index);
    // DEBUG_PRINT("unsigned long *addr_list: %p\n", reader->addr_list);
    // DEBUG_PRINT("unsigned long file_size %lu\n", reader->file_size);
    // DEBUG_PRINT("unsigned long curr_offset_in_block: %lu\n", reader->curr_offset_in_block);
    // DEBUG_PRINT("unsigned int size_in_last_block: %u\n", reader->size_in_last_block);

    cJSON_Delete(object_json);
    free(data_map_addr);
    return data;
}

size_t get_tensorstore_matrix_return_size(const char* respond_string) {
    cJSON *object_json = NULL;
    const cJSON *result = NULL;
    const cJSON *success = NULL;
    const cJSON *size = NULL;

    object_json = cJSON_Parse(respond_string);
    if (object_json == NULL)
    {
        const char *error_ptr = cJSON_GetErrorPtr();
        if (error_ptr != NULL)
        {
            fprintf(stderr, "Error before: %s\n", error_ptr);
        }
        cJSON_Delete(object_json);
        return 0;
    }
    result = cJSON_GetObjectItemCaseSensitive(object_json, "result");
    success = cJSON_GetObjectItemCaseSensitive(result, "success");
    size = cJSON_GetObjectItemCaseSensitive(result, "size");
    if (success->valueint != 1) {
        fprintf(stderr, "SPDK RPC call is failed\n");
        cJSON_Delete(object_json);
        return 0;

    }
    return (size_t) size->valuedouble;
}