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

cJSON *create_get_tensorstore_gather_matrix_param(int id, int x, int y, int sub_m) {
    cJSON *param = create_get_tensorstore_matrix_param(id, x, y);
    cJSON *sub_m_json = NULL;
    
    sub_m_json = cJSON_CreateNumber(sub_m);
    if (sub_m_json == NULL) {
        return param;
    }
    cJSON_AddItemToObject(param, "sub_m", sub_m_json);
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

char *create_tensorstore_get_gather_matrix_json_string(struct JSONRPCClient* client, int id, int x, int y, int sub_m, char* rpc_method) {
    cJSON *params = create_get_tensorstore_gather_matrix_param(id, x, y, sub_m);

    cJSON *request = create_rpc_json_object(client, rpc_method, params);
    char *request_string = cJSON_PrintUnformatted(request);
    
    cJSON_Delete(request);

    return request_string;
}

char *create_tensorstore_get_matrix_json_string(struct JSONRPCClient* client, int id, int x, int y) {
    cJSON *params = create_get_tensorstore_matrix_param(id, x, y);
    cJSON *request = create_rpc_json_object(client, "get_tensorstore_matrix", params);
    char *request_string = cJSON_PrintUnformatted(request);
    
    cJSON_Delete(request);

    return request_string;
}

size_t tensorstore_get_matrix_return_size(const char* respond_string) {
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