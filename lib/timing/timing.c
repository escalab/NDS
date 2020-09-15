#include "timing.h"

#include <sys/time.h>

struct timing_info {
	struct timeval starting_timeval;
	struct timeval *starts;
    struct timeval *ends;
	size_t count;
};


size_t timing_info_duration(struct timing_info *info) {
    size_t i, duration = 0;
    for (i = 0; i < info->count; i++) {
        duration += ((info->ends[i].tv_sec * 1000000 + info->ends[i].tv_usec) - 
            (info->starts[i].tv_sec * 1000000 + info->starts[i].tv_usec));
    }

    return duration;
}
			
void timing_info_push_end(struct timing_info *info) {
    struct timeval time;
    gettimeofday(&time, NULL);
    timing_info_push_end_with_timeval(info, &time);
}

void timing_info_push_end_with_timeval(struct timing_info *info, struct timeval *time) {
    memcpy(&info->ends[info->count], time, sizeof(struct timeval));
    info->count++;
}

void timing_info_push_start(struct timing_info *info) {
    struct timeval time;
    gettimeofday(&time, NULL);
    timing_info_push_start_with_timeval(info, &time);
}

void timing_info_push_start_with_timeval(struct timing_info *info, struct timeval *time) {
    memcpy(&info->starts[info->count], time, sizeof(struct timeval));
}

void timing_info_set_starting_time(struct timing_info *info) {
    struct timeval time;
    gettimeofday(&time, NULL);
    timing_info_set_starting_time_with_timeval(info, &time);
}

void timing_info_set_starting_time_with_timeval(struct timing_info *info, struct timeval *starting_timeval) {
    memcpy(&info->starting_timeval, starting_timeval, sizeof(struct timeval));
}

void timing_info_free(struct timing_info *info) {
    free(info->ends);
    free(info->starts);
    free(info);
}

struct timing_info *timing_info_new(size_t entries) {
    struct timing_info *new_info = calloc(1, sizeof(struct timing_info));
    if (new_info == NULL) {
        return NULL;
    }
    new_info->starts = calloc(entries, sizeof(struct timeval));
    if (new_info->starts == NULL) {
        free(new_info);
        return NULL;
    }

    new_info->ends = calloc(entries, sizeof(struct timeval));
    if (new_info->ends == NULL) {
        free(new_info->starts);
        free(new_info);
        return NULL;
    }

    return new_info;
}