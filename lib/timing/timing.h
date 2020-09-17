#include <stdlib.h>
#include <string.h>
#include <stdint.h>

struct timing_info;

struct timestamps {
    uint64_t count;
    uint64_t *timestamps;
};

void timing_info_free_timestamps(struct timestamps *tss);
struct timestamps *timing_info_get_timestamps(struct timing_info *info);
uint64_t timing_info_duration(struct timing_info *info);
void timing_info_push_end(struct timing_info *info);
void timing_info_push_end_with_timeval(struct timing_info *info, struct timeval *time);
void timing_info_push_start(struct timing_info *info);
void timing_info_push_start_with_timeval(struct timing_info *info, struct timeval *time);
void timing_info_set_starting_time(struct timing_info *info);
void timing_info_set_starting_time_with_timeval(struct timing_info *info, struct timeval *starting_timeval);
void timing_info_free(struct timing_info *info);
struct timing_info *timing_info_new(uint64_t entries);