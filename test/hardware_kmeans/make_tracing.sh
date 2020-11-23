#!/bin/bash
python3 parser.py col_fetch_ts.bin copy_in_ts.bin fetch_thread.json
python3 parser.py queue_ts.bin kernel_ts.bin copy_out_ts.bin cluster_fetch_ts.bin cluster_assignment_ts.bin main_thread.json
python3 merger.py main_thread.json fetch_thread.json