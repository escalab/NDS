#!/bin/bash
python3 parser.py fetch_ts.bin copy_in_B_ts.bin fetch_thread.json
python3 parser.py copy_in_C_ts.bin queue_ts.bin ttv_ts.bin copy_out_ts.bin main_thread.json
python3 merger.py main_thread.json fetch_thread.json