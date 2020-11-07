#!/bin/bash
python3 parser.py row_fetch_ts.bin col_fetch_ts.bin copy_in_ts.bin fetch_thread.json
python3 parser.py queue_ts.bin gemm_ts.bin copy_update_ts.bin copy_out_ts.bin row_fetch_ts.bin main_thread.json
python3 merger.py main_thread.json fetch_thread.json