#!/bin/bash

# setup the baseline environment
./setup_baseline.sh

# run baseline
./run_baseline.sh

# setup the software NDS environment
./setup_software_nds.sh

# run software NDS
./run_software_nds.sh

# setup the hardware NDS environment
./setup_hardware_nds.sh

# run hardware NDS
./run_hardware_nds.sh

# analyze results
python3 results.py
