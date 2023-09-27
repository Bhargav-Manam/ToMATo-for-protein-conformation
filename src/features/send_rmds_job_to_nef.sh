#!/bin/bash

#OAR -l /nodes=1/core=64,walltime=18:00:00

date > timing.txt
# Load any necessary modules
module load conda/2021.11-python3.9

# Activate the virtual environment
source ../../myenv/bin/activate

# Run the Python script
python3 build_features.py -n2500

# Deactivate the virtual environment
deactivate

date >> timing.txt