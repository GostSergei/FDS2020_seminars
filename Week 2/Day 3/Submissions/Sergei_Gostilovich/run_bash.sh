#!/bin/bash

# parameters
LOAD_SCRIPT="load_data.py"
MAIN_SCRIPT="main.py"


# main:

echo RUN SCRIPT started

python3 $LOAD_SCRIPT
python3 $MAIN_SCRIPT

echo RUN SCRIPT ended