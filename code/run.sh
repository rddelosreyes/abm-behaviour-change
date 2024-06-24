#!/bin/bash

echo "Running a single reversal learning simulation"
python main.py config_single_run.yaml

echo "Performing ABC on single reversal learning experiments"
python main.py config_single_calibrate.yaml

echo "Performing ABC on serial reversal learning experiments"
python main.py config_serial_calibrate.yaml