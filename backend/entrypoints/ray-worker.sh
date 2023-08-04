#!/bin/sh
echo Connecting to $RAY_ADDRESS
poetry run ray start --address=$RAY_ADDRESS:6379 --num-cpus=$RAY_NUM_CPU --block 
