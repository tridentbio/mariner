#!/bin/sh

poetry run ray start --num-gpus 1 --head --dashboard-port=8265 --port=6379 --dashboard-host=0.0.0.0 --block 
