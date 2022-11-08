#!/bin/sh

nohup poetry run ray start --address=$RAY_HEAD_ADDRESS:$RAY_HEAD_PORT --num-cpus=$RAY_NUM_CPU --block &
oauth2-proxy --email-domain=* --scope "user:email" 