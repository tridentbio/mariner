#!/bin/sh

nohup poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 80 &
oauth2-proxy --email-domain=* --scope "user:email" 


