#!/bin/bash
# RLM-style: Uses RLM executor with KB dict for chunked retrieval
docker build -t rolma . && docker run \
    --env-file .env \
    -v "$(pwd)/knowledge_base:/knowledge_base" \
    rolma python app_rlm.py
