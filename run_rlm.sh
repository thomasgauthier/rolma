#!/bin/bash
docker build -t rolma --cache-from rolma . && docker run -it \
    --env-file .env \
    -v "$(pwd)/knowledge_base:/knowledge_base" \
    rolma python app_rlm.py
