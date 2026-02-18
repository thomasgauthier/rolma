#!/bin/bash
docker build -t rolma . && docker run \
    --env-file .env \
    -v "$(pwd)/knowledge_base:/knowledge_base" \
    rolma
