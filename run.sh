#!/bin/bash
# ROMA-style: Uses ReAct executor with terminal tool (grep) for KB retrieval
docker build -t rolma --cache-from rolma . && docker run -it \
    --env-file .env \
    -v "$(pwd)/knowledge_base:/knowledge_base" \
    rolma python app.py
