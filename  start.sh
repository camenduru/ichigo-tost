#!/bin/bash

export NVM_DIR="/root/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"

nvm use $NODE_VERSION

(cd /content/tabbyAPI && python main.py) &
(cd /content/fish-speech/tools && uvicorn 'api:app' --workers 3 --host 0.0.0.0 --port 22311) &
(cd /content && python whisperAPI.py) &
(cd /content/ichigo-demo && OPENAI_BASE_URL=$OPENAI_BASE_URL TOKENIZE_BASE_URL=$TOKENIZE_BASE_URL TTS_BASE_URL=$TTS_BASE_URL npm start)
