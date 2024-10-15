FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04
WORKDIR /content
ENV PATH="/home/camenduru/.local/bin:${PATH}"
ENV OPENAI_BASE_URL=http://localhost:5000/v1/
ENV TOKENIZE_BASE_URL=http://localhost:3348
ENV TTS_BASE_URL=http://localhost:22311/v1/

RUN adduser --disabled-password --gecos '' camenduru && \
    adduser camenduru sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    chown -R camenduru:camenduru /content && \
    chmod -R 777 /content && \
    chown -R camenduru:camenduru /home && \
    chmod -R 777 /home && \
    apt update -y && add-apt-repository -y ppa:git-core/ppa && apt update -y && apt install -y aria2 git git-lfs unzip ffmpeg

USER camenduru

RUN pip install -q opencv-python imageio imageio-ffmpeg ffmpeg-python av runpod \
    torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121 torchtext==0.18.0 torchdata==0.8.0 --extra-index-url https://download.pytorch.org/whl/cu121 \
    xformers==0.0.27 whisperspeech matplotlib openai-whisper==20231117 git+https://github.com/homebrewltd/WhisperSpeech.git vector_quantize_pytorch bitsandbytes transformers accelerate webdataset && \
    git clone --recursive https://github.com/theroyallab/tabbyAPI /content/tabbyAPI && \
    git clone --recursive https://github.com/fishaudio/fish-speech /content/fish-speech && \
    git clone -b docker --recursive https://github.com/homebrewltd/ichigo-demo /content/ichigo-demo && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/janhq/llama3-s-instruct-v0.3-checkpoint-7000-phase-3-exllama2/raw/main/config.json -d /content/tabbyAPI/models/llama3s -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/janhq/llama3-s-instruct-v0.3-checkpoint-7000-phase-3-exllama2/raw/main/generation_config.json -d /content/tabbyAPI/models/llama3s -o generation_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/janhq/llama3-s-instruct-v0.3-checkpoint-7000-phase-3-exllama2/raw/main/loss_log.txt -d /content/tabbyAPI/models/llama3s -o loss_log.txt && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/janhq/llama3-s-instruct-v0.3-checkpoint-7000-phase-3-exllama2/raw/main/model.safetensors.index.json -d /content/tabbyAPI/models/llama3s -o model.safetensors.index.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/janhq/llama3-s-instruct-v0.3-checkpoint-7000-phase-3-exllama2/resolve/main/output.safetensors -d /content/tabbyAPI/models/llama3s -o output.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/janhq/llama3-s-instruct-v0.3-checkpoint-7000-phase-3-exllama2/raw/main/special_tokens_map.json -d /content/tabbyAPI/models/llama3s -o special_tokens_map.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/janhq/llama3-s-instruct-v0.3-checkpoint-7000-phase-3-exllama2/raw/main/tokenizer.json -d /content/tabbyAPI/models/llama3s -o tokenizer.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/janhq/llama3-s-instruct-v0.3-checkpoint-7000-phase-3-exllama2/raw/main/tokenizer_config.json -d /content/tabbyAPI/models/llama3s -o tokenizer_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/janhq/llama3-s-instruct-v0.3-checkpoint-7000-phase-3-exllama2/raw/main/training_config_phase3.yaml -d /content/tabbyAPI/models/llama3s -o training_config_phase3.yaml && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/fishaudio/fish-speech-1.4/raw/main/config.json -d /content/fish-speech/checkpoints/fish-speech-1.4 -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/fishaudio/fish-speech-1.4/resolve/main/firefly-gan-vq-fsq-8x1024-21hz-generator.pth -d /content/fish-speech/checkpoints/fish-speech-1.4 -o firefly-gan-vq-fsq-8x1024-21hz-generator.pth && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/fishaudio/fish-speech-1.4/resolve/main/model.pth -d /content/fish-speech/checkpoints/fish-speech-1.4 -o model.pth && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/fishaudio/fish-speech-1.4/raw/main/special_tokens_map.json -d /content/fish-speech/checkpoints/fish-speech-1.4 -o special_tokens_map.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/fishaudio/fish-speech-1.4/raw/main/tokenizer.json -d /content/fish-speech/checkpoints/fish-speech-1.4 -o tokenizer.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/fishaudio/fish-speech-1.4/raw/main/tokenizer_config.json -d /content/fish-speech/checkpoints/fish-speech-1.4 -o tokenizer_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://raw.githubusercontent.com/camenduru/ichigo-tost/refs/heads/main/tabbyAPI.yml -d /content/tabbyAPI -o config.yml && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://raw.githubusercontent.com/camenduru/ichigo-tost/refs/heads/main/fishSpeechAPI.py -d /content/fish-speech/tools -o api.py && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://raw.githubusercontent.com/camenduru/ichigo-tost/refs/heads/main/whisperAPI.py -d /content -o whisperAPI.py && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/jan-hq/WhisperVQ/resolve/main/whisper-vq-stoks-v3-7lang-fixed.model -d /content -o whisper-vq-stoks-v3-7lang-fixed.model

RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash && \
    export NVM_DIR="$HOME/.nvm" && \
    [ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh" && \
    nvm install 18.20.3 && \
    . "$NVM_DIR/nvm.sh" && npm install -g npm

RUN export NVM_DIR="$HOME/.nvm" && \
    [ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh" && \
    cd /content/ichigo-demo && npm install && npm run build

WORKDIR /content

ENTRYPOINT bash -c " \
    cd /content/tabbyAPI && python main.py & \
    cd /content/fish-speech/tools && uvicorn 'api:app' --workers 3 --host 0.0.0.0 --port 22311 & \
    python /content/whisperAPI.py & \
    cd /content/ichigo-demo && \
    OPENAI_BASE_URL=$OPENAI_BASE_URL TOKENIZE_BASE_URL=$TOKENIZE_BASE_URL TTS_BASE_URL=$TTS_BASE_URL npm start \
"