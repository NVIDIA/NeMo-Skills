#!/bin/bash
# Test script for TTS generation using Riva NIM on cluster
CONTAINER=nvcr.io/nvstaging/nim/magpie-tts-multilingual:1.3.0-34013444
SERVER_ARGS="--nim-tags-selector batch_size=32 --nim-disable-model-download false"
HF_HOME=?
INPUT_FILE=/workspace/tts.jsonl
OUTPUT_DIR=/workspace/tts
WORKSPACE=?

MOUNTS=$HF_HOME:$HF_HOME,$WORKSPACE:/workspace

ns generate \
    --cluster dfw \
    --model tts_nim \
    --generation_module recipes.asr_tts.riva_generate \
    --input_file "$INPUT_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --server_type generic \
    --num_chunks 1 \
    --server_entrypoint " DISABLE_RIVA_REALTIME_SERVER=True python3 -m nemo_skills.inference.server.serve_riva_nim" \
    --mount_paths "$MOUNTS" \
    --server_gpus 1 \
    --installation_command "pip install nvidia-riva-client==2.21.1" \
    --partition interactive \
    --server_container "$CONTAINER" \
    --server_args "$SERVER_ARGS" \
    ++generation_type=tts \
    ++tts_output_dir="$OUTPUT_DIR/audio_outputs" \
    ++voice='Magpie-Multilingual.EN-US.Mia' 
