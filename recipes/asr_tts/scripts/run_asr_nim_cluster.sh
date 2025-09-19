#!/bin/bash
# Test script for ASR transcription using Riva NIM on cluster

CONTAINER=nvcr.io/nim/nvidia/parakeet-tdt-0.6b-v2:1.0.0
SERVER_ARGS='--CONTAINER_ID parakeet-tdt-0.6b-v2 --NIM_TAGS_SELECTOR name=parakeet-tdt-0.6b-v2,mode=ofl --nim-disable-model-download false'
HF_HOME=?
INPUT_FILE=/workspace/asr.jsonl
OUTPUT_DIR=/workspace/asr_a1
WORKSPACE=?

MOUNTS="$HF_HOME:$HF_HOME,$WORKSPACE:/workspace"

ns generate \
    --cluster dfw \
    --model asr_nim \
    --generation_module recipes.asr_tts.riva_generate \
    --input_file "$INPUT_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --server_type generic \
    --server_entrypoint "python3 -m nemo_skills.inference.server.serve_riva_nim" \
    --mount_paths "$MOUNTS" \
    --server_gpus 1 \
    --installation_command "pip install nvidia-riva-client==2.21.1" \
    --num_chunks 1 \
    --partition interactive \
    --server_container "$CONTAINER" \
    --server_args "$SERVER_ARGS" \
    ++generation_type=asr \
    ++language_code='en-US' \
    ++automatic_punctuation=true \
    ++speaker_diarization=false

