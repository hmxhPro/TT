#!/bin/bash
# Backend startup script with offline mode for HuggingFace

# Force HuggingFace to use local cache only
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Silence FFmpeg / libav decoder chatter (SEI truncated, NAL warnings, etc.)
# These are non-fatal H.264 stream artifacts that flood stderr but do not
# affect detection. -8 = AV_LOG_QUIET.
export OPENCV_FFMPEG_LOGLEVEL=-8
export AV_LOG_FORCE_NOCOLOR=1

# Start the backend server
cd "$(dirname "$0")"
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
