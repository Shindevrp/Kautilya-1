#!/bin/bash
cd "$(dirname "$0")"
export PYTHONPATH="${PYTHONPATH}:."
python -m uvicorn api_server:app --host 0.0.0.0 --port 8000
