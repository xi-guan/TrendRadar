#!/bin/bash
cd "$(dirname "$0")/.."
uv run uvicorn web.api:app --host 0.0.0.0 --port 8007 --reload
