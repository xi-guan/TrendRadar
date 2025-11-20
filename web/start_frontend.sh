#!/bin/bash
cd "$(dirname "$0")/static"
python3 -m http.server 3006
