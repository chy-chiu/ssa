#!/usr/bin/env bash
set -euo pipefail

.venv/bin/python scripts/smoke/smoke_words_task.py \
  --secrets-path assets/secrets.yaml \
  --model-name gpt-5.4-cc \
  --output logs/smoke_words/smoke_words_0.log
