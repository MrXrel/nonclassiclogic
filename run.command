#!/bin/bash
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  candidate_python=""
  for executable in python3.12 python3.13 python3.14 python3; do
    if command -v "$executable" >/dev/null 2>&1 && "$executable" -c 'import sys; raise SystemExit(sys.version_info < (3,12))' 2>/dev/null; then
      candidate_python="$executable"
      break
    fi
  done
  if [ -z "$candidate_python" ]; then
    bundled_python="$HOME/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3"
    if [ -x "$bundled_python" ]; then candidate_python="$bundled_python"; fi
  fi
  if [ -z "$candidate_python" ]; then
    echo 'Установите Python 3.12 или новее с python.org и повторите запуск.'
    read -r -p 'Нажмите Enter для выхода.'
    exit 1
  fi
  "$candidate_python" -m venv .venv
fi
if ! cmp -s requirements.txt .venv/installed-requirements.txt; then
  .venv/bin/python -m pip install -r requirements.txt
  cp requirements.txt .venv/installed-requirements.txt
fi
exec .venv/bin/python manage.py start
