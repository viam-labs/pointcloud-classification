#!/usr/bin/env bash
export PATH=$PATH:$HOME/.local/bin

uv run pyinstaller --onefile --collect-all vosk --hidden-import pyaudio --paths src src/main.py
tar -czvf dist/archive.tar.gz meta.json ./dist/main
