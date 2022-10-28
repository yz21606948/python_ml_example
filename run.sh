#!/bin/sh

python3 /workdir/download.py
python3 /workdir/train.py
python3 /workdir/upload.py
