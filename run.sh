#!/usr/bin/env bash
python3 ./main.py toplayer --ckpt ./best.pth --ex deep
python3 ./hand.py
python3 ./vote.py
