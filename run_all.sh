#!/usr/bin/env bash
# 10 个训练并行：{无域随机化, 域随机化} x 5 个种子，每个 3M 步
cd "$(dirname "$0")"; export PYTHONPATH=.
for s in 0 1 2 3 4; do
  for dr in 0 1; do
    name=$([ $dr = 1 ] && echo dr || echo nodr)
    .venv/bin/python -m walkercmd.train --dr $dr --seed $s --steps 3000000 --out runs/${name}_s$s > runs/${name}_s$s.out 2>&1 &
  done
done
wait; echo ALLDONE > runs/ALLDONE
