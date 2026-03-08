#!/bin/bash
cd /home/parksean210/project/speech_enhancement
timeout 280 .venv/bin/python main.py fit \
  --config configs/labnet_metricgan.yaml \
  --trainer.max_steps=100 \
  --trainer.check_val_every_n_epoch=999 \
  --trainer.limit_val_batches=0 2>&1 | tail -6
