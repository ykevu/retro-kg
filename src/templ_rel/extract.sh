#!/bin/bash

export DATA_NAME="USPTO-50k"
export NUM_CORES=32

  python extract_templates.py \
  --model_name="template_relevance" \
  --data_name="$DATA_NAME" \
  --log_file="template_relevance_preprocess_$DATA_NAME" \
  --train_file=./data/raw_train.csv \
  --val_file=./data/raw_val.csv \
  --test_file=./data/raw_test.csv \
  --processed_data_path=./data/processed \
  --num_cores="$NUM_CORES" \
  --min_freq=1 \
  --seed=42
