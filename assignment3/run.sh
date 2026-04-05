#!/usr/bin/env bash
set -e

echo "Step 1: Hyperparameter Search (TTT)"
python3 -u training/hyperparameter_search.py

echo "Step 2: Train & Evaluate TTT"
python3 -u training/train_ttt.py

echo "Step 3: Train & Evaluate Connect 4"
python3 -u training/train_connect4.py

echo "Step 4: Curriculum Training (TTT + C4)"
python3 -u training/curriculum_training.py --game both

echo "Step 5: Curriculum Evaluation"
python3 -u training/evaluate_curriculum.py --game both

echo "Step 6: P1 vs P2 Analysis"
python3 -u training/gen_p1p2_analysis.py

echo "Completed. Results in results/ and models/"
