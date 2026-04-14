# Homework-4
# Q1: Character-Level RNN Language Model

👩‍🎓 Student Information

        Name: MOPARTHI APARNA
        Course: CS5760 Natural Language Processing
        Department: Computer Science & Cybersecurity
        Semester: Spring 2026
        Assignment: Homework 4, Question 1
  
📚 Overview

 This project implements a character-level recurrent neural network (RNN) using an LSTM. The model is trained to predict the next character given a sequence of previous characters. It demonstrates teacher forcing, temperature‑based sampling, and the effect of hyperparameters on text generation.


## Requirements
- Python 3.8+
- PyTorch
- NumPy
- Matplotlib

Install dependencies:
```bash
pip install torch numpy matplotlib

## Dataset
  Toy corpus: A small custom string: "hello hello world this is a tiny character rnn model hello help hello world hello hello hello"
  
  Optional larger corpus: Any plain text file (50–200 KB) – uncomment the file reading lines.

Usage
 Run the script:

bash
 python char_rnn_lm.py

The script will:

- Build character vocabulary

- Split data into train/validation sequences (length 50)

- Train an LSTM (embedding size 64, hidden size 128, 2 layers)

- Display training/validation loss curves

- Generate 200 characters with temperatures 0.7, 1.0, and 1.2

- Print a reflection on hyperparameters

Results
Loss curves show decreasing cross‑entropy over 15 epochs.

Temperature 0.7: Repetitive but coherent output (e.g., "hello hello hello world").

Temperature 1.0: Balanced, natural‑looking text.

Temperature 1.2: More creative, occasional non‑words.

Reflection
Increasing sequence length forces longer‑term dependencies but requires more data.

Larger hidden size improves capacity but risks overfitting on small corpora.

Temperature controls diversity vs. confidence; low τ yields “safe” repetition, high τ yields surprise.

Teacher forcing accelerates training but causes exposure bias (model never sees its own mistakes).
