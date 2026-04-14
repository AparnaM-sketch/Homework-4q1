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

 What is the goal?
        The task is to train a neural network that learns the probability distribution over the next character given a sequence of previous characters. After training, the model can generate new text character by character, sampling from its predictions.

Why character‑level?
        Character‑level models work on any text without needing a predefined word vocabulary. They can handle misspellings, novel words, and even languages without word boundaries. The downside is longer sequences and less semantic meaning per token, but for small‑scale learning they are ideal.

Data preparation
        First, I create a small toy corpus (e.g., "hello hello world ...") and later can switch to a larger public‑domain text file. I build a mapping from each unique character to an integer index (vocabulary size ~20‑50). The entire text is converted into a list of integers.

Sequence creation – teacher forcing
        For training, I slide a fixed‑length window (e.g., 50 characters) across the text. For each window, the input is the first 49 characters and the target is the next 49 characters (shifted by one). This is teacher forcing: during training, the model always receives the true previous character as input, not its own prediction. This speeds up convergence but introduces “exposure bias” – the model never learns to recover from its own mistakes.

Model architecture
        Embedding layer: maps each character index to a dense vector of size 64. This allows the model to learn similarities between characters (e.g., vowels might cluster).
        
        LSTM layer (2 layers, hidden size 128): processes the sequence step by step, maintaining a hidden state that acts as memory. LSTMs were chosen over vanilla RNNs because they handle long‑range dependencies better via gating mechanisms (forget, input, output gates).
        
        Linear layer: maps the LSTM’s output (hidden size 128) to the vocabulary size (e.g., 20) to produce logits for each possible next character.
        
        Cross‑entropy loss: compares the predicted probability distribution with the true next character (one‑hot encoded implicitly).

Training loop
        For each batch of 64 sequences, I run a forward pass through the LSTM, compute the loss, backpropagate, and update weights using Adam optimizer. Gradient clipping (max norm 5) prevents exploding gradients. I track both training and validation loss (on a separate 10% hold‑out) to monitor overfitting.

Sampling with temperature
        After training, to generate new text I give the model a starting string (e.g., "hello"). Then, repeatedly:
        
        Feed the current sequence into the model.
        
        Get logits for the next character.
        
        Divide logits by a temperature parameter τ and apply softmax.
        
        Sample the next character from this probability distribution.
        
        Append it to the sequence and repeat.

Temperature effect:

        τ < 1 (e.g., 0.7): sharpens probabilities → high‑confidence, repetitive, “safe” text.
        
        τ = 1: original probabilities → balanced diversity.
        
        τ > 1 (e.g., 1.2): flattens probabilities → more randomness, creative but sometimes nonsensical.

Reflection on hyperparameters
        Sequence length: longer windows force the model to capture long‑range dependencies (e.g., matching opening and closing quotes). However, it requires more data and longer training; with short toy text, too long a window may leave no training examples.
        
        Hidden size: larger size increases capacity, allowing the model to memorise more patterns. On a tiny corpus, this leads to overfitting (training loss goes down, validation loss stagnates or rises).
        
        Temperature: directly controls the trade‑off between quality and diversity – a crucial concept for generative models.


🚀 Requirements

        - Python 3.8+
        - PyTorch
        - NumPy
        - Matplotlib

📝 Install dependencies:
        ```bash
        pip install torch numpy matplotlib

📝 Dataset
          Toy corpus: A small custom string: "hello hello world this is a tiny character rnn model hello help hello world hello hello hello"
          
          Optional larger corpus: Any plain text file (50–200 KB) – uncomment the file reading lines.

📝 How to Run
         Run the script:
        
        bash
         python char_rnn_lm.py

📝 The script will:

        - Build character vocabulary
        
        - Split data into train/validation sequences (length 50)
        
        - Train an LSTM (embedding size 64, hidden size 128, 2 layers)
        
        - Display training/validation loss curves
        
        - Generate 200 characters with temperatures 0.7, 1.0, and 1.2
        
        - Print a reflection on hyperparameters

📈 Results
        Loss curves show decreasing cross‑entropy over 15 epochs.
        
        Temperature 0.7: Repetitive but coherent output (e.g., "hello hello hello world").
        
        Temperature 1.0: Balanced, natural‑looking text.
        
        Temperature 1.2: More creative, occasional non‑words.

📝 Reflection
        Increasing sequence length forces longer‑term dependencies but requires more data.
        
        Larger hidden size improves capacity but risks overfitting on small corpora.
        
        Temperature controls diversity vs. confidence; low τ yields “safe” repetition, high τ yields surprise.
        
        Teacher forcing accelerates training but causes exposure bias (model never sees its own mistakes).
