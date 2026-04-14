import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1. Data preparation (toy text)
# -------------------------------
toy_text = "hello hello world this is a tiny character rnn model hello help hello world hello hello hello"
# For larger text, uncomment and provide a file path
# with open('your_text_file.txt', 'r', encoding='utf-8') as f:
#     toy_text = f.read()

chars = sorted(list(set(toy_text)))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for ch, i in char_to_idx.items()}
vocab_size = len(chars)

data = torch.tensor([char_to_idx[ch] for ch in toy_text], dtype=torch.long)

# Train/val split (90/10)
split = int(0.9 * len(data))
train_data = data[:split]
val_data = data[split:]

seq_length = 50  # sequence length

def create_sequences(data, seq_len):
    """Return (inputs, targets) tensors, or None if not enough data."""
    if len(data) <= seq_len:
        return None, None
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        xs.append(data[i:i+seq_len])
        ys.append(data[i+1:i+seq_len+1])
    return torch.stack(xs), torch.stack(ys)

train_x, train_y = create_sequences(train_data, seq_length)
val_x, val_y = create_sequences(val_data, seq_length)

# If validation set is too small, use a smaller seq_length or skip validation
if val_x is None:
    print(f"Validation data too short ({len(val_data)} chars) for seq_length={seq_length}. Reducing seq_length or using train-only.")
    # Option 1: reduce seq_length
    seq_length = max(5, len(val_data) - 2)
    train_x, train_y = create_sequences(train_data, seq_length)
    val_x, val_y = create_sequences(val_data, seq_length)
    if val_x is None:
        # Option 2: no validation, just use training
        val_x, val_y = train_x[:1], train_y[:1]  # dummy for loop
        print("No validation sequences; using a dummy batch.")

class CharDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

batch_size = 64
train_loader = DataLoader(CharDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(CharDataset(val_x, val_y), batch_size=batch_size, shuffle=False) if val_x is not None else None

# -------------------------------
# 2. Model (LSTM)
# -------------------------------
class CharRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden

embed_size = 64
hidden_size = 128
num_layers = 2
lr = 0.003
epochs = 15
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CharRNN(vocab_size, embed_size, hidden_size, num_layers).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# -------------------------------
# 3. Training with loss curves
# -------------------------------
train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        output, _ = model(xb)
        loss = criterion(output.view(-1, vocab_size), yb.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        total_loss += loss.item()
    train_losses.append(total_loss / len(train_loader))
    
    # Validation (if available)
    if val_loader is not None:
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                output, _ = model(xb)
                loss = criterion(output.view(-1, vocab_size), yb.view(-1))
                val_loss += loss.item()
        val_losses.append(val_loss / len(val_loader))
        print(f"Epoch {epoch+1}: train loss {train_losses[-1]:.4f}, val loss {val_losses[-1]:.4f}")
    else:
        val_losses.append(0)
        print(f"Epoch {epoch+1}: train loss {train_losses[-1]:.4f}")

# Plot losses
plt.plot(train_losses, label='Train')
if val_loader is not None:
    plt.plot(val_losses, label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curves')
plt.savefig('loss_curves.png')
plt.show()

# -------------------------------
# 4. Sampling with temperature
# -------------------------------
def sample(model, start_str, length, temperature=1.0):
    model.eval()
    input_ids = torch.tensor([char_to_idx.get(ch, 0) for ch in start_str], dtype=torch.long).unsqueeze(0).to(device)
    hidden = None
    generated = list(start_str)
    for _ in range(length):
        with torch.no_grad():
            output, hidden = model(input_ids, hidden)
        logits = output[0, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        next_idx = np.random.choice(vocab_size, p=probs)
        generated.append(idx_to_char[next_idx])
        input_ids = torch.tensor([[next_idx]], dtype=torch.long).to(device)
    return ''.join(generated)

print("\n--- Sampling with different temperatures ---")
for temp in [0.7, 1.0, 1.2]:
    gen = sample(model, "hello", 200, temperature=temp)
    print(f"\nTemperature {temp}:\n{gen}\n")

# -------------------------------
# 5. Reflection (printed)
# -------------------------------
print("""
Reflection:
- Sequence length: longer sequences capture more context but require more data and longer training.
- Hidden size: larger size increases capacity; on small data it may overfit.
- Temperature: lower (0.7) yields more confident, repetitive text; higher (1.2) yields more diverse but sometimes nonsensical text.
- Teacher forcing speeds up training but causes exposure bias (model never corrects its own errors during training).
""")