"""
Train RNN Sentiment Model on IMDB
==================================
Run in tmux:
    tmux new -s train
    conda activate cs224n-gpu
    python train_imdb.py
    Ctrl+B, then D  (detach)
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModel


# =============================================================================
# CONFIG
# =============================================================================

SEQ_LENGTH = 128
BATCH_SIZE = 32
HIDDEN_SIZE = 128
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# =============================================================================
# 1. PRECOMPUTE BERT EMBEDDINGS
# =============================================================================

def precompute_embeddings(texts, labels, split_name="train"):
    emb_path = f"data/{split_name}_embeddings.pt"
    lab_path = f"data/{split_name}_labels.pt"

    if os.path.exists(emb_path):
        print(f"Loading precomputed {split_name} embeddings...")
        embeddings = torch.load(emb_path, map_location="cpu")
        labels_tensor = torch.load(lab_path, map_location="cpu")
        print(f"  Loaded: {embeddings.shape}")
        return embeddings, labels_tensor

    print(f"Precomputing {split_name} BERT embeddings ({len(texts)} samples)...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert = AutoModel.from_pretrained("bert-base-uncased").to(DEVICE)
    bert.eval()

    all_emb = []
    start = time.time()

    with torch.no_grad():
        for i, text in enumerate(texts):
            tokens = tokenizer(
                text, truncation=True, padding="max_length",
                max_length=SEQ_LENGTH, return_tensors="pt"
            ).to(DEVICE)
            emb = bert(**tokens).last_hidden_state.squeeze(0).cpu()
            all_emb.append(emb)

            if i % 500 == 0:
                elapsed = time.time() - start
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (len(texts) - i) / rate if rate > 0 else 0
                print(f"  {i}/{len(texts)}  ({rate:.0f} samples/sec, ETA: {eta/60:.1f} min)")

    embeddings = torch.stack(all_emb)
    labels_tensor = torch.tensor(labels, dtype=torch.float)

    os.makedirs("data", exist_ok=True)
    torch.save(embeddings, emb_path)
    torch.save(labels_tensor, lab_path)
    print(f"  Saved to {emb_path} ({embeddings.shape})")
    print(f"  Took {(time.time() - start)/60:.1f} min")

    del bert, tokenizer
    torch.cuda.empty_cache()

    return embeddings, labels_tensor


# =============================================================================
# 2. MODEL
# =============================================================================

class SentimentRNN(nn.Module):
    def __init__(self, input_size=768, hidden_size=128):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h_seq, h_last = self.rnn(x)
        out = h_last.squeeze(0)
        return self.fc(out)


# =============================================================================
# 3. TRAINING
# =============================================================================

def train():
    # --- Load IMDB dataset ---
    print("Loading IMDB dataset...")
    from datasets import load_dataset
    ds = load_dataset("imdb")

    # --- Precompute embeddings ---
    train_emb, train_lab = precompute_embeddings(
        ds["train"]["text"], ds["train"]["label"], "train"
    )
    test_emb, test_lab = precompute_embeddings(
        ds["test"]["text"], ds["test"]["label"], "test"
    )

    # --- DataLoaders ---
    train_loader = DataLoader(
        TensorDataset(train_emb, train_lab),
        batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(test_emb, test_lab),
        batch_size=BATCH_SIZE, shuffle=False
    )

    # --- Model ---
    model = SentimentRNN(input_size=768, hidden_size=HIDDEN_SIZE).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {total_params:,} parameters")
    print(f"Train: {len(train_emb)} samples, Test: {len(test_emb)} samples")
    print(f"Epochs: {NUM_EPOCHS}, Batch: {BATCH_SIZE}, LR: {LEARNING_RATE}")
    print(f"{'='*60}\n")

    # --- TensorBoard ---
    writer = SummaryWriter("runs/rnn_imdb")

    # --- Training loop ---
    best_acc = 0.0
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        start = time.time()

        for batch_idx, (emb, lab) in enumerate(train_loader):
            emb, lab = emb.to(DEVICE), lab.to(DEVICE)

            logits = model(emb)
            loss = criterion(logits.squeeze(), lab)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Track accuracy
            preds = (logits.squeeze() > 0).float()
            correct += (preds == lab).sum().item()
            total += lab.size(0)

            # Log batch loss
            step = epoch * len(train_loader) + batch_idx
            writer.add_scalar("Loss/batch", loss.item(), step)

            # Print progress every 100 batches
            if batch_idx % 100 == 0:
                print(f"  Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        # --- Epoch stats ---
        avg_loss = total_loss / len(train_loader)
        train_acc = correct / total
        elapsed = time.time() - start

        writer.add_scalar("Loss/epoch", avg_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)

        # --- Evaluate on test set ---
        model.eval()
        test_correct = 0
        test_total = 0
        test_loss = 0

        with torch.no_grad():
            for emb, lab in test_loader:
                emb, lab = emb.to(DEVICE), lab.to(DEVICE)
                logits = model(emb)
                loss = criterion(logits.squeeze(), lab)
                test_loss += loss.item()
                preds = (logits.squeeze() > 0).float()
                test_correct += (preds == lab).sum().item()
                test_total += lab.size(0)

        test_acc = test_correct / test_total
        test_avg_loss = test_loss / len(test_loader)

        writer.add_scalar("Loss/test", test_avg_loss, epoch)
        writer.add_scalar("Accuracy/test", test_acc, epoch)

        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} ({elapsed:.1f}s)")
        print(f"  Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Test  Loss: {test_avg_loss:.4f} | Test  Acc: {test_acc:.4f}")

        # --- Save best model ---
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "checkpoints/rnn_imdb_best.pt")
            print(f"  ★ New best model saved! (acc: {best_acc:.4f})")

        print()

    # --- Save final model ---
    torch.save({
        "epoch": NUM_EPOCHS,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_acc": best_acc,
    }, "checkpoints/rnn_imdb_final.pt")

    writer.close()

    print(f"{'='*60}")
    print(f"Training complete!")
    print(f"Best test accuracy: {best_acc:.4f}")
    print(f"Models saved in checkpoints/")
    print(f"TensorBoard logs in runs/")
    print(f"{'='*60}")


# =============================================================================
# 4. RUN
# =============================================================================

if __name__ == "__main__":
    train()
