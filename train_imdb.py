"""
Train RNN Sentiment Model on IMDB (Memory-Efficient)
=====================================================
Saves BERT embeddings in chunks to avoid OOM on 16GB RAM.

Run:
    tmux new -s train
    conda activate cs224n-gpu
    python train_imdb.py
    Ctrl+B, then D
"""

import os
import time
import gc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
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
CHUNK_SIZE = 2500  # save embeddings in chunks of 2500 (~1GB each)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# =============================================================================
# 1. PRECOMPUTE BERT EMBEDDINGS IN CHUNKS
# =============================================================================

def precompute_embeddings(texts, labels, split_name="train"):
    """Save embeddings in small chunk files to avoid OOM."""
    chunk_dir = f"data/{split_name}_chunks"
    done_flag = f"data/{split_name}_done.flag"

    # Already computed?
    if os.path.exists(done_flag):
        num_chunks = len([f for f in os.listdir(chunk_dir) if f.startswith("emb_")])
        print(f"Found {num_chunks} precomputed {split_name} chunks.")
        return

    print(f"Precomputing {split_name} BERT embeddings ({len(texts)} samples)...")
    os.makedirs(chunk_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert = AutoModel.from_pretrained("bert-base-uncased").to(DEVICE)
    bert.eval()

    chunk_embs = []
    chunk_labs = []
    chunk_idx = 0
    start = time.time()

    with torch.no_grad():
        for i, text in enumerate(texts):
            tokens = tokenizer(
                text, truncation=True, padding="max_length",
                max_length=SEQ_LENGTH, return_tensors="pt"
            ).to(DEVICE)
            emb = bert(**tokens).last_hidden_state.squeeze(0).cpu()
            chunk_embs.append(emb)
            chunk_labs.append(labels[i])

            # Save chunk to disk and free memory
            if len(chunk_embs) == CHUNK_SIZE:
                torch.save(torch.stack(chunk_embs), f"{chunk_dir}/emb_{chunk_idx:03d}.pt")
                torch.save(torch.tensor(chunk_labs, dtype=torch.float), f"{chunk_dir}/lab_{chunk_idx:03d}.pt")
                print(f"  Saved chunk {chunk_idx} ({(chunk_idx+1)*CHUNK_SIZE}/{len(texts)})")
                chunk_embs = []
                chunk_labs = []
                chunk_idx += 1
                gc.collect()

            if i % 500 == 0 and i > 0:
                elapsed = time.time() - start
                rate = i / elapsed
                eta = (len(texts) - i) / rate
                print(f"  {i}/{len(texts)}  ({rate:.0f} samples/sec, ETA: {eta/60:.1f} min)")

    # Save remaining samples
    if chunk_embs:
        torch.save(torch.stack(chunk_embs), f"{chunk_dir}/emb_{chunk_idx:03d}.pt")
        torch.save(torch.tensor(chunk_labs, dtype=torch.float), f"{chunk_dir}/lab_{chunk_idx:03d}.pt")
        print(f"  Saved chunk {chunk_idx} (final)")
        chunk_idx += 1

    # Mark as done
    with open(done_flag, "w") as f:
        f.write(f"chunks={chunk_idx}\n")

    del bert, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    print(f"  Done! {chunk_idx} chunks, took {(time.time()-start)/60:.1f} min")


# =============================================================================
# 2. CHUNKED DATASET (loads one chunk at a time)
# =============================================================================

class ChunkedDataset(Dataset):
    """Loads embeddings from chunk files on-the-fly. Memory friendly."""

    def __init__(self, split_name):
        chunk_dir = f"data/{split_name}_chunks"
        self.emb_files = sorted([f"{chunk_dir}/{f}" for f in os.listdir(chunk_dir) if f.startswith("emb_")])
        self.lab_files = sorted([f"{chunk_dir}/{f}" for f in os.listdir(chunk_dir) if f.startswith("lab_")])

        # Load all labels (small) to get total length
        self.all_labels = torch.cat([torch.load(f, map_location="cpu") for f in self.lab_files])
        self.total = len(self.all_labels)

        # Figure out chunk boundaries
        self.chunk_sizes = []
        for f in self.lab_files:
            self.chunk_sizes.append(len(torch.load(f, map_location="cpu")))

        # Current loaded chunk
        self.loaded_chunk_idx = -1
        self.loaded_embs = None

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        # Find which chunk this index belongs to
        cumsum = 0
        for chunk_i, size in enumerate(self.chunk_sizes):
            if idx < cumsum + size:
                local_idx = idx - cumsum
                # Load chunk if not already loaded
                if chunk_i != self.loaded_chunk_idx:
                    self.loaded_embs = torch.load(self.emb_files[chunk_i], map_location="cpu")
                    self.loaded_chunk_idx = chunk_i
                return self.loaded_embs[local_idx], self.all_labels[idx]
            cumsum += size


# =============================================================================
# 3. MODEL
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
# 4. TRAINING
# =============================================================================

def train():
    # --- Load IMDB dataset ---
    print("Loading IMDB dataset...")
    from datasets import load_dataset
    ds = load_dataset("imdb")

    # --- Precompute embeddings in chunks ---
    precompute_embeddings(ds["train"]["text"], ds["train"]["label"], "train")
    precompute_embeddings(ds["test"]["text"], ds["test"]["label"], "test")

    # Free the raw text from memory
    del ds
    gc.collect()

    # --- DataLoaders ---
    train_dataset = ChunkedDataset("train")
    test_dataset = ChunkedDataset("test")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- Model ---
    model = SentimentRNN(input_size=768, hidden_size=HIDDEN_SIZE).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {total_params:,} parameters")
    print(f"Train: {len(train_dataset)} | Test: {len(test_dataset)}")
    print(f"Epochs: {NUM_EPOCHS} | Batch: {BATCH_SIZE} | LR: {LEARNING_RATE}")
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
            preds = (logits.squeeze() > 0).float()
            correct += (preds == lab).sum().item()
            total += lab.size(0)

            step = epoch * len(train_loader) + batch_idx
            writer.add_scalar("Loss/batch", loss.item(), step)

            if batch_idx % 100 == 0:
                print(f"  Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        # --- Epoch stats ---
        avg_loss = total_loss / len(train_loader)
        train_acc = correct / total
        elapsed = time.time() - start

        writer.add_scalar("Loss/epoch", avg_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)

        # --- Evaluate ---
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

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "checkpoints/rnn_imdb_best.pt")
            print(f"  ★ New best! (acc: {best_acc:.4f})")

        print()

    # --- Save final ---
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
    print(f"{'='*60}")


if __name__ == "__main__":
    train()