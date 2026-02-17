"""
Train RNN Sentiment Model on IMDB (Ultra Memory-Efficient)
===========================================================
Loads ONE chunk at a time. Never more than ~1GB in RAM.

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
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter


# =============================================================================
# CONFIG
# =============================================================================

SEQ_LENGTH = 128
BATCH_SIZE = 32
HIDDEN_SIZE = 128
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
CHUNK_SIZE = 2500
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# =============================================================================
# 1. PRECOMPUTE BERT EMBEDDINGS IN CHUNKS (float16 to save memory)
# =============================================================================

def precompute_embeddings(texts, labels, split_name="train"):
    chunk_dir = f"data/{split_name}_chunks"
    done_flag = f"data/{split_name}_done.flag"

    if os.path.exists(done_flag):
        num_chunks = len([f for f in os.listdir(chunk_dir) if f.startswith("emb_")])
        print(f"Found {num_chunks} precomputed {split_name} chunks.")
        return

    print(f"Precomputing {split_name} BERT embeddings ({len(texts)} samples)...")
    os.makedirs(chunk_dir, exist_ok=True)

    from transformers import AutoTokenizer, AutoModel
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
            emb = bert(**tokens).last_hidden_state.squeeze(0).cpu().half()  # float16!
            chunk_embs.append(emb)
            chunk_labs.append(labels[i])

            if len(chunk_embs) == CHUNK_SIZE:
                torch.save(
                    torch.stack(chunk_embs),
                    f"{chunk_dir}/emb_{chunk_idx:03d}.pt"
                )
                torch.save(
                    torch.tensor(chunk_labs, dtype=torch.float),
                    f"{chunk_dir}/lab_{chunk_idx:03d}.pt"
                )
                print(f"  Saved chunk {chunk_idx} ({(chunk_idx+1)*CHUNK_SIZE}/{len(texts)})")
                chunk_embs = []
                chunk_labs = []
                chunk_idx += 1
                gc.collect()

            if i % 500 == 0 and i > 0:
                elapsed = time.time() - start
                rate = i / elapsed
                eta = (len(texts) - i) / rate
                print(f"  {i}/{len(texts)}  ({rate:.0f}/sec, ETA: {eta/60:.1f}min)")

    if chunk_embs:
        torch.save(torch.stack(chunk_embs), f"{chunk_dir}/emb_{chunk_idx:03d}.pt")
        torch.save(torch.tensor(chunk_labs, dtype=torch.float), f"{chunk_dir}/lab_{chunk_idx:03d}.pt")
        chunk_idx += 1

    with open(done_flag, "w") as f:
        f.write(f"chunks={chunk_idx}\n")

    del bert, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    print(f"  Done! {chunk_idx} chunks, {(time.time()-start)/60:.1f}min")


# =============================================================================
# 2. MODEL
# =============================================================================

class SentimentRNN(nn.Module):
    def __init__(self, input_size=768, hidden_size=128):
        super().__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, h_last = self.rnn(x)
        return self.fc(h_last.squeeze(0))


# =============================================================================
# 3. LOAD ONE CHUNK → DATALOADER
# =============================================================================

def load_chunk(split_name, chunk_idx):
    """Load a single chunk from disk. Returns DataLoader."""
    chunk_dir = f"data/{split_name}_chunks"
    emb = torch.load(f"{chunk_dir}/emb_{chunk_idx:03d}.pt", map_location="cpu").float()
    lab = torch.load(f"{chunk_dir}/lab_{chunk_idx:03d}.pt", map_location="cpu")
    shuffle = (split_name == "train")
    return DataLoader(TensorDataset(emb, lab), batch_size=BATCH_SIZE, shuffle=shuffle)


def count_chunks(split_name):
    chunk_dir = f"data/{split_name}_chunks"
    return len([f for f in os.listdir(chunk_dir) if f.startswith("emb_")])


# =============================================================================
# 4. TRAINING
# =============================================================================

def train():
    # --- Precompute if needed ---
    print("Loading IMDB dataset...")
    train_done = os.path.exists("data/train_done.flag")
    test_done = os.path.exists("data/test_done.flag")

    if not train_done or not test_done:
        from datasets import load_dataset
        ds = load_dataset("imdb")
        if not train_done:
            precompute_embeddings(ds["train"]["text"], ds["train"]["label"], "train")
        if not test_done:
            precompute_embeddings(ds["test"]["text"], ds["test"]["label"], "test")
        del ds
        gc.collect()

    n_train_chunks = count_chunks("train")
    n_test_chunks = count_chunks("test")
    print(f"Train: {n_train_chunks} chunks | Test: {n_test_chunks} chunks")

    # --- Model ---
    model = SentimentRNN(input_size=768, hidden_size=HIDDEN_SIZE).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"Epochs: {NUM_EPOCHS} | Batch: {BATCH_SIZE} | LR: {LEARNING_RATE}")
    print(f"{'='*60}\n")

    # --- TensorBoard ---
    writer = SummaryWriter("runs/rnn_imdb")
    os.makedirs("checkpoints", exist_ok=True)
    best_acc = 0.0
    global_step = 0

    # --- Training ---
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        start = time.time()

        # Train: iterate through chunks one at a time
        for chunk_i in range(n_train_chunks):
            loader = load_chunk("train", chunk_i)

            for emb, lab in loader:
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

                writer.add_scalar("Loss/batch", loss.item(), global_step)
                global_step += 1

            # Free chunk memory
            del loader
            gc.collect()

        avg_loss = total_loss / (total / BATCH_SIZE)
        train_acc = correct / total
        elapsed = time.time() - start

        writer.add_scalar("Loss/epoch", avg_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)

        # --- Evaluate: same chunk-by-chunk approach ---
        model.eval()
        test_correct = 0
        test_total = 0
        test_loss = 0

        with torch.no_grad():
            for chunk_i in range(n_test_chunks):
                loader = load_chunk("test", chunk_i)
                for emb, lab in loader:
                    emb, lab = emb.to(DEVICE), lab.to(DEVICE)
                    logits = model(emb)
                    loss = criterion(logits.squeeze(), lab)
                    test_loss += loss.item()
                    preds = (logits.squeeze() > 0).float()
                    test_correct += (preds == lab).sum().item()
                    test_total += lab.size(0)
                del loader
                gc.collect()

        test_acc = test_correct / test_total
        test_avg = test_loss / (test_total / BATCH_SIZE)

        writer.add_scalar("Loss/test", test_avg, epoch)
        writer.add_scalar("Accuracy/test", test_acc, epoch)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} ({elapsed:.1f}s)")
        print(f"  Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Test  Loss: {test_avg:.4f} | Test  Acc: {test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "checkpoints/rnn_imdb_best.pt")
            print(f"  ★ New best! ({best_acc:.4f})")
        print()

    torch.save({
        "epoch": NUM_EPOCHS,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_acc": best_acc,
    }, "checkpoints/rnn_imdb_final.pt")

    writer.close()
    print(f"{'='*60}")
    print(f"Done! Best test accuracy: {best_acc:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    train()