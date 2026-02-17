"""
Use Trained LSTM Sentiment Model
=================================
Run:
    python predict.py
    python predict.py --text "This movie was terrible"
    python predict.py --interactive
"""

import argparse
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


# =============================================================================
# 1. MODEL (same architecture as training)
# =============================================================================

class SentimentLSTM(nn.Module):
    def __init__(self, input_size=768, hidden_size=64, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        output, (h_last, c_last) = self.lstm(x)
        hidden = torch.cat([h_last[0], h_last[1]], dim=1)
        hidden = self.dropout(hidden)
        return self.fc(hidden)


# =============================================================================
# 2. LOAD EVERYTHING
# =============================================================================

def load_model(checkpoint_path="checkpoints/lstm_imdb_best.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load BERT (frozen, for embeddings)
    print("Loading BERT tokenizer + model...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert = AutoModel.from_pretrained("bert-base-uncased").to(device)
    bert.eval()

    # Load trained LSTM
    print("Loading trained LSTM...")
    model = SentimentLSTM(input_size=768, hidden_size=64, dropout=0.0).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    print(f"Ready! (device: {device})\n")
    return tokenizer, bert, model, device


# =============================================================================
# 3. PREDICT
# =============================================================================

@torch.no_grad()
def predict(text, tokenizer, bert, model, device, seq_length=128):
    """
    Full pipeline:
        text → tokenizer → BERT embeddings → LSTM → sentiment

    Returns:
        label: "POSITIVE" or "NEGATIVE"
        confidence: 0.0 to 1.0
    """

    # Step 1: Tokenize
    tokens = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=seq_length,
        return_tensors="pt"
    ).to(device)

    # Step 2: BERT embeddings
    embeddings = bert(**tokens).last_hidden_state  # (1, T, 768)

    # Step 3: LSTM prediction
    logit = model(embeddings)  # (1, 1)

    # Step 4: Convert to probability
    prob = torch.sigmoid(logit).item()
    label = "POSITIVE" if prob > 0.5 else "NEGATIVE"
    confidence = prob if prob > 0.5 else 1 - prob

    return label, confidence, prob


# =============================================================================
# 4. DEMO
# =============================================================================

def demo(tokenizer, bert, model, device):
    """Run on example reviews."""

    examples = [
        "This movie was absolutely fantastic! Great acting and beautiful cinematography.",
        "Terrible film. Waste of time. The plot made no sense and the acting was awful.",
        "It was okay. Nothing special but not bad either.",
        "One of the best movies I've ever seen. A masterpiece of storytelling.",
        "I fell asleep halfway through. Boring and predictable.",
        "The special effects were amazing but the story was weak.",
        "A heartwarming film that made me laugh and cry. Highly recommend!",
        "Worst movie of the year. Don't waste your money.",
    ]

    print("=" * 65)
    print(f"  {'REVIEW':<45} {'PRED':>10} {'CONF':>6}")
    print("=" * 65)

    for text in examples:
        label, confidence, prob = predict(text, tokenizer, bert, model, device)
        short = text[:42] + "..." if len(text) > 45 else text
        icon = "🟢" if label == "POSITIVE" else "🔴"
        print(f"  {short:<45} {icon} {label:>8} {confidence:>5.1%}")

    print("=" * 65)


def interactive(tokenizer, bert, model, device):
    """Chat with the model."""

    print("=" * 50)
    print("  IMDB Sentiment Analyzer")
    print("  Type a movie review, get a prediction.")
    print("  Type 'quit' to exit.")
    print("=" * 50)

    while True:
        text = input("\n📝 Review: ").strip()
        if text.lower() in ("quit", "exit", "q"):
            break
        if not text:
            continue

        label, confidence, prob = predict(text, tokenizer, bert, model, device)
        icon = "🟢" if label == "POSITIVE" else "🔴"
        bar_len = int(prob * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)

        print(f"\n  {icon} {label} ({confidence:.1%} confident)")
        print(f"  NEG [{bar}] POS")
        print(f"       {'▲':>{bar_len+1}}")
        print(f"       p = {prob:.4f}")


# =============================================================================
# 5. RUN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IMDB Sentiment Prediction")
    parser.add_argument("--text", type=str, help="Single text to classify")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--checkpoint", default="checkpoints/lstm_imdb_best.pt")
    args = parser.parse_args()

    tokenizer, bert, model, device = load_model(args.checkpoint)

    if args.text:
        label, confidence, prob = predict(args.text, tokenizer, bert, model, device)
        icon = "🟢" if label == "POSITIVE" else "🔴"
        print(f"{icon} {label} ({confidence:.1%})")

    elif args.interactive:
        interactive(tokenizer, bert, model, device)

    else:
        demo(tokenizer, bert, model, device)