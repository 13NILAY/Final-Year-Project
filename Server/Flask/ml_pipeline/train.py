"""
Training Module
===============
Fine-tune RoBERTa for ESG metric classification.
Multi-head architecture with Environmental/Social/Governance heads.
Optimized for NVIDIA MX550 (2GB VRAM) with small batch sizes.
"""

import os
import json
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, get_linear_schedule_with_warmup
from typing import List, Dict, Tuple, Optional
from sklearn.utils.class_weight import compute_class_weight

from .extractor import ESGMetricClassifier
from .labeling import METRIC_TO_ID, ID_TO_METRIC, NUM_CLASSES, load_dataset

# ─── CUDA WARNING ───────────────────────────────────────────────────────────

# if not torch.cuda.is_available():
#     print("\n" + "!" * 60)
#     print("WARNING: CUDA is not available. Training will run on CPU, which is very slow.")
#     print("Please install PyTorch with CUDA support:")
#     print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
#     print("(Adjust cu124 to match your driver's CUDA version, check with nvidia-smi)")
#     print("!" * 60 + "\n")

# ─── DATASET CLASS ───────────────────────────────────────────────────────────

class ESGDataset(Dataset):
    """PyTorch Dataset for ESG metric classification."""

    def __init__(self, samples: List[Dict], tokenizer, max_length: int = 512):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample['text']
        label = METRIC_TO_ID.get(sample['metric_name'], METRIC_TO_ID['no_metric'])

        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long),
        }


# ─── TRAINING FUNCTIONS ─────────────────────────────────────────────────────

def compute_weights(samples: List[Dict]) -> torch.Tensor:
    """Compute class weights to handle label imbalance."""
    labels = [METRIC_TO_ID.get(s['metric_name'], METRIC_TO_ID['no_metric']) for s in samples]
    unique_labels = sorted(set(labels))

    weights = compute_class_weight(
        class_weight='balanced',
        classes=np.array(unique_labels),
        y=np.array(labels),
    )

    # Create full weight tensor (for all classes)
    full_weights = torch.ones(NUM_CLASSES)
    for ul, w in zip(unique_labels, weights):
        full_weights[ul] = w

    return full_weights


def train_model(
    train_data: List[Dict],
    val_data: List[Dict],
    output_dir: str = "models",
    epochs: int = 20,
    batch_size: int = 4,       # Small for 2GB VRAM
    learning_rate: float = 1e-5,
    warmup_ratio: float = 0.1,
    patience: int = 5,
    max_length: int = 512,
    gradient_accumulation: int = 4,  # Effective batch = 4 * 4 = 16
    device: Optional[str] = None,
) -> Dict:
    """
    Fine-tune RoBERTa for ESG metric classification.
    
    Optimized for NVIDIA MX550 (2GB VRAM):
    - batch_size=2-4, max_length=512
    - gradient_accumulation=4-8 (effective batch=16)
    - Mixed precision training (float16)
    
    Changes from v1.0:
    - RoBERTa (from DistilBERT) for better financial language understanding
    - Lower learning rate (1e-5 from 2e-5) for better convergence
    - More epochs (15 from 10) with higher patience (4 from 3)
    - Max length increased to 512 for full context
    
    Args:
        train_data: List of training samples
        val_data: List of validation samples
        output_dir: Directory to save model checkpoints
        epochs: Max training epochs (default: 15)
        batch_size: Batch size (keep small for low VRAM)
        learning_rate: Learning rate for AdamW (default: 1e-5)
        warmup_ratio: Fraction of steps for warmup
        patience: Early stopping patience (default: 4)
        max_length: Max token sequence length (default: 512)
        gradient_accumulation: Steps before optimizer update
        device: 'cuda' or 'cpu'
        
    Returns:
        Dict with training history and best metrics
    """
    # Setup device
    if device is None:
        device = torch.device('cpu')
    else:
        device = torch.device(device)
    print(f"[Training] Device: {device}")

    # Check CUDA memory and auto-adjust for RoBERTa
    if device.type == 'cuda':
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[Training] GPU Memory: {total_mem:.1f} GB")
        if total_mem < 3.0:
            print("[Training] Low VRAM detected — using conservative settings for RoBERTa")
            batch_size = min(batch_size, 2)
            gradient_accumulation = max(gradient_accumulation, 8)  # Keep effective batch = 16
            max_length = min(max_length, 384)
            print(f"[Training] Adjusted: batch={batch_size}, grad_accum={gradient_accumulation}, max_len={max_length}")

    # Setup tokenizer and datasets
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    train_dataset = ESGDataset(train_data, tokenizer, max_length)
    val_dataset = ESGDataset(val_data, tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"[Training] Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    print(f"[Training] Batch: {batch_size}, Max length: {max_length}, "
          f"Grad accum: {gradient_accumulation}")
    print(f"[Training] Effective batch size: {batch_size * gradient_accumulation}")

    # Model
    model = ESGMetricClassifier(num_classes=NUM_CLASSES)
    model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Training] Model: RoBERTa + Multi-Head Classifier")
    print(f"[Training] Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Class weights for imbalanced data
    class_weights = compute_weights(train_data).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer with differential learning rates
    # Lower LR for RoBERTa layers, higher for classifier heads
    optimizer = torch.optim.AdamW([
        {'params': model.bert.parameters(), 'lr': learning_rate},
        {'params': model.env_head.parameters(), 'lr': learning_rate * 10},
        {'params': model.soc_head.parameters(), 'lr': learning_rate * 10},
        {'params': model.gov_head.parameters(), 'lr': learning_rate * 10},
        {'params': model.no_metric_head.parameters(), 'lr': learning_rate * 10},
    ], weight_decay=0.01)

    # Scheduler
    total_steps = (len(train_loader) // gradient_accumulation) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Mixed precision scaler (for CUDA)
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    # Training loop
    best_val_f1 = 0.0
    best_epoch = 0
    no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'val_f1': [], 'val_accuracy': []}

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[Training] Starting training for {epochs} epochs...")
    print(f"[Training] Learning rate: {learning_rate}, Patience: {patience}")
    print("-" * 60)

    for epoch in range(epochs):
        epoch_start = time.time()

        # ── TRAIN ──
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            if use_amp:
                with torch.amp.autocast('cuda'):
                    logits = model(input_ids, attention_mask)
                    loss = criterion(logits, labels) / gradient_accumulation
                scaler.scale(loss).backward()
            else:
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels) / gradient_accumulation
                loss.backward()

            train_loss += loss.item() * gradient_accumulation

            # Predictions
            preds = torch.argmax(logits, dim=-1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            # Gradient accumulation step
            if (step + 1) % gradient_accumulation == 0 or (step + 1) == len(train_loader):
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()

        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / max(train_total, 1) * 100

        # ── VALIDATE ──
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                if use_amp:
                    with torch.amp.autocast('cuda'):
                        logits = model(input_ids, attention_mask)
                        loss = criterion(logits, labels)
                else:
                    logits = model(input_ids, attention_mask)
                    loss = criterion(logits, labels)

                val_loss += loss.item()
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / max(len(val_loader), 1)

        # Calculate metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        val_acc = np.mean(all_preds == all_labels) * 100

        # Macro F1
        val_f1 = _compute_macro_f1(all_preds, all_labels)

        elapsed = time.time() - epoch_start

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_f1'].append(val_f1)
        history['val_accuracy'].append(val_acc)

        print(f"  Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.1f}% | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.1f}% | "
              f"Val F1: {val_f1:.4f} | Time: {elapsed:.1f}s")

        # ── EARLY STOPPING ──
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            no_improve = 0

            # Save best model
            save_path = os.path.join(output_dir, "best_model.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'metrics': {
                    'val_f1': val_f1,
                    'val_accuracy': val_acc,
                    'val_loss': avg_val_loss,
                },
                'label_map': METRIC_TO_ID,
                'num_classes': NUM_CLASSES,
                'model_type': 'roberta-base',
                'architecture': 'multi-head',
            }, save_path)
            print(f"    ✓ Best model saved (F1: {val_f1:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\n[Training] Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break

    print("-" * 60)
    print(f"[Training] Best model: epoch {best_epoch}, Val F1: {best_val_f1:.4f}")
    print(f"[Training] Model saved to: {os.path.join(output_dir, 'best_model.pt')}")

    # Save training history
    history_path = os.path.join(output_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    return {
        'best_epoch': best_epoch,
        'best_val_f1': best_val_f1,
        'history': history,
        'model_path': os.path.join(output_dir, "best_model.pt"),
    }


def _compute_macro_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    """Compute macro F1 score across all classes."""
    unique_labels = set(labels.tolist()) | set(preds.tolist())
    f1s = []

    for label in unique_labels:
        tp = np.sum((preds == label) & (labels == label))
        fp = np.sum((preds == label) & (labels != label))
        fn = np.sum((preds != label) & (labels == label))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1s.append(f1)

    return np.mean(f1s) if f1s else 0.0


def save_model(model: ESGMetricClassifier, path: str, metrics: Optional[Dict] = None):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'metrics': metrics or {},
        'label_map': METRIC_TO_ID,
        'num_classes': NUM_CLASSES,
        'model_type': 'roberta-base',
        'architecture': 'multi-head',
    }, path)
    print(f"[Training] Model saved to {path}")


def load_model(path: str, device: Optional[str] = None) -> ESGMetricClassifier:
    """Load a trained model."""
    if device is None:
        device = torch.device('cpu')
    else:
        device = torch.device(device)

    model = ESGMetricClassifier(num_classes=NUM_CLASSES)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"[Training] Model loaded from {path}")
    if 'metrics' in checkpoint:
        print(f"[Training] Metrics: {checkpoint['metrics']}")
    if 'model_type' in checkpoint:
        print(f"[Training] Model type: {checkpoint['model_type']}")

    return model


if __name__ == "__main__":
    import sys

    # Default paths
    pipeline_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(pipeline_dir, "data", "esg_labeled.jsonl")
    output_dir = os.path.join(pipeline_dir, "models")

    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]

    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset not found at {dataset_path}")
        print("Run labeling.py first to generate the dataset.")
        sys.exit(1)

    print("=" * 60)
    print("ESG METRIC CLASSIFIER - Training (RoBERTa + Multi-Head)")
    print("=" * 60)

    train_data, val_data, test_data = load_dataset(dataset_path)

    if len(train_data) < 10:
        print(f"WARNING: Only {len(train_data)} training samples. Need more data for reliable training.")

    results = train_model(
        train_data=train_data,
        val_data=val_data,
        output_dir=output_dir,
        epochs=20,
        batch_size=4,
        learning_rate=1e-5,
        patience=5,
        max_length=512,
    )

    print(f"\nTraining complete!")
    print(f"Best F1: {results['best_val_f1']:.4f} at epoch {results['best_epoch']}")
