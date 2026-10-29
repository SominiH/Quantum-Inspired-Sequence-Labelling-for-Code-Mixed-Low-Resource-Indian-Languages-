"""
train.py
========
Complete End-to-End Training and Comparative Evaluation Pipeline:
Baseline (XLM-RoBERTa) vs. Proposed (XLM-RoBERTa + Quantum-Inspired Sequence Labelling)
"""

import os
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
import pandas as pd
import numpy as np

# Import our custom modules
from src.dataset_preprocessor import CodeMixedPreprocessor, load_comi_lingua_splits
from src.model_baseline import XLMRobertaBaselineForLID
from src.model_quantum import XLMRobertaQuantumInspiredForLID
from src.evaluator import SequenceLabellingEvaluator


def set_seed(seed: int = 42):
    """Ensure experimental reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


class TorchSeqDataset(torch.utils.data.Dataset):
    """Converts tokenized Hugging Face dataset items into PyTorch tensors."""
    def __init__(self, tokenized_dict):
        self.input_ids = torch.tensor(tokenized_dict["input_ids"], dtype=torch.long)
        self.attention_mask = torch.tensor(tokenized_dict["attention_mask"], dtype=torch.long)
        self.labels = torch.tensor(tokenized_dict["labels"], dtype=torch.long)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def train_single_epoch(model, dataloader, optimizer, scheduler, device):
    """Executes a single training epoch and reports average token classification loss."""
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs["loss"]

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def run_pipeline():
    print("=" * 65)
    print("  QUANTUM-INSPIRED SEQUENCE LABELLING FOR INDIAN CODE-MIXING  ")
    print("=" * 65)

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on Device: {device}")
    if device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    # 1. Initialize XLM-RoBERTa Tokenizer
    model_name = "xlm-roberta-base"
    print(f"\n[Stage 1/5] Initializing Subword Tokenizer ({model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 2. Load COMI-LINGUA Dataset
    print("\n[Stage 2/5] Loading and Preparing COMI-LINGUA LID Dataset...")
    # Using 5,000 train instances and 1,000 test instances for fast, verified demonstration
    train_subset, test_subset, token_col, label_col = load_comi_lingua_splits(num_train=5000, num_test=1000)

    preprocessor = CodeMixedPreprocessor(tokenizer=tokenizer, max_length=64)
    print("Aligning subwords and mapping word labels...")
    train_encodings = preprocessor.tokenize_and_align_labels(train_subset, token_col, label_col)
    test_encodings = preprocessor.tokenize_and_align_labels(test_subset, token_col, label_col)

    train_dataset = TorchSeqDataset(train_encodings)
    test_dataset = TorchSeqDataset(test_encodings)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    evaluator = SequenceLabellingEvaluator(id2label=preprocessor.id2label)

    # 3. Train Baseline Model (XLM-RoBERTa + Linear Head)
    print("\n" + "-" * 55)
    print("[Stage 3/5] Initializing & Training BASELINE Model (XLM-RoBERTa)")
    print("-" * 55)
    baseline_model = XLMRobertaBaselineForLID(pretrained_model_name=model_name, num_labels=3).to(device)

    epochs = 2
    b_optimizer = torch.optim.AdamW(baseline_model.parameters(), lr=3e-5, weight_decay=0.01)
    b_total_steps = len(train_loader) * epochs
    b_scheduler = get_linear_schedule_with_warmup(b_optimizer, num_warmup_steps=int(0.1 * b_total_steps), num_training_steps=b_total_steps)

    for epoch in range(epochs):
        t0 = time.time()
        loss = train_single_epoch(baseline_model, train_loader, b_optimizer, b_scheduler, device)
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}/{epochs} | Training Loss: {loss:.4f} | Time: {elapsed:.1f}s")

    print("\nEvaluating Baseline on Held-out Test Set...")
    baseline_results = evaluator.evaluate_model(baseline_model, test_loader, device)
    print(f"Baseline Accuracy: {baseline_results['accuracy']*100:.2f}% | Macro-F1: {baseline_results['macro_f1']:.4f}")

    # 4. Train Proposed Quantum-Inspired Model (XLM-RoBERTa + QISL)
    print("\n" + "-" * 55)
    print("[Stage 4/5] Initializing & Training PROPOSED Model (XLM-RoBERTa + QISL)")
    print("-" * 55)
    quantum_model = XLMRobertaQuantumInspiredForLID(
        pretrained_model_name=model_name,
        num_labels=3,
        hilbert_dim=64,
        temperature=0.5,
    ).to(device)

    q_optimizer = torch.optim.AdamW(quantum_model.parameters(), lr=3e-5, weight_decay=0.01)
    q_total_steps = len(train_loader) * epochs
    q_scheduler = get_linear_schedule_with_warmup(q_optimizer, num_warmup_steps=int(0.1 * q_total_steps), num_training_steps=q_total_steps)

    for epoch in range(epochs):
        t0 = time.time()
        loss = train_single_epoch(quantum_model, train_loader, q_optimizer, q_scheduler, device)
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}/{epochs} | Training Loss: {loss:.4f} | Time: {elapsed:.1f}s")

    print("\nEvaluating Quantum-Inspired Model on Held-out Test Set...")
    quantum_results = evaluator.evaluate_model(quantum_model, test_loader, device)
    print(f"Quantum-Inspired Accuracy: {quantum_results['accuracy']*100:.2f}% | Macro-F1: {quantum_results['macro_f1']:.4f}")

    # 5. Comparative Evaluation Report & Checkpoint Export
    print("\n" + "=" * 65)
    print("                 EXPERIMENTAL RESULTS COMPARISON                 ")
    print("=" * 65)
    comp_df = evaluator.print_comparison_table(baseline_results, quantum_results)
    print(comp_df.to_string(index=False))

    # Save Checkpoints
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(baseline_model.state_dict(), "checkpoints/baseline_xlmr_lid.pt")
    torch.save(quantum_model.state_dict(), "checkpoints/quantum_xlmr_lid.pt")
    print("\nCheckpoints successfully saved to 'checkpoints/' directory.")

    # 6. Unseen Real-World Inference Test
    print("\n" + "=" * 65)
    print("         INFERENCE DEMONSTRATION ON UNSEEN CODE-MIXED TEXT       ")
    print("=" * 65)
    test_sentences = [
        "Mujhe today bahut khushi hai",
        "Yeh quantum inspired architecture high accuracy deliver karega",
        "Ami am very pleased with this result",
    ]

    for sent in test_sentences:
        words = sent.split()
        enc = tokenizer(words, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=64).to(device)
        quantum_model.eval()
        with torch.no_grad():
            out = quantum_model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            probs = torch.softmax(out["logits"][0], dim=-1)

        word_ids = enc.word_ids(batch_index=0)
        seen = set()
        print(f"\nSentence: '{sent}'")
        for idx, w_id in enumerate(word_ids):
            if w_id is not None and w_id not in seen:
                seen.add(w_id)
                tag_id = int(torch.argmax(probs[idx]).item())
                confidence = float(probs[idx][tag_id].item())
                print(f"  {words[w_id]:<16} -> {preprocessor.id2label[tag_id]} (Confidence: {confidence*100:.1f}%)")


if __name__ == "__main__":
    run_pipeline()
