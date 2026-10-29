"""
evaluator.py
============
Evaluation module for sequence labelling models.
Computes token-level accuracy, precision, recall, macro-F1, per-class breakdown,
confusion matrix, and inference latency.
"""

import time
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score


class SequenceLabellingEvaluator:
    """
    Evaluator for token-level language identification models.
    Properly handles subword masking (-100) during metric computation.
    """

    def __init__(self, id2label: Dict[int, str]):
        self.id2label = id2label
        self.label_names = [id2label[i] for i in sorted(id2label.keys())]

    def evaluate_model(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
    ) -> Dict[str, Any]:
        """
        Runs evaluation on a test dataloader and measures latency, accuracy, and macro-F1.
        """
        model.eval()
        all_preds = []
        all_true = []
        latencies = []

        total_tokens = 0
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                # Measure inference latency
                start_t = time.perf_counter()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = (time.perf_counter() - start_t) * 1000  # ms
                latencies.append(elapsed / input_ids.shape[0])

                logits = outputs["logits"]
                preds = torch.argmax(logits, dim=-1)

                # Flatten and mask out -100
                mask = labels != -100
                valid_preds = preds[mask].cpu().numpy()
                valid_labels = labels[mask].cpu().numpy()

                all_preds.extend(valid_preds)
                all_true.extend(valid_labels)
                total_tokens += len(valid_labels)

        # Calculate metrics
        all_preds = np.array(all_preds)
        all_true = np.array(all_true)

        acc = accuracy_score(all_true, all_preds)
        macro_f1 = f1_score(all_true, all_preds, average="macro", zero_division=0)
        weighted_f1 = f1_score(all_true, all_preds, average="weighted", zero_division=0)

        # Classification report per class
        clf_dict = classification_report(
            all_true,
            all_preds,
            target_names=self.label_names,
            output_dict=True,
            zero_division=0,
        )

        cm = confusion_matrix(all_true, all_preds, labels=list(range(len(self.label_names))))
        avg_latency = float(np.mean(latencies))

        # Model parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        results = {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "per_class": clf_dict,
            "confusion_matrix": cm,
            "avg_latency_ms_per_sentence": avg_latency,
            "total_tokens_evaluated": total_tokens,
            "total_params": total_params,
            "trainable_params": trainable_params,
        }
        return results

    def print_comparison_table(
        self, baseline_results: Dict[str, Any], quantum_results: Dict[str, Any]
    ) -> pd.DataFrame:
        """Constructs an aligned comparison table between Baseline and Quantum-Inspired models."""
        comparison_data = [
            {
                "Metric": "Overall Token Accuracy (%)",
                "Baseline (XLM-R)": f"{baseline_results['accuracy'] * 100:.2f}%",
                "Proposed (XLM-R + QISL)": f"{quantum_results['accuracy'] * 100:.2f}%",
                "Delta": f"{(quantum_results['accuracy'] - baseline_results['accuracy']) * 100:+.2f}%",
            },
            {
                "Metric": "Macro-F1 Score",
                "Baseline (XLM-R)": f"{baseline_results['macro_f1']:.4f}",
                "Proposed (XLM-R + QISL)": f"{quantum_results['macro_f1']:.4f}",
                "Delta": f"{quantum_results['macro_f1'] - baseline_results['macro_f1']:+.4f}",
            },
            {
                "Metric": "Weighted-F1 Score",
                "Baseline (XLM-R)": f"{baseline_results['weighted_f1']:.4f}",
                "Proposed (XLM-R + QISL)": f"{quantum_results['weighted_f1']:.4f}",
                "Delta": f"{quantum_results['weighted_f1'] - baseline_results['weighted_f1']:+.4f}",
            },
            {
                "Metric": "Hindi (hi) F1",
                "Baseline (XLM-R)": f"{baseline_results['per_class']['hi']['f1-score']:.4f}",
                "Proposed (XLM-R + QISL)": f"{quantum_results['per_class']['hi']['f1-score']:.4f}",
                "Delta": f"{quantum_results['per_class']['hi']['f1-score'] - baseline_results['per_class']['hi']['f1-score']:+.4f}",
            },
            {
                "Metric": "English (en) F1",
                "Baseline (XLM-R)": f"{baseline_results['per_class']['en']['f1-score']:.4f}",
                "Proposed (XLM-R + QISL)": f"{quantum_results['per_class']['en']['f1-score']:.4f}",
                "Delta": f"{quantum_results['per_class']['en']['f1-score'] - baseline_results['per_class']['en']['f1-score']:+.4f}",
            },
            {
                "Metric": "Other/Symbols (ot) F1",
                "Baseline (XLM-R)": f"{baseline_results['per_class']['ot']['f1-score']:.4f}",
                "Proposed (XLM-R + QISL)": f"{quantum_results['per_class']['ot']['f1-score']:.4f}",
                "Delta": f"{quantum_results['per_class']['ot']['f1-score'] - baseline_results['per_class']['ot']['f1-score']:+.4f}",
            },
            {
                "Metric": "Inference Latency (ms/sentence)",
                "Baseline (XLM-R)": f"{baseline_results['avg_latency_ms_per_sentence']:.2f} ms",
                "Proposed (XLM-R + QISL)": f"{quantum_results['avg_latency_ms_per_sentence']:.2f} ms",
                "Delta": f"{quantum_results['avg_latency_ms_per_sentence'] - baseline_results['avg_latency_ms_per_sentence']:+.2f} ms",
            },
            {
                "Metric": "Trainable Parameters",
                "Baseline (XLM-R)": f"{baseline_results['trainable_params']:,}",
                "Proposed (XLM-R + QISL)": f"{quantum_results['trainable_params']:,}",
                "Delta": f"{quantum_results['trainable_params'] - baseline_results['trainable_params']:+,}",
            },
        ]
        df = pd.DataFrame(comparison_data)
        return df
