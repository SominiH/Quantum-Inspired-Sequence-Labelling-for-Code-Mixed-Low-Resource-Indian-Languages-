"""
dataset_preprocessor.py
======================
Handles data loading, cleaning, label mapping, and subword-to-word alignment
for code-mixed token-level Language Identification (LID).
"""

import re
from typing import Dict, List, Tuple, Any
from datasets import load_dataset, Dataset, DatasetDict
from transformers import PreTrainedTokenizerFast


class CodeMixedPreprocessor:
    """
    Preprocessor for Code-Mixed Sequence Labelling datasets.
    Supports subword tokenization and label alignment for XLM-RoBERTa.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerFast, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 6-Class Multilingual schema: Hindi, English, Bengali, Gujarati, Marathi, Other
        self.label2id = {"hi": 0, "en": 1, "bn": 2, "gu": 3, "mr": 4, "ot": 5}
        self.id2label = {0: "hi", 1: "en", 2: "bn", 3: "gu", 4: "mr", 5: "ot"}

    def add_language_label(self, label: str) -> int:
        """Dynamically add new language labels (e.g. 'bn', 'mr', 'te') if needed."""
        if label not in self.label2id:
            new_id = len(self.label2id)
            self.label2id[label] = new_id
            self.id2label[new_id] = label
        return self.label2id[label]

    def clean_text(self, text: str) -> str:
        """Standardize spaces and clean invisible control characters while preserving casing."""
        if not isinstance(text, str):
            return ""
        # Remove consecutive whitespaces
        cleaned = re.sub(r"\s+", " ", text).strip()
        return cleaned

    def parse_entry(self, entry: Dict[str, Any], token_col: str, label_col: str) -> Tuple[List[str], List[str]]:
        """Extract and clean word-level tokens and their respective labels from raw dataset record."""
        import ast
        raw_toks = entry.get(token_col, "")
        raw_lbls = entry.get(label_col, "")

        # Check if raw_lbls is a string representation of [{'key': ..., 'value': ...}]
        if isinstance(raw_lbls, (str, list)):
            try:
                items = ast.literal_eval(raw_lbls) if isinstance(raw_lbls, str) else raw_lbls
                if isinstance(items, list) and len(items) > 0 and isinstance(items[0], dict) and "key" in items[0]:
                    words = [str(it["key"]).strip() for it in items if "key" in it and "value" in it]
                    labels = [str(it["value"]).strip().lower() for it in items if "key" in it and "value" in it]
                    return words, labels
            except Exception:
                pass

        tokens = raw_toks.split() if isinstance(raw_toks, str) else list(raw_toks)
        labels = raw_lbls.split() if isinstance(raw_lbls, str) else list(raw_lbls)

        # Standardize labels to lowercase
        labels = [str(l).strip().lower() for l in labels]
        tokens = [str(t).strip() for t in tokens]

        # Filter out empty pairs
        clean_toks, clean_lbls = [], []
        for t, l in zip(tokens, labels):
            if t:
                clean_toks.append(t)
                mapped_l = l if l in self.label2id else "ot"
                clean_lbls.append(mapped_l)

        return clean_toks, clean_lbls

    def tokenize_and_align_labels(self, examples: Dict[str, List[Any]], token_col: str, label_col: str) -> Dict[str, Any]:
        """
        Tokenizes word sequences using XLM-RoBERTa's SentencePiece subword tokenizer
        and aligns word labels to subwords using word_ids masking (-100 for non-first subwords).
        """
        batch_words = []
        batch_labels = []

        for i in range(len(examples[token_col])):
            raw_entry = {token_col: examples[token_col][i], label_col: examples[label_col][i]}
            words, labels = self.parse_entry(raw_entry, token_col, label_col)
            if len(words) == 0:
                words = ["[EMPTY]"]
                labels = ["ot"]
            batch_words.append(words)
            batch_labels.append(labels)

        # Tokenize with is_split_into_words=True
        tokenized_inputs = self.tokenizer(
            batch_words,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors=None,
        )

        all_aligned_labels = []
        for i, labels in enumerate(batch_labels):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            aligned_labels = []
            previous_word_idx = None

            for word_idx in word_ids:
                if word_idx is None:
                    # Special tokens (<s>, </s>, <pad>) are masked from loss calculation
                    aligned_labels.append(-100)
                elif word_idx != previous_word_idx:
                    # First subword of a word receives the true label
                    label_str = labels[word_idx] if word_idx < len(labels) else "ot"
                    aligned_labels.append(self.label2id.get(label_str, self.label2id["ot"]))
                else:
                    # Subsequent subwords of the same word receive -100
                    aligned_labels.append(-100)
                previous_word_idx = word_idx

            all_aligned_labels.append(aligned_labels)

        tokenized_inputs["labels"] = all_aligned_labels
        return tokenized_inputs


def load_comi_lingua_splits(num_train: int = 10000, num_test: int = 2000) -> Tuple[Dataset, Dataset, str, str]:
    """
    Loads and slices COMI-LINGUA dataset into train and test partitions.
    Returns (train_dataset, test_dataset, token_col, label_col).
    """
    print("Loading COMI-LINGUA 'LID' configuration from Hugging Face...")
    ds = load_dataset("LingoIITGN/COMI-LINGUA", "LID")

    train_data = ds["train"]
    test_data = ds["test"]

    # Detect column names
    available_cols = train_data.column_names
    token_col = "Sentences" if "Sentences" in available_cols else available_cols[0]
    label_col = "Annotated by: Annotator 1" if "Annotated by: Annotator 1" in available_cols else (
        "Predicted tags" if "Predicted tags" in available_cols else available_cols[1]
    )

    # Select representative subset for reproducible training
    actual_train = min(num_train, len(train_data))
    actual_test = min(num_test, len(test_data))

    train_subset = train_data.select(range(actual_train))
    test_subset = test_data.select(range(actual_test))

    print(f"Loaded train subset: {len(train_subset):,} sentences")
    print(f"Loaded test subset : {len(test_subset):,} sentences")
    return train_subset, test_subset, token_col, label_col
