import os
import numpy as np
from pathlib import Path
import json
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

SELECTED_FACE_DIM = 92
TWO_HANDS_DIM = 42

class TextTokenizer:
    def __init__(self, name: str = "t5-small", max_length: int = 128):
        
        self.tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
        self.max_length = max_length
        if self.tokenizer.bos_token is None:
            self.tokenizer.add_special_tokens({"bos_token": "<bos>"})
        self.pad_id = self.tokenizer.pad_token_id
        self.eos_id = self.tokenizer.eos_token_id
        self.bos_id = self.tokenizer.bos_token_id
        self.vocab_size = len(self.tokenizer)

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
        )
        

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

def mask_nan(landmarks_xyz: np.ndarray):
    """
    seq_xyz: (T, N, 3)
    """
    mask = np.isfinite(landmarks_xyz).all(axis=-1)      # (T, N)
    clean = np.nan_to_num(landmarks_xyz, nan=0.0).astype(np.float32)
    present = mask.any(axis=-1)                   # (T,)
    return clean, mask, present


class GestureTextDataset(Dataset):
    def __init__(self, csv_path, landmark_dir, tokenizer, max_frames=None):
        self.df = pd.read_csv(csv_path , sep="\t")
        self.df.columns = self.df.columns.str.strip().str.lower()
        self.landmark_dir = Path(landmark_dir)
        self.tokenizer = tokenizer
        self.max_frames = max_frames
        valid_rows = []

        for i in range(len(self.df)):
            row = self.df.iloc[i]

            video_name = row["sentence_name"]
            stem = Path(video_name).stem

            npz_path = self.landmark_dir / f"{stem}.npz"

            if npz_path.exists():
                valid_rows.append(i)

        self.df = self.df.iloc[valid_rows].reset_index(drop=True)

        

    def __len__(self):
        return len(self.df)

    def _load_npz(self, path):
        data = np.load(path)

        face = data["face"].astype(np.float32)             # (T, Nf, 3)
        hands = data["hands"].astype(np.float32)   # (T, Nh, 3)
        
        return face, hands

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_name = row["sentence_name"]
        sentence = row["sentence"]
        #print(video_name)
        stem = Path(video_name).stem
        npz_path = self.landmark_dir / f"{stem}.npz"
        
        face, hands = self._load_npz(npz_path)
        face, face_mask, face_present = mask_nan(face)
        hands, hands_mask, hands_present = mask_nan(hands)
        if self.max_frames is not None:
            face = face[:self.max_frames]
            hands = hands[:self.max_frames]
            face_mask = face_mask[:self.max_frames]
            hands_mask = hands_mask[:self.max_frames]
                
        
            

        token_ids = self.tokenizer.encode(sentence)

        return {
                "video_name": video_name,
                "sentence": sentence,
                "face": torch.from_numpy(face),              
                "hands": torch.from_numpy(hands),   
                "face_mask": torch.from_numpy(face_mask), 
                "hands_mask": torch.from_numpy(hands_mask),
                "input_length": face.shape[0],
                "labels": torch.tensor(token_ids, dtype=torch.long),
            }






def collate_gesture_text(batch, pad_id=0):
    B = len(batch)

    max_t = max(item["input_length"] for item in batch)
    n_face = batch[0]["face"].shape[1]
    n_hand = batch[0]["hands"].shape[1]

    # pad landmark tensors
    face = torch.zeros(B, max_t, n_face, 3, dtype=torch.float32)
    hands = torch.zeros(B, max_t, n_hand, 3, dtype=torch.float32)

    face_mask = torch.zeros(B, max_t, n_face, dtype=torch.bool)
    hands_mask = torch.zeros(B, max_t, n_hand, dtype=torch.bool)

    input_lengths = torch.zeros(B, dtype=torch.long)

    # pad text labels
    max_l = max(len(item["labels"]) for item in batch)
    labels = torch.full((B, max_l), fill_value=pad_id, dtype=torch.long)

    video_names = []
    sentences = []

    for i, item in enumerate(batch):
        T = item["input_length"]
        L = len(item["labels"])

        face[i, :T] = item["face"]
        hands[i, :T] = item["hands"]

        face_mask[i, :T] = item["face_mask"]
        hands_mask[i, :T] = item["hands_mask"]

        input_lengths[i] = T
        labels[i, :L] = item["labels"]

        video_names.append(item["video_name"])
        sentences.append(item["sentence"])

    return {
        "video_name": video_names,
        "sentence": sentences,
        "face": face,
        "hands": hands,
        "face_mask": face_mask,
        "hands_mask": hands_mask,
        "input_lengths": input_lengths,
        "labels": labels,
    }
