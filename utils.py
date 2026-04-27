import random
import torch
import torch.nn.functional as F
import os
import numpy as np
import json

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def edit_distance_word_level(pred: str, ref: str) -> int:
    a = pred.split()
    b = ref.split()
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        dp[i][0] = i
    for j in range(len(b) + 1):
        dp[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[-1][-1]


def wer(pred: str, ref: str) -> float:
    ref_words = ref.split()
    if len(ref_words) == 0:
        return 0.0
    return edit_distance_word_level(pred, ref) / len(ref_words)



def shift_right(labels: torch.Tensor, bos_id: int, pad_id: int) -> torch.Tensor:
    out = labels.new_full(labels.shape, pad_id)
    out[:, 0] = bos_id
    out[:, 1:] = labels[:, :-1]
    return out


def sequence_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, pad_id: int) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=pad_id,
    )


        
def compute_losses(batch: dict, outputs: dict, pad_id: int, ctc_weight: float) -> tuple[torch.Tensor, dict[str, float]]:
    labels = batch["labels"]
    lm_logits = outputs["lm_logits"]
    ctc_logits = outputs["ctc_logits"]
    memory_lengths = outputs["memory_lengths"]
    
    seq_loss = sequence_cross_entropy(lm_logits, labels, pad_id=pad_id)

    blank_id = 0
    ctc_targets = labels.clone()
    ctc_mask = ~ctc_targets.eq(pad_id)
    flat_targets = (ctc_targets[ctc_mask] + 1).to(torch.long)
    target_lengths = ctc_mask.sum(dim=1)

    ctc_log_probs = F.log_softmax(ctc_logits, dim=-1)
    ctc_log_probs = torch.cat([ctc_log_probs.new_full((*ctc_log_probs.shape[:-1], 1), fill_value=-20.0), ctc_log_probs], dim=-1)
    ctc_loss = F.ctc_loss(
        ctc_log_probs.transpose(0, 1),
        flat_targets,
        input_lengths=memory_lengths,
        target_lengths=target_lengths,
        blank=blank_id,
        zero_infinity=True,
    )

    total = (1.0 - ctc_weight) * seq_loss + ctc_weight * ctc_loss
    return total, {"loss": float(total.detach().item()), "seq_loss": float(seq_loss.detach().item()), "ctc_loss": float(ctc_loss.detach().item())}

@torch.no_grad()
def evaluate(model, loader, tokenizer, device: str, ctc_weight: float) -> dict[str, float]:
    model.eval()
    losses, wers = [], []
    for batch in loader:
        face = batch["face"].to(device)
        hands = batch["hands"].to(device)
        face_mask = batch["face_mask"].to(device)
        hands_mask = batch["hands_mask"].to(device)
        input_lengths = batch["input_lengths"].to(device)
        labels = batch["labels"].to(device)
        decoder_in = shift_right(labels, tokenizer.bos_id, tokenizer.pad_id)
        outputs = model(face=face,hands=hands,face_mask=face_mask,hands_mask=hands_mask, input_lengths=input_lengths, decoder_input_ids=decoder_in)
        loss, _ = compute_losses(batch={**batch, "labels": labels}, outputs=outputs, pad_id=tokenizer.pad_id, ctc_weight=ctc_weight)
        losses.append(float(loss.item()))
        pred_ids = model.generate(
        face=face,
        hands=hands,
        face_mask=face_mask,
        hands_mask=hands_mask,
        input_lengths=input_lengths,
        bos_id=tokenizer.bos_id,
        eos_id=tokenizer.eos_id,
        pad_id=tokenizer.pad_id,
        max_new_tokens=64,
        num_beams=3,
        length_penalty=1.0)
        
        for i in range(pred_ids.size(0)):
            pred_text = tokenizer.decode(pred_ids[i].tolist())
            ref_text = batch["sentence"][i]
            wers.append(wer(pred_text.lower(), ref_text.lower()))

    return {"loss": float(np.mean(losses)) if losses else 0.0, "wer": float(np.mean(wers)) if wers else 1.0}


def save_checkpoint(path: str, model, optimizer, epoch: int, config: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "epoch": epoch,
            "config": config
        },
        path,
    )


def save_metrics(path: str, metrics: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
