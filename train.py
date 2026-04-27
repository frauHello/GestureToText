import argparse
import os
from torch.utils.data import DataLoader
import torch
from dataloader import  TextTokenizer, GestureTextDataset, collate_gesture_text
from model import GestureToTextModel
from utils import set_seed, shift_right, compute_losses, save_checkpoint, save_metrics, evaluate
from pathlib import Path
import pandas as pd

def parse_args():
    p = argparse.ArgumentParser()
   
    p.add_argument("--seed", type=int, default=111)
    p.add_argument("--tokenizer_name", type=str, default="t5-small")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--max_frames", type=int, default=200)
    p.add_argument("--frame_stride", type=int, default=1)
    p.add_argument("--cache_dir", type=str, default="./cache")
    p.add_argument("--save_dir", type=str, default="./outputs")
    p.add_argument("--max_text_length", type=int, default=64)
    p.add_argument("--ctc_weight", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_workers", type=int, default=0)
    
    return p.parse_args(args=[])






def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
  
    
    tokenizer = TextTokenizer(name=args.tokenizer_name, max_length=args.max_text_length)
    train_csv = Path("dataset/how2sign_realigned_train.csv")
    train_landmark_dir = "dataset/train_rgb_front_clips/processed_features"
    val_csv = Path("dataset/how2sign_realigned_val.csv")
    val_landmark_dir = "dataset/val_rgb_front_clips/processed_features"
    train_ds = GestureTextDataset( csv_path=train_csv, landmark_dir = train_landmark_dir, tokenizer = tokenizer)
    val_ds = GestureTextDataset( csv_path=val_csv, landmark_dir = val_landmark_dir, tokenizer = tokenizer)

    sample0 = train_ds[0]
    model = GestureToTextModel(
        d_point=32,
        d_face=64,
        d_hand=128,
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_id,
        d_model=256,
        enc_depth=3,
        dec_depth=3,
        n_heads=4,
    ).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_gesture_text(b, pad_id=tokenizer.pad_id),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_gesture_text(b, pad_id=tokenizer.pad_id),
        pin_memory=True,
    )

    best_wer = float("inf")
    validation = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        for step, batch in enumerate(train_loader, start=1):
            face = batch["face"].to(args.device)
            hands = batch["hands"].to(args.device)
            face_mask = batch["face_mask"].to(args.device)
            hands_mask = batch["hands_mask"].to(args.device)
            input_lengths = batch["input_lengths"].to(args.device)
            labels = batch["labels"].to(args.device)
            decoder_in = shift_right(labels, tokenizer.bos_id, tokenizer.pad_id)
            scaler = torch.amp.GradScaler("cuda")
            with torch.amp.autocast("cuda"):
                outputs = model(face=face,hands=hands,face_mask=face_mask,hands_mask=hands_mask, input_lengths=input_lengths, decoder_input_ids=decoder_in)
            
                loss, metrics = compute_losses(
                batch={**batch, "labels": labels},
                outputs=outputs,
                pad_id=tokenizer.pad_id,
                ctc_weight=args.ctc_weight,
            )

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            if step % 10 == 0:
                print(f"epoch={epoch} step={step} loss={metrics['loss']:.4f} seq={metrics['seq_loss']:.4f} ctc={metrics['ctc_loss']:.4f}")

        scheduler.step()

        val_metrics = evaluate(model, val_loader, tokenizer, args.device, args.ctc_weight)
        record = {"epoch": epoch, "val_loss": val_metrics["loss"], "val_wer": val_metrics["wer"]}
        validation.append(record)
        print(f"[epoch {epoch}] val_loss={val_metrics['loss']:.4f} val_wer={val_metrics['wer']:.4f}")

        save_checkpoint(os.path.join(args.save_dir, "last.pt"), model=model, optimizer=optimizer, epoch=epoch, config=vars(args))
        save_metrics(os.path.join(args.save_dir, "validation.json"), {"validation": validation})

        if val_metrics["wer"] < best_wer:
            best_wer = val_metrics["wer"]
            save_checkpoint(os.path.join(args.save_dir, "best.pt"), model=model, optimizer=optimizer, epoch=epoch, config=vars(args))
            print(f"saved new best model to {os.path.join(args.save_dir, 'best.pt')}")
    print("4")
    print(f"training done. last checkpoint: {os.path.join(args.save_dir, 'last.pt')}")
    print(f"best checkpoint: {os.path.join(args.save_dir, 'best.pt')}")




if __name__ == "__main__":
    main()
