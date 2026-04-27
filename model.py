import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers.generation.beam_search import BeamSearchScorer
from transformers.generation.logits_process import LogitsProcessorList, MinLengthLogitsProcessor
from transformers.generation.stopping_criteria import StoppingCriteriaList, MaxLengthCriteria
from typing import Optional

class PositionalEncoding(torch.nn.Module):
  
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32) 
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d-model]
        return x + self.pe[:, :x.size(1)]
    
def lengths_to_mask(lengths: torch.Tensor, max_len: Optional[int]) -> torch.Tensor:
    if max_len is None:
        max_len = int(lengths.max().item())
    rng = torch.arange(max_len, device=lengths.device)[None, :]
    return rng < lengths[:, None]


def subsequent_mask(size: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(size, size, device=device, dtype=torch.bool), diagonal=1)

class FeedForward(nn.Module):
    def __init__(self, d_model: int, factor: int = 4, dropout: float = 0.1):
        super().__init__()
        d_h = d_model * factor
        self.net = nn.Sequential(
            nn.Linear(d_model, d_h),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_h, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)



class MHSA(nn.Module):
   
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        
        out, _ = self.mha(
            x, x, x,
            attn_mask=None,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        return self.dropout(out)

class DepthwiseConv1dModule(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 15, dropout: float = 0.1):
        super().__init__()
        self.pointwise_in = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.depthwise = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=kernel_size // 2, groups=d_model)
        self.bn = nn.BatchNorm1d(d_model)
        self.pointwise_out = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.transpose(1, 2)
        y = self.pointwise_in(y)
        a, b = y.chunk(2, dim=1)
        y = a * torch.sigmoid(b)
        y = self.depthwise(y)
        y = self.bn(y)
        y = F.silu(y)
        y = self.pointwise_out(y)
        y = self.dropout(y)
        return y.transpose(1, 2)



class ConformerLiteBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_mult: int = 4, conv_kernel: int = 15, dropout: float = 0.1):
        super().__init__()
        self.ff1 = FeedForward(d_model, ff_mult, dropout)
        self.ff2 = FeedForward(d_model, ff_mult, dropout)
        self.attn = MHSA(d_model, n_heads, dropout)
        self.conv = DepthwiseConv1dModule(d_model, conv_kernel, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = x + 0.5 * self.ff1(self.norm1(x))
        x = x + self.attn(self.norm2(x), key_padding_mask=key_padding_mask)
        x = x + self.conv(self.norm3(x))
        x = x + 0.5 * self.ff2(self.norm4(x))
        return self.final_norm(x)

class LandmarkAttentionPool(nn.Module):
    def __init__(self, d_point=32, n_heads=1, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_point,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.query = nn.Parameter(torch.randn(1, 1, d_point))

    def forward(self, x, mask):
        B, T, N, D = x.shape

        x = x.reshape(B * T, N, D)
        mask = mask.reshape(B * T, N).bool()

        valid_any = mask.any(dim=1)          # [B*T]
        safe_mask = mask.clone()
        safe_mask[~valid_any] = True         

        key_padding_mask = ~safe_mask
        q = self.query.expand(B * T, 1, D)

        pooled, _ = self.attn(
            query=q,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        pooled = pooled.squeeze(1)
        pooled[~valid_any] = 0.0             # no landmarks => neutral vector
        return pooled.reshape(B, T, D)


class LandmarkProjector(nn.Module):
    def __init__(self, d_point, d_face, d_hand, d_model=256, dropout=0.1):
        super().__init__()
        self.face_point_proj = nn.Sequential(
            nn.Linear(3, d_point),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_point, d_point),
        )
        self.hand_point_proj = nn.Sequential(
            nn.Linear( 3, d_point),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_point, d_point),
        )
        # Frame-level modality projections
        self.face_proj = nn.Sequential(
            nn.Linear(d_point, d_face),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.hand_proj = nn.Sequential(
            nn.Linear(d_point, d_hand),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.fuse = nn.Sequential(
            nn.Linear(d_face + d_hand, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.face_attn = LandmarkAttentionPool(d_point, n_heads=1, dropout=0.1)
        self.hand_attn = LandmarkAttentionPool(d_point, n_heads=1, dropout=0.1)
        
    def forward(self, face, hands, face_mask, hands_mask):
        # face: [B,T,Nf,3]
        # hand: [B,T,Nh,3]
        
        face_pts = self.face_point_proj(face)  
        hand_pts = self.hand_point_proj(hands)  
        face_feat = self.face_attn(face_pts, face_mask)   
        hands_feat = self.hand_attn(hand_pts, hands_mask) 
        # Project each modality
        face_feat = self.face_proj(face_feat)    
        hands_feat = self.hand_proj(hands_feat) 

        # Fuse
        x = torch.cat([face_feat, hands_feat], dim=-1)
        x = self.fuse(x)                         
        return x

class TemporalConvSubsampler(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.net1 = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=5, stride=2, padding=2, groups=d_model),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.Dropout(dropout)
        )
        self.net2 = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=5, stride=2, padding=2, groups=d_model),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.Dropout(dropout)
        )

    def forward(self, x, lengths):
        # x: [B,T,d_model]
        x = x.transpose(1, 2)   
        x = self.net1(x)
        lengths = torch.div(lengths + 1, 2, rounding_mode="floor")
        x = self.net2(x)
        lengths = torch.div(lengths + 1, 2, rounding_mode="floor")
        x = x.transpose(1, 2)  
        return x, lengths


class ConformerEncoder(nn.Module):
    def __init__(self, d_point: int, d_face: int, d_hand: int, d_model: int = 256, depth: int = 4, n_heads: int = 4, ff_mult: int = 4, conv_kernel: int = 15, dropout: float = 0.1):
        super().__init__()
        self.projector = LandmarkProjector(d_point, d_face, d_hand, d_model)
        self.subsample = TemporalConvSubsampler(d_model)
        self.pos = PositionalEncoding(d_model)
        self.blocks = nn.ModuleList([ConformerLiteBlock(d_model, n_heads, ff_mult, conv_kernel, dropout) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, face_x: torch.Tensor, hand_x: torch.Tensor, face_mask: torch.tensor, hands_mask: torch.tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.projector(face_x, hand_x, face_mask, hands_mask)
        x, lengths = self.subsample(x, lengths)
        x = self.pos(x)
        pad_mask = ~lengths_to_mask(lengths, x.size(1))
        for block in self.blocks:
            x = block(x, key_padding_mask=pad_mask)
        return self.out_norm(x), lengths


class TransformerTextDecoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, n_heads: int = 4, depth: int = 3, ff_mult: int = 4, dropout: float = 0.1, pad_id: int = 0):
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=depth)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, tgt_ids: torch.Tensor, memory: torch.Tensor, memory_lengths: torch.Tensor) -> torch.Tensor:
        tgt = self.pos(self.embedding(tgt_ids))
        tgt_mask = subsequent_mask(tgt.size(1), tgt.device)
        tgt_key_padding_mask = tgt_ids.eq(self.pad_id)
        memory_key_padding_mask = ~lengths_to_mask(memory_lengths, memory.size(1))
        dec = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.out(dec)

class GestureToTextModel(nn.Module):
    def __init__(
        self,
        d_point: int,
        d_face: int,
        d_hand: int,
        vocab_size: int,
        pad_id: int,
        d_model: int = 256,
        enc_depth: int = 4,
        dec_depth: int = 3,
        n_heads: int = 4
    ):
        super().__init__()
        self.encoder = ConformerEncoder(
            d_point=d_point,
            d_face=d_face,
            d_hand=d_hand,
            d_model=d_model,
            depth=enc_depth,
            n_heads=n_heads,
        )
        self.decoder = TransformerTextDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            depth=dec_depth,
            pad_id=pad_id,
        )
        self.linear = nn.Linear(d_model, vocab_size)
        self.vocab_size = vocab_size
        self.pad_id = pad_id

    def forward(
        self,
        face: torch.Tensor,
        hands: torch.Tensor,
        face_mask: torch.Tensor,
        hands_mask: torch.Tensor,
        input_lengths: torch.Tensor,
        decoder_input_ids: torch.Tensor,
    ):
        memory, memory_lengths = self.encoder(face, hands, face_mask, hands_mask, input_lengths)
        lm_logits = self.decoder(decoder_input_ids, memory, memory_lengths)
        ctc_logits = self.linear(memory)
        return {
            "lm_logits": lm_logits,
            "ctc_logits": ctc_logits,
            "memory_lengths": memory_lengths,
        }

    def _decode_step(self, ys, memory, memory_lengths):
        logits = self.decoder(ys, memory, memory_lengths)
        return logits[:, -1, :]

    @torch.no_grad()
    def beam_search_generate(
        self,
        face: torch.Tensor,
        hands: torch.Tensor,
        face_mask: torch.Tensor,
        hands_mask: torch.Tensor,
        input_lengths: torch.Tensor,
        bos_id: int,
        eos_id: int,
        pad_id: int,
        max_new_tokens: int = 48,
        num_beams: int = 6,
        length_penalty: float = 0.8,
        min_new_tokens: int = 0,
        early_stopping: bool = True,
    ) -> torch.Tensor:

        device = face.device
        memory, memory_lengths = self.encoder(face, hands, face_mask, hands_mask, input_lengths)
        batch_size = face.size(0)

        memory = memory.repeat_interleave(num_beams, dim=0)
        memory_lengths = memory_lengths.repeat_interleave(num_beams, dim=0)

        input_ids = torch.full(
            (batch_size * num_beams, 1),
            fill_value=bos_id,
            dtype=torch.long,
            device=device,
        )

        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=num_beams,
            device=device,
            length_penalty=length_penalty,
            do_early_stopping=early_stopping,
        )

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)

        logits_processor = LogitsProcessorList()
        if min_new_tokens > 0:
            logits_processor.append(
                MinLengthLogitsProcessor(min_length=1 + min_new_tokens, eos_token_id=eos_id)
            )

        cur_len = input_ids.shape[1]

        while True:
            next_logits = self._decode_step(input_ids, memory, memory_lengths)
            next_log_probs = F.log_softmax(next_logits, dim=-1)

            next_log_probs = logits_processor(input_ids, next_log_probs)
            next_log_probs = next_log_probs + beam_scores[:, None]

            vocab_size = next_log_probs.shape[-1]
            next_scores = next_log_probs.view(batch_size, num_beams * vocab_size)

            next_scores, next_tokens = torch.topk(
                next_scores,
                k=2 * num_beams,
                dim=1,
                largest=True,
                sorted=True,
            )

            next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
            next_tokens = next_tokens % vocab_size

            beam_outputs = beam_scorer.process(
                input_ids=input_ids,
                next_scores=next_scores,
                next_tokens=next_tokens,
                next_indices=next_indices,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
            )

            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
            memory = memory[beam_idx]
            memory_lengths = memory_lengths[beam_idx]

            cur_len += 1
            if beam_scorer.is_done or cur_len >= 1 + max_new_tokens:
                break

        final = beam_scorer.finalize(
            input_ids=input_ids,
            final_beam_scores=beam_scores,
            final_beam_tokens=beam_next_tokens,
            final_beam_indices=beam_idx,
            max_length=cur_len,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
        )
        return final["sequences"]

    @torch.no_grad()
    def generate(
        self,
        face: torch.Tensor,
        hands: torch.Tensor,
        face_mask: torch.Tensor,
        hands_mask: torch.Tensor,
        input_lengths: torch.Tensor,
        bos_id: int,
        eos_id: int,
        pad_id: int,
        max_new_tokens: int = 48,
        num_beams: int = 4,
        length_penalty: float = 1.0,
    ) -> torch.Tensor:
        return self.beam_search_generate(
            face=face,
            hands=hands,
            face_mask=face_mask,
            hands_mask=hands_mask,
            input_lengths=input_lengths,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            length_penalty=length_penalty,
        )
