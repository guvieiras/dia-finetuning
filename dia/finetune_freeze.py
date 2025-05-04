import argparse
import logging
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
import torchaudio
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from torch.cuda.amp import autocast
from torch.nn.functional import pad, cross_entropy
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import get_scheduler
import bitsandbytes as bnb

import dac
from dia.config import DiaConfig, DataConfig
from dia.layers import DiaModel
from dia.model import Dia
from dia.audio import build_delay_indices, apply_audio_delay

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Select compute device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


def freeze_encoder_and_embedding(model: DiaModel):
    """
    Freeze encoder embeddings, encoder layers, and final encoder norm to
    preserve multilingual text understanding during fine-tuning.
    """
    # Freeze embedding layer
    if hasattr(model.encoder, "embedding"):
        for param in model.encoder.embedding.parameters():
            param.requires_grad = False
    # Freeze each encoder transformer layer
    for layer in model.encoder.layers:
        for param in layer.parameters():
            param.requires_grad = False
    # Freeze final normalization layer
    if hasattr(model.encoder, "norm"):
        for param in model.encoder.norm.parameters():
            param.requires_grad = False


def get_args() -> argparse.Namespace:
    """
    Parse command-line arguments for fine-tuning.
    """
    parser = argparse.ArgumentParser(description="Fine-tune the Dia text-to-speech model")
    parser.add_argument("--config", type=Path, default=Path("dia/config.json"), help="Path to Dia configuration JSON file")
    parser.add_argument("--dataset", type=str, default="Paradoxia/opendata-iisys-hui", help="HuggingFace dataset identifier")
    parser.add_argument("--local_ckpt", type=Path, default=None, help="Local checkpoint file path (overrides downloading)")
    parser.add_argument("--csv_path", type=Path, default=None, help="Path to local CSV/TSV file with audio and text columns")
    parser.add_argument("--audio_root", type=Path, default=None, help="Root directory for local audio files (required with --csv_path)")
    parser.add_argument("--run_name", type=str, default=None, help="TensorBoard run name")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"), help="Directory to save checkpoints and logs")
    # HF dataset columns
    parser.add_argument("--hf_audio_col", type=str, default="audio", help="Name of audio column in HF dataset")
    parser.add_argument("--hf_text_col", type=str, default="text", help="Name of text column in HF dataset")
    parser.add_argument("--max_samples_per_epoch", type=int, default=None, help="Max samples per epoch (HF only)")
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per step")
    parser.add_argument("--grad_accum_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for optimizer")
    parser.add_argument("--warmup_percentage", type=float, default=0.001, help="Warmup fraction for scheduler")
    parser.add_argument("--audio_prompt_frac", type=float, default=0.2, help="Fraction of steps using audio prompt")
    parser.add_argument("--eval_step", type=int, default=200, help="Evaluation every N steps")
    parser.add_argument("--save_step", type=int, default=2000, help="Checkpoint save every N steps")
    parser.add_argument("--split_ratio", type=float, default=0.9997, help="Train/val split ratio")
    return parser.parse_args()


@dataclass
class TrainConfig:
    """
    Configuration for training loop and hyperparameters.
    """
    epochs: int
    batch_size: int
    grad_accum_steps: int
    learning_rate: float
    warmup_percentage: float
    audio_prompt_frac: float
    eval_step: int
    save_step: int
    split_ratio: float
    runs_dir: Path = Path("runs")
    run_name: str = "dia_finetune"
    output_dir: Path = Path("outputs")


class LocalDiaDataset(Dataset):
    """
    Dataset wrapper for local CSV/TSV and audio files.
    """
    def __init__(
        self,
        csv_path: Path,
        audio_root: Path,
        config: DiaConfig,
        dac_model: dac.DAC,
        audio_col: str = "audio",
        text_col: str = "text",
    ):
        try:
            self.df = pd.read_csv(csv_path, sep="|", engine="python")
        except Exception:
            self.df = pd.read_csv(csv_path, sep="|", engine="python", header=None)
            self.df.columns = [audio_col, text_col]
        if audio_col not in self.df.columns or text_col not in self.df.columns:
            raise ValueError(f"CSV must contain columns: {audio_col}, {text_col}")
        self.audio_root = audio_root
        self.config = config
        self.dac_model = dac_model
        self.audio_col = audio_col
        self.text_col = text_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        text = row[self.text_col]
        audio_path = self.audio_root / row[self.audio_col]
        waveform, sr = torchaudio.load(audio_path)
        if sr != 44100:
            waveform = torchaudio.functional.resample(waveform, sr, 44100)
        waveform = waveform.unsqueeze(0)
        with torch.no_grad():
            audio_tensor = self.dac_model.preprocess(waveform, 44100)
            audio_tensor = audio_tensor.to(next(self.dac_model.parameters()).device)
            _, encoded, *_ = self.dac_model.encode(audio_tensor, n_quantizers=None)
            encoded = encoded.squeeze(0).transpose(0, 1)
        return text, encoded, waveform


class HFDiaDataset(Dataset):
    """
    Dataset wrapper for HuggingFace `datasets.Dataset`.
    """
    def __init__(
        self,
        hf_dataset,
        config: DiaConfig,
        dac_model: dac.DAC,
        audio_col: str = "audio",
        text_col: str = "text",
    ):
        self.dataset = hf_dataset
        self.config = config
        self.dac_model = dac_model
        self.audio_col = audio_col
        self.text_col = text_col

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        sample = self.dataset[idx]
        text = sample[self.text_col]
        audio_info = sample[self.audio_col]
        waveform = torch.tensor(audio_info['array'], dtype=torch.float32)
        waveform = waveform.unsqueeze(0).unsqueeze(0)
        sr = audio_info['sampling_rate']
        if sr != 44100:
            waveform = torchaudio.functional.resample(waveform, sr, 44100)
        with torch.no_grad():
            audio_tensor = self.dac_model.preprocess(waveform, 44100)
            audio_tensor = audio_tensor.to(next(self.dac_model.parameters()).device)
            _, encoded, *_ = self.dac_model.encode(audio_tensor, n_quantizers=None)
            encoded = encoded.squeeze(0).transpose(0, 1)
        return text, encoded, waveform


def collate_fn(batch, config: DiaConfig, device: torch.device):
    """
    Collate batch of samples into tensors for model input and target.
    """
    texts, encodings, waveforms = zip(*batch)
    # Text to byte tokens with padding
    max_text = config.data.text_length
    pad_tok = config.data.text_pad_value
    text_ids = []
    for txt in texts:
        bts = txt.encode('utf-8')[:max_text]
        arr = list(bts) + [pad_tok] * (max_text - len(bts))
        text_ids.append(torch.tensor(arr, dtype=torch.long))
    src = torch.stack(text_ids).to(device)
    src_pos = torch.arange(max_text, device=device).unsqueeze(0).expand(src.size(0), -1)
    src_pad = src.ne(pad_tok)
    enc_self_attn_mask = (src_pad.unsqueeze(2) & src_pad.unsqueeze(1)).unsqueeze(1)

    # Audio code padding and delay
    max_audio = config.data.audio_length
    seq_lens = [e.size(0) for e in encodings]
    true_len = min(max(seq_lens), max_audio)
    padded = [pad(e, (0, 0, 0, true_len - e.size(0))) if e.size(0) < true_len else e[:true_len]
              for e in encodings]
    codes = torch.stack(padded).to(device)
    B, T, C = codes.shape
    t_idx, idxs = build_delay_indices(B, T, C, config.data.delay_pattern)
    delayed = apply_audio_delay(codes, config.data.audio_pad_value,
                                config.data.audio_bos_value, (t_idx, idxs))[:, :max_audio, :]

    # Prepare target sequences
    max_tgt_len = max_audio + 2
    tgt = torch.full((B, max_tgt_len, C), config.data.audio_pad_value,
                     device=device, dtype=torch.long)
    tgt[:, 0, :] = config.data.audio_bos_value
    tgt[:, 1:1 + true_len, :] = delayed[:, :true_len, :]
    eos_pos = 1 + true_len
    tgt[:, eos_pos, :] = config.data.audio_eos_value
    tgt_len = eos_pos + 1
    tgt_pos = torch.arange(max_tgt_len, device=device).unsqueeze(0).expand(B, -1)
    tgt_pad = tgt.ne(config.data.audio_pad_value).any(-1)
    causal = torch.tril(torch.ones((max_tgt_len, max_tgt_len), dtype=torch.bool, device=device))
    dec_self_attn_mask = (tgt_pad.unsqueeze(2) & tgt_pad.unsqueeze(1) & causal).unsqueeze(1)
    dec_cross_attn_mask = (tgt_pad.unsqueeze(2) & src_pad.unsqueeze(1)).unsqueeze(1)

    return {
        'src_tokens': src,
        'src_positions': src_pos,
        'enc_self_attn_mask': enc_self_attn_mask,
        'tgt_tokens': tgt,
        'tgt_positions': tgt_pos,
        'dec_self_attn_mask': dec_self_attn_mask,
        'dec_cross_attn_mask': dec_cross_attn_mask,
        'waveforms': waveforms,
        'raw_text': texts[0],
        'tgt_len': tgt_len,
    }


def setup_loaders(dataset: Dataset, config: DiaConfig, train_cfg: TrainConfig):
    """
    Split dataset and create DataLoader for training and validation.
    """
    n_train = int(train_cfg.split_ratio * len(dataset))
    train_ds, val_ds = random_split(dataset, [n_train, len(dataset) - n_train])
    coll = lambda b: collate_fn(b, config, device)
    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size,
                              shuffle=True, collate_fn=coll)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=coll)
    return train_loader, val_loader


def setup_optimizer_and_scheduler(model, train_loader, train_cfg: TrainConfig):
    """
    Create optimizer (8-bit AdamW) and cosine scheduler.
    """
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=train_cfg.learning_rate)
    total_steps = len(train_loader) * train_cfg.epochs
    scheduler = get_scheduler(
        'cosine', optimizer,
        num_warmup_steps=int(train_cfg.warmup_percentage * total_steps),
        num_training_steps=total_steps
    )
    return optimizer, scheduler


def train_step(model, batch, config: DiaConfig, train_cfg: TrainConfig,
               optimizer, scheduler, writer, global_step: int):
    """
    Perform one training step with optional audio prompting.
    """
    use_prompt = random.random() < train_cfg.audio_prompt_frac
    with autocast():
        logits = model(
            src_BxS=batch['src_tokens'],
            tgt_BxTxC=batch['tgt_tokens'],
            src_positions=batch['src_positions'],
            tgt_positions=batch['tgt_positions'],
            enc_self_attn_mask=batch['enc_self_attn_mask'],
            dec_self_attn_mask=batch['dec_self_attn_mask'],
            dec_cross_attn_mask=batch['dec_cross_attn_mask'],
            enable_dropout=True,
        )
        L = batch['tgt_len']
        logits = logits[:, :L-1]
        target = batch['tgt_tokens'][:, 1:L]
        loss = cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target.reshape(-1),
            ignore_index=config.data.audio_pad_value
        )
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    writer.add_scalar('Loss/train', loss.item(), global_step)


def eval_step(model, val_loader, config: DiaConfig, dac_model: dac.DAC,
              writer, global_step: int):
    """
    Evaluate on validation set and log audio examples.
    """
    eval_losses = []
    last_batch = None
    for batch in tqdm(val_loader, desc="Evaluation"):
        last_batch = batch
        logits_eval = model(
            src_BxS=batch['src_tokens'],
            tgt_BxTxC=batch['tgt_tokens'],
            src_positions=batch['src_positions'],
            tgt_positions=batch['tgt_positions'],
            enc_self_attn_mask=batch['enc_self_attn_mask'],
            dec_self_attn_mask=batch['dec_self_attn_mask'],
            dec_cross_attn_mask=batch['dec_cross_attn_mask'],
            enable_dropout=False,
        )[:, :-1]
        target_eval = batch['tgt_tokens'][:, 1:]
        B_e, T_e, C_e = target_eval.shape
        V_e = logits_eval.size(-1)
        loss_e = 0.0
        weights_e = [4.0] + [1.0] * (C_e - 1)
        for c, w in enumerate(weights_e):
            lc = logits_eval[:, :, c, :].reshape(-1, V_e)
            tc = target_eval[:, :, c].reshape(-1)
            loss_e += w * cross_entropy(lc, tc, ignore_index=config.data.audio_pad_value)
        eval_losses.append(loss_e / sum(weights_e))

    avg_eval_loss = sum(eval_losses) / len(eval_losses)
    writer.add_scalar('Loss/eval', avg_eval_loss, global_step)

    # Log example audio
    dia_gen = Dia(config, device)
    dia_gen.model, dia_gen.dac_model = model, dac_model
    with autocast():
        audio_no = dia_gen.generate(text=last_batch['raw_text'])
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        torchaudio.save(tmp.name, last_batch['waveforms'][0].squeeze(0), 44100)
        audio_with = dia_gen.generate(text=last_batch['raw_text'], audio_prompt_path=tmp.name)
        os.unlink(tmp.name)
    writer.add_audio('Eval/no_prompt', audio_no, global_step, 44100)
    writer.add_audio('Eval/with_prompt', audio_with, global_step, 44100)


def train():
    """
    Main training loop: load data, model, freeze encoder, and train.
    """
    args = get_args()
    config = DiaConfig.load(str(args.config))

    # Load or initialize model
    if args.local_ckpt:
        model = DiaModel(config)
        state_dict = torch.load(args.local_ckpt, map_location=device)
        model.load_state_dict(state_dict)
    else:
        model = DiaModel(config)
    model.to(device)

    # Freeze encoder and embeddings
    freeze_encoder_and_embedding(model)

    # Load DAC model for audio preprocessing
    dac_model_path = dac.utils.download()
    dac_model = dac.DAC.load(dac_model_path).to(device)

    # Prepare dataset
    if args.csv_path and args.audio_root:
        dataset = LocalDiaDataset(
            csv_path=args.csv_path,
            audio_root=args.audio_root,
            config=config,
            dac_model=dac_model,
            audio_col=args.hf_audio_col,
            text_col=args.hf_text_col
        )
    else:
        hf_data = load_dataset(args.dataset, split="train")
        if args.max_samples_per_epoch:
            hf_data = hf_data.select(range(args.max_samples_per_epoch))
        dataset = HFDiaDataset(
            hf_dataset=hf_data,
            config=config,
            dac_model=dac_model,
            audio_col=args.hf_audio_col,
            text_col=args.hf_text_col
        )

    train_cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        warmup_percentage=args.warmup_percentage,
        audio_prompt_frac=args.audio_prompt_frac,
        eval_step=args.eval_step,
        save_step=args.save_step,
        split_ratio=args.split_ratio,
        runs_dir=Path("runs"),
        run_name=args.run_name or "dia_finetune",
        output_dir=args.output_dir
    )

    # Setup loaders, optimizer, and scheduler
    train_loader, val_loader = setup_loaders(dataset, config, train_cfg)
    optimizer, scheduler = setup_optimizer_and_scheduler(model, train_loader, train_cfg)
    writer = SummaryWriter(log_dir=train_cfg.runs_dir / train_cfg.run_name)

    global_step = 0
    model.train()
    for epoch in range(train_cfg.epochs):
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_cfg.epochs}"):
            train_step(model, batch, config, train_cfg, optimizer, scheduler, writer, global_step)
            global_step += 1
            if global_step % train_cfg.eval_step == 0:
                model.eval()
                eval_step(model, val_loader, config, dac_model, writer, global_step)
                model.train()
            if global_step % train_cfg.save_step == 0:
                ckpt = train_cfg.output_dir / f"checkpoint_{global_step}.pth"
                torch.save(model.state_dict(), ckpt)

    # Save final model
    final_path = train_cfg.output_dir / "final_model.pth"
    torch.save(model.state_dict(), final_path)
    writer.close()


if __name__ == "__main__":
    train()
