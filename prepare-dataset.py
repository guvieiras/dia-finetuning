import argparse
import os
import uuid
import tempfile
import warnings

import torch
# Limit PyTorch threads for stability
torch.set_num_threads(1)
# Enable TF32 for performance changing reproducibility
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torchaudio
import pandas as pd
import numpy as np
import librosa
import re
from tqdm import tqdm
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline as hf_pipeline
)
from pyannote.audio import Pipeline as DiarizationPipeline

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning,
                        module='pyannote.audio.utils.reproducibility')
warnings.filterwarnings('ignore', category=FutureWarning,
                        module='transformers.models.whisper.generation_whisper')

# Utility: load and resample audio to 16k mono
def load_and_resample(path: str, target_sr: int = 16000):
    audio, sr = librosa.load(path, sr=None)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio, target_sr

# Split audio into fixed-length segments with optional trim and random names
def split_audio_fixed(
    audio_path: str,
    output_dir: str,
    segment_length: int = 30,
    trim_start: float = 0.0,
    trim_end: float = 0.0
) -> list[str]:
    os.makedirs(output_dir, exist_ok=True)
    waveform, sr = torchaudio.load(audio_path)
    if waveform.dim() > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    total = waveform.size(1)
    # Trim start and end
    start = int(trim_start * sr)
    end = total - int(trim_end * sr)
    waveform = waveform[:, start:end]

    paths = []
    seg_samples = int(segment_length * sr)
    for i in range(0, waveform.size(1), seg_samples):
        seg = waveform[:, i:i + seg_samples]
        fname = f"{uuid.uuid4().hex}.wav"
        seg_path = os.path.join(output_dir, fname)
        torchaudio.save(seg_path, seg, sr)
        paths.append(seg_path)
    return paths

# Clean transcription text
def clean_text(text: str) -> str:
    text = re.sub(r"<\|.*?\|>", " ", text)
    return " ".join(text.split())

# Process a single segment: diarize + transcribe
def process_segment(seg_path: str, asr_pipe, diar_pipe) -> str:
    diar = diar_pipe(seg_path)
    turns = list(diar.itertracks(yield_label=True))

    result = asr_pipe(seg_path, return_timestamps=True)
    lines = []
    for chunk in result.get('chunks', []):
        t0, t1 = chunk.get('timestamp', (None, None))
        if t0 is None or t1 is None:
            continue
        mid = (t0 + t1) / 2
        speaker = 'S?'
        for turn, _, label in turns:
            if turn.start <= mid <= turn.end:
                try:
                    idx = int(label.split('_')[-1]) + 1
                    speaker = f"S{idx}"
                except:
                    speaker = 'S?'
                break
        txt = clean_text(chunk.get('text', ''))
        if txt:
            lines.append(f"[{speaker}] {txt}")
    return "\n".join(lines)

# Main function
def main():
    parser = argparse.ArgumentParser(
        description="Split, diarize & transcribe with speaker tags"
    )
    parser.add_argument('audio_path', help='Input audio file')
    parser.add_argument('--output_dir', default='segments', help='Segments folder')
    parser.add_argument('--csv_path', default='transcriptions.csv', help='CSV output path')
    parser.add_argument('--segment_length', type=int, default=30, help='Segment seconds')
    parser.add_argument('--hf_token', required=True, help='HF token')
    parser.add_argument('--trim_start', type=float, default=0.0, help='Trim start sec')
    parser.add_argument('--trim_end', type=float, default=0.0, help='Trim end sec')
    parser.add_argument('--append_csv', action='store_true', help='Append CSV')
    parser.add_argument('--full_transcribe', action='store_true', help='Full then split')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup ASR pipeline
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        'openai/whisper-large-v3', torch_dtype=torch.float16
    ).to(device)
    proc = AutoProcessor.from_pretrained('openai/whisper-large-v3')
    asr_pipe = hf_pipeline(
        'automatic-speech-recognition',
        model=model,
        tokenizer=proc.tokenizer,
        feature_extractor=proc.feature_extractor,
        device=0 if device.type=='cuda' else -1,
        chunk_length_s=args.segment_length,
        return_timestamps=True,
        generate_kwargs={
            'task':'transcribe','language':'pt',
            'num_beams':5,'early_stopping':True,
            'no_repeat_ngram_size':2,'forced_decoder_ids':None
        }
    )

    # Diarization pipeline
    diar_pipe = DiarizationPipeline.from_pretrained(
        'pyannote/speaker-diarization-3.1', use_auth_token=args.hf_token
    ).to(device)

    # Precompute full ASR & diarization
    full_out, full_turns = None, None
    if args.full_transcribe:
        print("🔹 Full ASR & diarization...")
        audio_np, sr = load_and_resample(args.audio_path)
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        torchaudio.save(tmp.name, torch.from_numpy(audio_np).unsqueeze(0), sr)
        full_out = asr_pipe(tmp.name, return_timestamps=True)
        tmp.close(); os.unlink(tmp.name)
        diar_full = diar_pipe(args.audio_path)
        full_turns = list(diar_full.itertracks(yield_label=True))

    # Segmenting
    if args.full_transcribe:
        print("🔹 Splitting post-full...")
        segments = split_audio_fixed(
            args.audio_path, args.output_dir,
            args.segment_length, args.trim_start, args.trim_end
        )
        seg_times = [(
            seg,
            args.trim_start + i*args.segment_length,
            args.trim_start + (i+1)*args.segment_length
        ) for i, seg in enumerate(segments)]
    else:
        print("🔹 Splitting audio...")
        segments = split_audio_fixed(
            args.audio_path, args.output_dir,
            args.segment_length, args.trim_start, args.trim_end
        )
        seg_times = [(seg, None, None) for seg in segments]
    print(f"🔸 {len(segments)} segments created in '{args.output_dir}'")

    # Process segments
    rows = []
    print("🔹 Processing...")
    for seg, st, en in tqdm(seg_times, desc='Segments'):
        if args.full_transcribe:
            parts = []
            for ch in full_out.get('chunks', []):
                t0,t1 = ch.get('timestamp', (None,None))
                if t0 is None or t1 is None: continue
                mid = (t0 + t1)/2
                if not (st <= mid < en): continue
                sp = 'S?'
                for turn,_,lbl in full_turns:
                    if turn.start <= mid <= turn.end:
                        try: sp = f"S{int(lbl.split('_')[-1])+1}"
                        except: pass
                        break
                txt = clean_text(ch.get('text',''))
                if txt: parts.append(f"[{sp}] {txt}")
            tr = "\n".join(parts)
        else:
            tr = process_segment(seg, asr_pipe, diar_pipe)
        print(f"{os.path.basename(seg)} ->\n{tr}\n")
        rows.append({'audio': seg, 'text': tr})

    # Save CSV
    df = pd.DataFrame(rows, columns=['audio','text'])
    mode = 'a' if args.append_csv and os.path.exists(args.csv_path) else 'w'
    df.to_csv(args.csv_path, index=False, mode=mode, header=False, sep='|')
    print(f"✅ CSV saved to '{args.csv_path}'")

if __name__=='__main__':
    main()


# **Instructions:**
# 1. Install dependencies: `pip install torch torchaudio transformers pyannote.audio librosa tqdm`.
# 2. Run:
#    ```bash
#    python prepare-data.py podcast.m4a --output_dir segments --csv_path resultado.csv --segment_length 30 --hf_token YOUR_HF_TOKEN
#    ```
# 3. Segments get random UUID names; CSV contains `audio_path` and `transcription` with `[S1]`, `[S2]` tags.

# Let me know if you'd like further tweaks!```
