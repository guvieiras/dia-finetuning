import argparse
import os
import uuid
import tempfile
import warnings
import glob

import torch
torch.set_num_threads(1)
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

warnings.filterwarnings('ignore', category=UserWarning,
                        module='pyannote.audio.utils.reproducibility')
warnings.filterwarnings('ignore', category=FutureWarning,
                        module='transformers.models.whisper.generation_whisper')

def load_and_resample(path: str, target_sr: int = 16000):
    audio, sr = librosa.load(path, sr=None)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio, target_sr

def split_audio_fixed(audio_path: str, output_dir: str, segment_length: int = 30,
                      trim_start: float = 0.0, trim_end: float = 0.0) -> list[str]:
    os.makedirs(output_dir, exist_ok=True)
    waveform, sr = torchaudio.load(audio_path)
    if waveform.dim() > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    total = waveform.size(1)
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

def clean_text(text: str) -> str:
    text = re.sub(r"<\|.*?\|>", " ", text)
    return " ".join(text.split())

def has_speaker_tags(text: str) -> bool:
    return bool(re.search(r"\[S\d\]", text))

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

def main():
    parser = argparse.ArgumentParser(description="Split, diarize & transcribe with speaker tags")
    parser.add_argument('audio_path', nargs='?', help='Input audio file (ignored if --from_segments)')
    parser.add_argument('--output_dir', default='segments', help='Segments folder')
    parser.add_argument('--csv_path', default='transcriptions.csv', help='CSV output path')
    parser.add_argument('--segment_length', type=int, default=30, help='Segment seconds')
    parser.add_argument('--hf_token', required=True, help='HF token')
    parser.add_argument('--trim_start', type=float, default=0.0, help='Trim start sec')
    parser.add_argument('--trim_end', type=float, default=0.0, help='Trim end sec')
    parser.add_argument('--append_csv', action='store_true', help='Append CSV')
    parser.add_argument('--full_transcribe', action='store_true', help='Full then split')
    parser.add_argument('--from_segments', help='Path to folder with pre-segmented audio files (supports subfolders)')
    parser.add_argument('--include_original', action='store_true', help='Include original .txt transcription if exists')
    parser.add_argument('--skip_existing', action='store_true', help='Skip .wav files that already have .txt files')
    parser.add_argument('--add_diarization_to_existing', action='store_true', help='Add [Sx] tags to .txt files without them')
    parser.add_argument('--overwrite_txt', action='store_true', help='Allow overwriting existing .txt files')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        'openai/whisper-large-v3', torch_dtype=torch.float16
    ).to(device)
    proc = AutoProcessor.from_pretrained('openai/whisper-large-v3')
    proc.tokenizer.pad_token_id = proc.tokenizer.eos_token_id
    asr_pipe = hf_pipeline(
        'automatic-speech-recognition',
        model=model,
        tokenizer=proc.tokenizer,
        feature_extractor=proc.feature_extractor,
        device=0 if device.type=='cuda' else -1,
        chunk_length_s=args.segment_length,
        return_timestamps=True,
        generate_kwargs={
            'task': 'transcribe', 'language': 'pt',
            'num_beams': 5, 'early_stopping': True,
            'no_repeat_ngram_size': 2, 'forced_decoder_ids': None
        }
    )

    diar_pipe = DiarizationPipeline.from_pretrained(
        'pyannote/speaker-diarization-3.1', use_auth_token=args.hf_token
    ).to(device)

    if args.from_segments:
        print(f"🔹 Searching for segments in: {args.from_segments}")
        all_segments = glob.glob(os.path.join(args.from_segments, '**', '*.wav'), recursive=True)
        seg_times = [(seg, None, None) for seg in sorted(all_segments)]
    else:
        if args.full_transcribe:
            print("🔹 Full ASR & diarization...")
            audio_np, sr = load_and_resample(args.audio_path)
            tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            torchaudio.save(tmp.name, torch.from_numpy(audio_np).unsqueeze(0), sr)
            full_out = asr_pipe(tmp.name, return_timestamps=True)
            tmp.close(); os.unlink(tmp.name)
            diar_full = diar_pipe(args.audio_path)
            full_turns = list(diar_full.itertracks(yield_label=True))

        print("🔹 Splitting audio...")
        segments = split_audio_fixed(
            args.audio_path, args.output_dir,
            args.segment_length, args.trim_start, args.trim_end
        )
        seg_times = [(seg, None, None) for seg in segments]

    print(f"🔹 {len(seg_times)} segments to process.")

    mode = 'a' if args.append_csv and os.path.exists(args.csv_path) else 'w'
    if mode == 'w':
        with open(args.csv_path, 'w', encoding='utf-8') as f:
            header = ['audio', 'text']
            if args.include_original:
                header.append('original_text')
            f.write('|'.join(header) + '\n')

    for seg, st, en in tqdm(seg_times, desc='Segments'):
        base = os.path.splitext(seg)[0]
        txt_file = base + '.txt'
        original = None
        tr = None

        has_txt = os.path.isfile(txt_file)
        if has_txt:
            with open(txt_file, 'r', encoding='utf-8') as f:
                original = f.read().strip()

        if args.skip_existing and has_txt:
            continue

        if args.add_diarization_to_existing and has_txt and not has_speaker_tags(original or ''):
            print(f"🔄 Adding diarization to {os.path.basename(txt_file)} (will not overwrite)")
            tr = process_segment(seg, asr_pipe, diar_pipe)
            if args.overwrite_txt:
                with open(txt_file, 'w', encoding='utf-8') as f:
                    f.write(tr)

        if tr is None:
            tr = process_segment(seg, asr_pipe, diar_pipe)
            if not has_txt or args.overwrite_txt:
                with open(txt_file, 'w', encoding='utf-8') as f:
                    f.write(tr)

        print(f"{os.path.basename(seg)} -> Transcription:\n{tr}\n")
        row = [seg, tr]
        if args.include_original:
            row.append(original or '')
            print(f"Original:\n{tr}\n")
        with open(args.csv_path, 'a', encoding='utf-8') as f:
            f.write('|'.join(row).replace('\n', ' ') + '\n')

    print(f"✅ All transcriptions saved incrementally to: {args.csv_path}")

if __name__ == '__main__':
    main()

