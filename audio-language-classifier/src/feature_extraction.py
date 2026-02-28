"""
Feature extraction using wav2vec2-xls-r-300m.

Produces a (1024,) mean-pooled embedding per audio segment.
Supports batched extraction with GPU acceleration.
"""

import os
import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import Wav2Vec2Model, Wav2Vec2Processor

from src.config.settings import (
    WAV2VEC_MODEL, SAMPLE_RATE, BATCH_SIZE, DEVICE,
    EMBEDDING_DIM, EMBEDDINGS_DIR,
)


class Wav2VecExtractor:
    """Extracts mean-pooled wav2vec2 embeddings from audio arrays."""

    def __init__(self, model_name: str = WAV2VEC_MODEL, device: str = DEVICE):
        self.device = device
        print(f"Loading {model_name} on {device}...")
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name).to(device)
        self.model.eval()
        total = sum(p.numel() for p in self.model.parameters())
        print(f"  Model loaded: {total:,} parameters")

    def extract_single(self, waveform: np.ndarray) -> np.ndarray:
        """Extract embedding for a single waveform (1-D float32 array).

        Returns a (EMBEDDING_DIM,) numpy array.
        """
        inputs = self.processor(
            [waveform], sampling_rate=SAMPLE_RATE,
            return_tensors="pt", padding=True,
        )
        input_values = inputs.input_values.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_values)
            embedding = outputs.last_hidden_state.mean(dim=1)  # (1, 1024)

        return embedding.cpu().numpy().squeeze()

    def extract_batch(self, waveforms: list[np.ndarray]) -> np.ndarray:
        """Extract embeddings for a batch of waveforms.

        Returns an (N, EMBEDDING_DIM) numpy array.
        """
        inputs = self.processor(
            waveforms, sampling_rate=SAMPLE_RATE,
            return_tensors="pt", padding=True,
        )
        input_values = inputs.input_values.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_values)
            embeddings = outputs.last_hidden_state.mean(dim=1)

        return embeddings.cpu().numpy()

    def extract_from_npy_files(self, npy_paths: list[str],
                               batch_size: int = BATCH_SIZE,
                               cache_path: str | None = None) -> np.ndarray:
        """Extract embeddings from a list of .npy file paths.

        Supports checkpointing: if `cache_path` is given and a partial
        result exists, resumes from where it left off.

        Returns an (N, EMBEDDING_DIM) numpy array.
        """
        # Resume support
        start_idx = 0
        partial_embeds = []

        ckpt_path = (cache_path.replace(".npy", "_partial.npy")
                     if cache_path else None)

        if cache_path and os.path.exists(cache_path):
            print(f"  Loading cached embeddings from {cache_path}")
            return np.load(cache_path)

        if ckpt_path and os.path.exists(ckpt_path):
            prev = np.load(ckpt_path)
            partial_embeds.append(prev)
            start_idx = prev.shape[0]
            print(f"  Resuming from sample {start_idx}")

        remaining = npy_paths[start_idx:]
        all_embeds = list(partial_embeds)

        for i in tqdm(range(0, len(remaining), batch_size),
                      desc="Extracting embeddings"):
            batch_paths = remaining[i: i + batch_size]
            waveforms = [np.load(p).astype(np.float32) for p in batch_paths]

            try:
                embeds = self.extract_batch(waveforms)
                all_embeds.append(embeds)
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    torch.cuda.empty_cache()
                    raise RuntimeError(
                        f"OOM at batch {i}. Reduce batch_size (currently {batch_size})."
                    ) from e
                print(f"  ⚠ Batch {i} error: {e}")
                all_embeds.append(np.zeros((len(batch_paths), EMBEDDING_DIM)))

            # Checkpoint every 100 batches
            if ckpt_path and ((i // batch_size + 1) % 100 == 0):
                partial = np.vstack(all_embeds)
                np.save(ckpt_path, partial)

        result = np.vstack(all_embeds)

        if cache_path:
            np.save(cache_path, result)
            # Cleanup checkpoint
            if ckpt_path and os.path.exists(ckpt_path):
                os.remove(ckpt_path)

        return result


# ── Convenience function ──────────────────────────────────────────

_extractor: Wav2VecExtractor | None = None


def get_extractor() -> Wav2VecExtractor:
    """Lazy-loaded singleton extractor to avoid reloading the model."""
    global _extractor
    if _extractor is None:
        _extractor = Wav2VecExtractor()
    return _extractor


def extract_features(waveform: np.ndarray) -> np.ndarray:
    """Extract a (1024,) embedding from a single waveform."""
    return get_extractor().extract_single(waveform)