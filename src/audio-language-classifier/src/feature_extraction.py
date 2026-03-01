"""
Feature extraction using Facebook MMS-LID (Massively Multilingual Speech -
Language Identification) model.

The MMS-LID model is a wav2vec2 model specifically fine-tuned for language
identification across 256 languages.  We extract the *projector* output
(compact, speaker-invariant language embeddings) rather than the raw
hidden states, because the projector was trained to maximise language
discrimination.

Produces a (EMBEDDING_DIM,) language embedding per audio segment.
Supports batched extraction with checkpointing.
"""

import os
import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor

from src.config.settings import (
    WAV2VEC_MODEL, SAMPLE_RATE, BATCH_SIZE, DEVICE,
    EMBEDDING_DIM, EMBEDDINGS_DIR,
)


class Wav2VecExtractor:
    """Extracts language embeddings from audio using MMS-LID model.

    Uses the projector output of the fine-tuned MMS Language-ID model
    to produce speaker-invariant, language-discriminative embeddings.
    """

    def __init__(self, model_name: str = WAV2VEC_MODEL, device: str = DEVICE):
        self.device = device
        print(f"Loading {model_name} on {device}...")

        self.processor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
            model_name
        ).to(device)
        self.model.eval()

        # Prefer the projector for compact language embeddings
        self.has_projector = hasattr(self.model, "projector")
        if self.has_projector:
            self.embed_dim = self.model.config.classifier_proj_size
            print(f"  Using projector → {self.embed_dim}-dim language embeddings")
        else:
            self.embed_dim = self.model.config.hidden_size
            print(f"  Using hidden states → {self.embed_dim}-dim embeddings")

        total = sum(p.numel() for p in self.model.parameters())
        print(f"  Model loaded: {total:,} parameters")

    # ── Core embedding logic ──────────────────────────────────────

    def _get_embeddings(self, input_values: torch.Tensor,
                        attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Return (B, embed_dim) language embeddings."""
        with torch.no_grad():
            backbone_out = self.model.wav2vec2(
                input_values, attention_mask=attention_mask
            )
            hidden = backbone_out.last_hidden_state        # (B, T, H)

            if self.has_projector:
                projected = self.model.projector(hidden)   # (B, T, proj_dim)
                return projected.mean(dim=1)               # (B, proj_dim)
            return hidden.mean(dim=1)                      # (B, H)

    # ── Public API ────────────────────────────────────────────────

    def extract_single(self, waveform: np.ndarray) -> np.ndarray:
        """Extract embedding for a single waveform (1-D float32 array).

        Returns a (embed_dim,) numpy array.
        """
        inputs = self.processor(
            [waveform], sampling_rate=SAMPLE_RATE,
            return_tensors="pt", padding=True,
        )
        iv = inputs.input_values.to(self.device)
        am = inputs.get("attention_mask")
        if am is not None:
            am = am.to(self.device)

        return self._get_embeddings(iv, am).cpu().numpy().squeeze()

    def extract_batch(self, waveforms: list[np.ndarray]) -> np.ndarray:
        """Extract embeddings for a batch of waveforms.

        Returns an (N, embed_dim) numpy array.
        """
        inputs = self.processor(
            waveforms, sampling_rate=SAMPLE_RATE,
            return_tensors="pt", padding=True,
        )
        iv = inputs.input_values.to(self.device)
        am = inputs.get("attention_mask")
        if am is not None:
            am = am.to(self.device)

        return self._get_embeddings(iv, am).cpu().numpy()

    def extract_from_npy_files(self, npy_paths: list[str],
                               batch_size: int = BATCH_SIZE,
                               cache_path: str | None = None) -> np.ndarray:
        """Extract embeddings from a list of .npy file paths.

        Supports checkpointing: if `cache_path` is given and a partial
        result exists, resumes from where it left off.

        Returns an (N, embed_dim) numpy array.
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
                all_embeds.append(np.zeros((len(batch_paths), self.embed_dim)))

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
    """Extract a language embedding from a single waveform."""
    return get_extractor().extract_single(waveform)