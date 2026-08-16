"""
Lightweight embedding client that calls the Hugging Face Inference API instead
of running sentence-transformers locally.

Why: running the model in-process on a low-CPU host (e.g. Render's free tier)
can turn a normally-fast embedding call into a 30-60+ second bottleneck,
because sentence-transformers' encode() is CPU-bound. Offloading to HF's
hosted endpoint moves that computation off your container entirely, at the
cost of a network round-trip (typically well under the CPU-bound time on a
throttled instance) and HF's own free-tier availability/rate limits.

Usage mirrors sentence-transformers' SentenceTransformer just enough to be a
drop-in replacement for the .encode() calls used in this project:

    embed_model = HFEmbedder()
    vec = embed_model.encode("some text")            # -> list[float]
    vecs = embed_model.encode(["a", "b"])             # -> list[list[float]]

Falls back to raising a clear error (not a silent wrong answer) if the HF
call fails, so callers can decide how to handle it (e.g. retry, 503, etc).
"""

import os
import time
import logging
from typing import List, Union

import requests

logger = logging.getLogger("medsmitra.hf_embedder")

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
HF_API_URL = f"https://router.huggingface.co/hf-inference/models/{HF_EMBED_MODEL}/pipeline/feature-extraction"

HF_TIMEOUT_SECONDS = float(os.getenv("HF_EMBED_TIMEOUT", "20"))
HF_MAX_RETRIES = int(os.getenv("HF_EMBED_MAX_RETRIES", "3"))


class HFEmbedderError(RuntimeError):
    pass


class _EncodedResult(list):
    """Thin wrapper so `.tolist()` works the same way numpy arrays do,
    since existing call sites do `embed_model.encode(x).tolist()`."""

    def tolist(self):
        return list(self)


class HFEmbedder:
    def __init__(self, model: str = HF_EMBED_MODEL, token: str = HF_API_TOKEN):
        if not token:
            raise HFEmbedderError(
                "HF_API_TOKEN is not set. Create a free token at "
                "https://huggingface.co/settings/tokens and add it to your env."
            )
        self.model = model
        self.url = f"https://router.huggingface.co/hf-inference/models/{model}/pipeline/feature-extraction"
        self._headers = {"Authorization": f"Bearer {token}"}

    def encode(
        self,
        texts: Union[str, List[str]],
        show_progress_bar: bool = False,  # accepted for interface compatibility, unused
        **_ignored,
    ):
        single_input = isinstance(texts, str)
        payload_texts = [texts] if single_input else list(texts)

        last_error = None
        for attempt in range(1, HF_MAX_RETRIES + 1):
            try:
                resp = requests.post(
                    self.url,
                    headers=self._headers,
                    json={"inputs": payload_texts, "options": {"wait_for_model": True}},
                    timeout=HF_TIMEOUT_SECONDS,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    # feature-extraction can return per-token vectors for some
                    # models; mean-pool if we get a 3D response instead of one
                    # vector per input sentence.
                    result = [self._mean_pool_if_needed(v) for v in data]
                    return _EncodedResult(result[0]) if single_input else result

                # Model is loading on HF's side — wait_for_model should handle
                # this, but fall back to a short sleep + retry just in case.
                if resp.status_code == 503:
                    logger.warning(
                        "HF embedding endpoint warming up (attempt %d/%d): %s",
                        attempt, HF_MAX_RETRIES, resp.text[:200],
                    )
                    time.sleep(2 * attempt)
                    continue

                raise HFEmbedderError(
                    f"HF Inference API returned {resp.status_code}: {resp.text[:300]}"
                )
            except requests.RequestException as exc:
                last_error = exc
                logger.warning(
                    "HF embedding request failed (attempt %d/%d): %s",
                    attempt, HF_MAX_RETRIES, exc,
                )
                time.sleep(1 * attempt)

        raise HFEmbedderError(f"HF Inference API failed after {HF_MAX_RETRIES} attempts: {last_error}")

    @staticmethod
    def _mean_pool_if_needed(vector):
        # Expected: List[float] (one vector per sentence).
        # Some models return List[List[float]] (per-token) instead — mean-pool.
        if vector and isinstance(vector[0], list):
            length = len(vector)
            dim = len(vector[0])
            summed = [0.0] * dim
            for token_vec in vector:
                for i, val in enumerate(token_vec):
                    summed[i] += val
            return [v / length for v in summed]
        return vector
