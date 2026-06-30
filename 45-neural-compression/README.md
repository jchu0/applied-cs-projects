# Neural Compression Engine

A learned image compression system implementing VAE-based hyperprior codecs in the style of
Ballé et al. and DeepMind research: end-to-end rate-distortion optimization, arithmetic coding,
and Generalized Divisive Normalization (GDN) transforms.

> **Status:** reference implementation / teaching scaffold built to a strong blueprint — not production-grade. See [../PROJECTS_STATUS.md](../PROJECTS_STATUS.md) and the [2026-06 audit](../../docs/AUDIT_2026-06_public-readiness.md).

> **Concepts covered:** §03 ml-engineering `02-deep-learning`; §04 ai-engineering `07-custom-models`

---

## What's real vs simulated

The core codec architecture, entropy model math (Gaussian likelihood, factorized prior,
rate-distortion loss), arithmetic coder, and training loop are fully implemented.
Two caveats:

- **No pretrained weights are shipped.** `NeuralCompressionCodec` starts from random
  initialization; you must train it yourself before compress/decompress produce meaningful
  output. The API reference examples that call `load_state_dict` assume weights you
  provide externally.
- **`MultiRateCodec.compress` / `.decompress` raise `NotImplementedError`** — those paths
  stub out to `NeuralCompressionCodec` for now and are not end-to-end wired.

Everything else — GDN transforms, hyperprior hyper-encoder/decoder, arithmetic coding,
multi-rate gain units, model-weight compressor, perceptual losses, training scheduler —
is real, runnable code.

---

## Layout

```
src/neural_compression/
    transforms.py    — AnalysisTransform, SynthesisTransform, GDN
    entropy.py       — EntropyModel, HyperAnalysis, HyperSynthesis, FactorizedPrior, ArithmeticCoder
    codecs.py        — NeuralCompressionCodec (full compress / decompress pipeline)
    multirate.py     — MultiRateCodec, ScaleAdaptiveCodec (compress stub — see above)
    losses.py        — RateDistortionLoss, PerceptualLoss (LPIPS-style via VGG)
    training.py      — Trainer, MultiRateTrainer, checkpoint save/restore
    data.py          — Dataset wrappers, augmentations (random crop, flip)

tests/               — 143 tests across 7 files (transforms, entropy, codecs, losses,
                       multirate, training, data)
BLUEPRINT.md         — full architecture design with code sketches and phase plan
```

---

## Build & run

```bash
conda activate dev
cd 06-real-world-projects/45-neural-compression
pip install -e ".[dev]"
pytest tests/ -v
```

To include the FastAPI compression server and Pillow image I/O:

```bash
pip install -e ".[full]"
```

---

## References

- Ballé et al., *Variational Image Compression with a Scale Hyperprior* (2018)
- Minnen et al., *Joint Autoregressive and Hierarchical Priors for Learned Image Compression* (2018)
- Mentzer et al., *Practical Full Resolution Learned Lossless Image Compression* (2019)
