# USS browser demo

Static, browser-only source separation for SJTU AudioCC Lab. Serve this directory over HTTP(S), or use GitHub Pages with this repository's root as the publishing folder. No build or inference server is required. Audio remains on the visitor's device.

## Current model

- AudioCC checkpoint: `/exp/bohan.liu/urgent2026_challenge/exp/moe_large_event/last.ckpt`.
- EMA weights, FP32 ONNX; 7 layers, 48 channels, 100 Mel bands, 3 sources.
- Each F/T module has 12 event experts (FFN1) and 12 TF experts (FFN2), top-1 routing. GroupNorm is enabled; this checkpoint has no shared expert branch.
- 16 kHz audio, 960-point periodic Hann STFT, 480-sample hop. Dynamic 3-30-second input, processed in one pass.
- `models/manifest.json` records checkpoint/model hashes, exact architecture and tensor contract. The worker checks the model SHA-256 and uses a hash-specific cache key.

Event labels follow `baseline_code/event_routing.py` in the training repository:

| ID | Category |
| --- | --- |
| E0 | English (LibriTTS-R speech) |
| E1 | Mandarin (AISHELL-1 speech) |
| E2 | Mandarin (AISHELL-3 speech) |
| E3 | Other speech |
| E4 | Singing / vocals |
| E5 | Other human sounds |
| E6 | Birds |
| E7 | Other animals |
| E8 | String instruments |
| E9 | Other music |
| E10 | Transport |
| E11 | Nature / other residual sounds |

These are event-supervision categories, not sound-class confidence scores. TF expert IDs have no event-class labels. The bottom strip shows per-frame routing from F modules; the right strip shows per-band routing from T modules. Both display actual outputs of the current browser inference.

## Preserved behavior

The default HIVE mixture is unchanged (SHA-256 `3e6817450f59a99e798d615f33b9b16a14f478afda52742f577cfa07d9ef641a`). Upload, microphone recording, full-duration selection, minimum 3-second range selection, synchronized comparison and WAV downloads remain available. Results appear above the input/routing workspace when ready. Separated sources use constant-gain RMS normalization with peak protection; the original mixture is unchanged.

ONNX Runtime Web uses browser CPU/WASM. Cross-origin isolated hosts can use up to four threads. Hosts without isolation headers, including ordinary GitHub Pages, use one thread.

## Re-export

Use the original training repository and its Python environment, with compatible PyTorch, ONNX, ONNX Runtime and soundfile installed:

```sh
python tools/export_model.py \
  --repo /path/to/urgent2026_challenge \
  --checkpoint /path/to/moe_large_event/last.ckpt \
  --audio demos/mixture.wav \
  --output /path/to/export-output
```

The exporter reads the checkpoint configuration, strictly loads the weights, applies EMA, exports sparse top-1 expert dispatch and dynamic duration, and validates 3/6/10/7.13/30-second cases against the original model. Copy the verified `uss.onnx` and `manifest.json` into `models/`. Validation reports are included under `validation/`.

Floating-point implementations can choose different experts when top probabilities tie. Such differences are explicitly counted in the export report with their reference probability gaps; displayed routes always come from the executed graph.

## Validation for this release

- Original PyTorch versus ONNX: 3, 6, 7.13 (offset 1.17), 10 and 30 seconds. Spectral SNR is 97-132 dB; all event selections agree. One of the 700 band selections in the 3-second case differs at an exact probability tie; all other tested TF selections agree. See [export report](validation/export.json).
- Edge WASM: all five durations pass waveform comparison and exact event/TF route comparison against native ONNX Runtime. Upload length limits, interval controls, microphone recording, cancellation and responsive layouts pass. See [browser report](validation/browser-duration.json).
- Results placement, synchronized playback, RMS balancing, peak protection and matching WAV downloads pass. See [results report](validation/results-presentation.json).

- Without cross-origin isolation (GitHub Pages conditions), a full 30-second input completes with one WASM thread and correct output lengths/event routing. This machine measured about 92 seconds for inference; browser and hardware performance varies. See [single-thread report](validation/single-thread-30s.json).
