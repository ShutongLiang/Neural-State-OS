# Released artifacts

Each model size has a matched writer and reader contract:

- `writer/direct44_raw_three_layer_writer.npz` stores axes, categories,
  early/middle/late layers, directions, and layer gains;
- `reader/direct44_postwriter_live_affect_reader.npz` stores post-writer layers,
  projection directions, endpoint scales, and controlled validity scores;
- `reader/direct44_postwriter_response_calibration.npz` stores the measured
  writer-to-reader response matrix;
- `reader/direct44_postwriter_attribution_decoder.npz` stores the regularized
  inverse used only for writer-source attribution.

The 12B and 31B directories are not interchangeable. Base-model checkpoints,
raw activation captures, and generated calibration records are not included.
