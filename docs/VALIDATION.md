# Validation and claim boundary

Neural State OS is a research preview. The available evidence supports a
controlled activation-intervention and readout path; it does not support claims
about consciousness or general emotion recognition.

## Frozen 31B measurements

| Check | Result | Scope |
|---|---:|---|
| Best selected layer per axis, strict Spearman | mean 0.9276; min 0.8611 | Controlled cross-scene/style calibration and selection |
| Early-band mean | 0.8708 | Controlled calibration |
| Middle-band mean | 0.9201 | Controlled calibration |
| Late-band mean | 0.9047 | Controlled calibration |
| Post-writer reader strict Spearman | mean 0.8626; min 0.5079; max 0.9722 | Reader restricted to layers after the final writer |
| Controlled endpoint accuracy | mean 1.000 | Factorial endpoint comparison |
| Source-axis recovery | top-1 88/88; top-3 88/88 | Two neutral calibration contexts |

The source-recovery contexts were used in the calibration path. Despite a
legacy metadata key named `heldout_context_*`, this is not an untouched
natural-dialog test and is not reported as one here.

## Frozen 12B measurements

The 12B post-writer reader reports mean strict cross-scene/style Spearman
0.9034 and minimum 0.7897 under its corresponding controlled protocol. Its
source-attribution calibration recovers 87/88 top-1 and 88/88 top-3 calibration
interventions.

## What has been demonstrated

- model-specific residual directions can be composed and applied at selected
  transformer depths;
- paired passes isolate the immediate hidden-state change caused by a writer;
- post-writer readers preserve ordered intensity information in the controlled
  calibration domain;
- a measured response matrix can attribute controlled writes despite reader
  cross-talk;
- 12B and 31B runtimes enforce incompatible hidden-width contracts.

## What has not been demonstrated

- that a projection is a ground-truth feeling;
- that the model has human-like subjective experience;
- open-domain emotion recognition accuracy;
- independent population-level generalization across models, languages, or
  checkpoints;
- stable long-horizon state persistence without repeated control;
- safe closed-loop use with robots, wearables, or other actuators;
- the same quality on MoE architectures.

## Known evaluation limitations

1. Direction fitting, layer selection, and several summary values use the same
   controlled factorial corpus.
2. Scenes and styles are varied, but all records come from one designed
   calibration domain.
3. The 31B post-writer reader is weak on some axes because the final write is
   late and only a narrow set of later layers remains available.
4. The chat-domain decoder is a three-axis pilot, not a 44-axis natural-chat
   benchmark.
5. Greedy generation behavior is not a substitute for causal representation
   validation.

## Reproducing figures

The plotting script requires the full private research-artifact tree because
raw layerwise activation banks are intentionally not committed:

```bash
python scripts/render_research_figures.py \
  --backup-root /absolute/path/to/full-31b-research-artifacts \
  --output-dir docs/assets/results
```

Frozen numeric summaries are recorded in `docs/figure_metrics.json`.
