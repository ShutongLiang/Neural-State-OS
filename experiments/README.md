# Open-domain experiment profiles

`profiles_31b.json` freezes the Gemma 4 31B state compositions used in the
public behavioral reports. The profiles are research probes, not recommended
production presets.

All reported runs used direct generation, the standard first-person Neural
State OS persona, `all_tokens` residual writing, raw vector composition, and a
mix headroom of `1.35`. Each independent comparison used a fresh session unless
the protocol explicitly studied multi-turn conditioning or state switching.

The associated results and claim boundaries are documented in:

- [`docs/EXPERIMENTS.md`](../docs/EXPERIMENTS.md)
- [`docs/MATURE_STATE_APPENDIX.md`](../docs/MATURE_STATE_APPENDIX.md)

The repository does not include model checkpoints, private conversations, or
generated high-risk operational content.
