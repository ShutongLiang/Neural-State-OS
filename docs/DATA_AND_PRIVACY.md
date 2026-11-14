# Data, privacy, and release policy

## Included

- source code;
- machine-readable 44-axis taxonomy;
- small model-matched writer and reader artifacts;
- assembled response calibrations and attribution decoders;
- aggregate validation metadata and publication figures.

## Excluded

- Gemma checkpoints and tokenizer files;
- third-party datasets;
- generated calibration text and raw batches;
- full layerwise activation captures;
- private conversations, histories, logs, and user profiles;
- API keys, tunnel addresses, SSH material, and machine-specific paths.

The local demo keeps conversation history in server memory by `session_id`.
`POST /api/reset` removes that session history and clears active residual
controls. The reference server does not implement accounts, authentication, or
encrypted remote transport; bind it to `127.0.0.1` unless you supply those
controls yourself.

The public NPZ files are derived control/readout artifacts and do not contain a
copy of the base-model checkpoint. Users remain responsible for the license and
policy requirements of any checkpoint they load.
