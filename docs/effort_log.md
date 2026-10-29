# Effort log

What each architecture cost to build, the half of the cost table no code can
measure. One row per work session, written the same day. Trials are not logged
here: their time and API cost are columns in `data/evaluation/<arch>_tier<n>.csv`.

- **Arch:** A, B, or Shared (helps both: calibration, envelope, harness, cell)
- **Kind:** code, prompt, bench, demos, training, debugging
- **Demos:** episodes recorded that session, kept ones only
- **GPU h:** GPU-hours of fine-tuning that session

| Date | Arch | Kind | Hours | Demos | GPU h | What |
|------|------|------|-------|-------|-------|------|
