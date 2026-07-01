# DeepPath Lab Tasks

## Status: Mature Learning Baseline (Modules 01–15)

All 15 modules ship with: original notes, from_scratch, reproduce, experiments, report, and Chinese README.

| Layer | Artifact | Status |
|-------|----------|--------|
| Modules 01–15 | notes + code + report | done |
| Learning docs | getting-started, learning-guide, study-checklist | done |
| Navigation | modules/README.md | done |
| Verification | scripts/verify_all.py (15/15 smoke) | done |
| Unit tests | tests/ + scripts/run_tests.py | done |
| Helpers | scripts/run_module.py, lib/ | done |

## Definition Of Meaningful Progress

Each module includes: original notes, from_scratch artifact, reproduce baseline, experiment/diagnostic, report with concrete observations.

## How To Verify

```bash
python3 scripts/verify_all.py
python3 scripts/run_tests.py
```

## Future Work

- Expand unit test coverage to more from_scratch files (CNN, attention, skip-gram)
- Deepen experiments with larger datasets (MovieLens, full sentiment corpora)
- Optional Track G extensions (diffusion, GANs, etc.)
- Add visualization gallery under docs/
