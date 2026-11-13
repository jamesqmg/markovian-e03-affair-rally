# Is infidelity priced in?

An event study of what happens to a company's stock when its CEO is pushed out over sexual or relationship misconduct, plus a
synthetic-control look at Nvidia in November 2026. Companion code for the Markovian Kickflip video "Is infidelity priced in?".

**Result in one line:** across 24 CEO conduct dismissals the average two-day abnormal return is about −1.9% (19 of 24 fell), a borderline
result that agrees with two published studies; after a Holm correction across 40 secondary tests only abnormal trading volume survives.
Nvidia's primary-window gap to its synthetic twin is about −17%, rank 1 of 27 against placebo stocks. This is an explanation, not a trading strategy.

- `prereg.md`: the pre-registration (locked 2026-11-12) with every later change logged as a dated erratum. Internal file paths and the
  name of the viral reel's creator are redacted in this public copy; nothing else is edited.
- `E03_affair_rally.ipynb`: the walkthrough, with outputs (figures and tables)
- `e03/`: the code (`fetch`, `events`, `event_study`, `viral_test`, `synth`, `garch`, `bayes`, `calendar_time`, `placebo`, `card`)
- `tests/`: `pytest`, including a null-size check (no false effect on pure noise)
- `out/`: the result files the notebook reads. The Nvidia file has its daily price series removed (see Data).

```bash
uv sync
uv run python -m e03.fetch && uv run python -m e03.events && uv run python -m e03.run_arm_c
uv run python -m e03.viral_test && uv run python -m e03.synth --last 2026-11-12 && uv run python -m e03.card
uv run --group dev pytest
```

## Data

No raw vendor data is redistributed. `data/manifest.json` lists the source URLs and SHA-256 checksums of what `e03.fetch` downloads:
Ken French Data Library factors; Gentry et al. (2021) CEO Dismissal Database (Zenodo, CC BY 4.0); Yahoo Finance and Tiingo prices (set
`TIINGO_API_KEY` in your environment); SEC EDGAR. Re-running pulls fresh prices, so numbers can differ slightly from the published run.
The card step reads one more local file, the chip peers' Nov 3 to 11 closes, if you have it at `data/raw/chip_raw_closes_2026-11-03_11.csv`;
otherwise it keeps the Nov 9 numbers from `out/results_card.json`.

## Limits (read these)

Primary p ≈ 0.09 and the interval only just excludes zero; the rank and sign tests reject. n = 24. The Nvidia gap bundles every news item
in the window (the spy allegation, the board statement), so it is not the affair alone. The "disclosure vagueness" split is post-hoc.
Details and the full list of tests are in `prereg.md` and the notebook.

## License

MIT (see `LICENSE`).
