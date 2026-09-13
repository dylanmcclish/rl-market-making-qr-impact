# Market Making via Reinforcement Learning Under Simulated Market Impact

Code accompanying the paper *"Market Making via Reinforcement Learning Under
Simulated Market Impact: A Queue-Reactive Reassessment of Spooner et al."*
(`paper/main.tex` / `paper/main.pdf`).

Spooner et al.'s market-making RL agents (basic and consolidated) are
retrained and evaluated in two simulated limit-order-book environments
instead of Spooner's original historical-data replay: a Zero-Intelligence
(ZI) model as a control, and a Queue-Reactive (QR) model to introduce
market impact. Both simulators are calibrated on IEX DEEP data.

## Layout

- `agent_spooner.py`, `tile_coding.py`, `sparse_traces.py` — the SARSA(λ)
  agent (basic + consolidated configurations) and its tile-coding function
  approximator.
- `env.py`, `market_zi.py`, `market_qr.py` — the trading environment and the
  two order-book simulators.
- `train.py`, `test_env.py`, `test_agent.py`, `validate_simulators.py` —
  training entrypoint and test/validation suites.
- `daily_iex_collector.py`, `iex_calibrate.py`, `iex_stream_collect.py`,
  `DEPLOY.md`, `symbols.txt` — IEX DEEP data collection and simulator
  calibration; `DEPLOY.md` covers running the collector unattended on a
  free-tier VM.
- `cpp/` — a C++ reimplementation of the agent/environment used for the
  full training/evaluation grid (faster than the Python version for the
  full episode counts in the paper).
- `paper/` — the LaTeX source and PDF of the writeup.

## Note on data

Raw IEX DEEP archives and trained model weights are not included in this
repo (too large, and IEX's data isn't cleared for redistribution) — you'll
need to collect and calibrate your own via `daily_iex_collector.py` /
`iex_calibrate.py` if you want to reproduce the calibration end-to-end.
