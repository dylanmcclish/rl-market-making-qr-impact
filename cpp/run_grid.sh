#!/bin/bash
cd ~/rl_markets/cpp || exit 1

# Wait for the already-running QR + consolidated-agent job (1000 episodes,
# eps_t=1000, launched earlier) to finish on its own -- untouched. NOT one
# of the 4 grid cells below (different training budget -- would confound
# the comparison), kept purely as a separate bonus/robustness data point
# (does the pattern hold at both 400 and 1000 episodes?).
echo "=== waiting for existing QR/consolidated (1000ep) job (PID 6386) to finish at $(date) ==="
while kill -0 6386 2>/dev/null; do
    sleep 60
done
echo "=== existing QR/consolidated (1000ep) job finished at $(date) ==="

# Core grid: basic vs consolidated agent x ZI vs QR simulator -- all 4
# cells at the SAME budget (400 episodes, eps_t=400) for a clean,
# confound-free comparison. eps_t=400 reproduces the identical epsilon
# trajectory Table 2's eps_t=1000/1000-episode schedule would over its
# full run, just compressed to 400 episodes -- verified:
# eps(400; eps_t=400) == eps(1000; eps_t=1000) == 0.350.

echo "=== [1/4] QR + basic agent (400 episodes, eps_t=400) started at $(date) ==="
./train --mode train --episodes 400 --workers 4 --book qr --events 17600000 \
    --state basic --reward pnl --eps-t 400 --tag qr_basic
echo "=== [1/4] finished at $(date) ==="

echo "=== [2/4] QR + consolidated agent (400 episodes, eps_t=400) started at $(date) ==="
./train --mode train --episodes 400 --workers 4 --book qr --events 17600000 \
    --state full --reward asym --eps-t 400 --tag qr_consolidated_400ep
echo "=== [2/4] finished at $(date) ==="

echo "=== [3/4] ZI + basic agent (400 episodes, eps_t=400) started at $(date) ==="
./train --mode train --episodes 400 --workers 4 --book zi --events 17600000 \
    --state basic --reward pnl --eps-t 400 --tag zi_basic
echo "=== [3/4] finished at $(date) ==="

echo "=== [4/4] ZI + consolidated agent (400 episodes, eps_t=400) started at $(date) ==="
./train --mode train --episodes 400 --workers 4 --book zi --events 17600000 \
    --state full --reward asym --eps-t 400 --tag zi_consolidated
echo "=== [4/4] finished at $(date) ==="

echo "=== ALL GRID RUNS DONE at $(date) ==="
