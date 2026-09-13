#!/bin/bash
# Real 2x2 training grid: basic vs consolidated agent x ZI vs QR simulator,
# all 4 cells matched at 400 episodes, eps_t=400, full 17,600,000-event
# episodes, run sequentially using all 12 cores per cell to minimize wall
# time. Supersedes the earlier smoke-test artifacts (archived to
# smoketest_backup/) which were only 4-8 episodes x 20-30k events and were
# mistakenly treated as the real training runs.
set -e
cd "$(dirname "$0")"

echo "=== [1/4] QR + basic agent started at $(date) ==="
./train --mode train --episodes 400 --workers 12 --book qr --events 17600000 \
    --state basic --reward pnl --eps-t 400 --tag qr_basic
echo "=== [1/4] finished at $(date) ==="

echo "=== [2/4] ZI + basic agent started at $(date) ==="
./train --mode train --episodes 400 --workers 12 --book zi --events 17600000 \
    --state basic --reward pnl --eps-t 400 --tag zi_basic
echo "=== [2/4] finished at $(date) ==="

echo "=== [3/4] QR + consolidated agent started at $(date) ==="
./train --mode train --episodes 400 --workers 12 --book qr --events 17600000 \
    --state full --reward asym --eps-t 400 --tag qr_consolidated
echo "=== [3/4] finished at $(date) ==="

echo "=== [4/4] ZI + consolidated agent started at $(date) ==="
./train --mode train --episodes 400 --workers 12 --book zi --events 17600000 \
    --state full --reward asym --eps-t 400 --tag zi_consolidated
echo "=== [4/4] finished at $(date) ==="

echo "=== ALL 4 GRID CELLS DONE at $(date) ==="
