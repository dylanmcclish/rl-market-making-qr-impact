#!/bin/bash
cd ~/rl_markets/cpp || exit 1
echo "=== QR training started at $(date) ==="
./train --mode train --episodes 1000 --workers 4 --book qr --events 17600000
echo "=== QR training finished at $(date) ==="
echo "=== ZI training started at $(date) ==="
./train --mode train --episodes 1000 --workers 4 --book zi --events 17600000
echo "=== ZI training finished at $(date) ==="
echo "=== ALL DONE at $(date) ==="
