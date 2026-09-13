"""
Generates figures/market_environment_timeseries.png for main.tex --
mid-price, best bid, and best ask over a representative excerpt of a
testing episode in the Queue-Reactive simulator.

Uses the exact same book construction as train.py's make_qr_book() (same
calibrated params, same empirical trade-size sampler) and the same
env.reset() call as the paper's evaluation runs (mid_price=100.0,
book_burn_in=3000, warmup_steps=200), but with an eval seed disjoint from
training. Runs the fixed theta=2 benchmark policy (one of the paper's own
evaluation policies) for N_EVENTS steps and records the market's own best
bid/ask/mid at each step -- these are the background book's prices, not
the agent's quotes, and are unaffected by which policy is driving the
agent (Sec. Method: "the agent's own resting orders never impact the
market"), so any of the paper's evaluation policies would show the same
market process for a given seed.

Only a few thousand events (not the full 5,000,000-event episode) are
plotted -- a full episode is illegible as a line plot and the point of
this figure is to show the qualitative shape of the process, matching
how Fig. 1-style order-book timeseries plots are usually presented.
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import MarketMakingEnv, ACTION_THETA
from market_qr import QueueReactiveBook, load_calibrated_params, load_trade_size_sampler
from agent_spooner import ORDER_SIZE, MIN_INV, MAX_INV

N_EVENTS = 8000
EVAL_SEED = 1
THETA_ACTION = ACTION_THETA.index((2, 2))  # the fixed theta=2 benchmark

rng = np.random.default_rng(EVAL_SEED)
params = load_calibrated_params()
sampler = load_trade_size_sampler()
book = QueueReactiveBook(params, mo_size_sampler=sampler, rng=rng)
env = MarketMakingEnv(book, order_size=ORDER_SIZE, min_inv=MIN_INV, max_inv=MAX_INV,
                      reward='pnl')
env.reset(mid_price=100.0, book_burn_in=3000, warmup_steps=200)

mids, bids, asks = [], [], []
for _ in range(N_EVENTS):
    _, _, _, info = env.step(THETA_ACTION)
    snap = book.snapshot()
    mids.append(info['mid'])
    bids.append(snap['bid_prices'][0])
    asks.append(snap['ask_prices'][0])

mids, bids, asks = np.array(mids), np.array(bids), np.array(asks)
x = np.arange(N_EVENTS)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x, asks, label='Best ask', color='tab:red', linewidth=0.8)
ax.plot(x, mids, label='Mid-price', color='tab:blue', linewidth=1.0)
ax.plot(x, bids, label='Best bid', color='tab:green', linewidth=0.8)
ax.set_xlabel('Event index')
ax.set_ylabel('Price ($)')
ax.set_title(f'QR simulator: mid-price, best bid, and best ask ({N_EVENTS:,}-event excerpt)')
ax.legend(loc='upper right', fontsize=9)
fig.tight_layout()

out_path = os.path.join('paper', 'figures', 'market_environment_timeseries.png')
os.makedirs(os.path.dirname(out_path), exist_ok=True)
fig.savefig(out_path, dpi=200)
print(f'wrote {out_path}')
print(f'mid range: [{mids.min():.3f}, {mids.max():.3f}], '
      f'mean spread: {(asks - bids).mean():.4f}')
