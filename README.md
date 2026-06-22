# RL Trading Agent Exploration

**Author:** Sophie Zhang

A tabular Q-learning agent trained to make daily long/flat/short trading decisions on SPY (S&P 500 ETF) using discrete technical indicators as state features.

---

## Overview

This project applies reinforcement learning to algorithmic trading. A Q-learning agent observes a compact 18-state representation of market conditions, chooses a daily position (long, flat, or short), and receives a reward proportional to next-day returns. The agent is trained on SPY data from 2018–2022 and evaluated out-of-sample on 2023–2024.

**Key result:** The best-tuned agent (α = 0.01) achieves a Sharpe ratio of **0.856** and total return of **21.52%** on the test set, but does not beat the buy-and-hold benchmark (58.82% return, Sharpe 1.883), primarily due to the strong bull-market trend and information loss from discretizing a continuous state space.

---

## Repository Structure

```
ieor198_finalproj.ipynb   # Main notebook (data → features → training → evaluation)
features_overview.png     # SPY price, RSI, and 5-day return plots
training_convergence.png  # Q-learning reward curve across 500 episodes
equity_curves.png         # In-sample and out-of-sample portfolio vs. buy-and-hold
position_distribution.png # Agent position breakdown (short/flat/long) on test set
```

---

## Methodology

### State Space (18 states)

Three discrete features are combined into a single integer state index:

| Feature | Description | Bins |
|---|---|---|
| `rsi_bin` | 14-day RSI | Oversold (<35) / Neutral / Overbought (>65) |
| `ma_signal` | SMA10 vs. SMA50 crossover | 0 = downtrend, 1 = uptrend |
| `ret_bin` | 5-day return | Negative (<−1%) / Flat / Positive (>+1%) |

3 × 2 × 3 = **18 states** — small enough for a tabular approach.

### Q-Learning

The agent maintains a Q-table Q(s, a) and updates it via the Bellman equation:

```
Q(s, a) ← Q(s, a) + α [ r + γ · max_a' Q(s', a') − Q(s, a) ]
```

- **Actions:** Short (−1), Flat (0), Long (+1)
- **Reward:** `position × next_day_return`
- **Policy:** ε-greedy during training, greedy (argmax) at evaluation

### Hyperparameters

| Parameter | Value |
|---|---|
| Learning rate α | 0.01 (best) / swept over [0.01, 0.05, 0.1, 0.3, 0.5] |
| Discount factor γ | 0.99 |
| Episodes | 500 |
| ε start / end / decay | 1.0 / 0.05 / 0.995 |

---

## Results

### Out-of-Sample Performance (2023–2024)

| Metric | RL Agent | Buy & Hold |
|---|---|---|
| Total Return | 21.52% | 58.82% |
| Sharpe Ratio | 0.856 | 1.883 |

### Ablation — Varying Learning Rate α

Lower learning rates generalized significantly better. α = 0.30 produced a Sharpe of −1.420, suggesting aggressive updates cause the agent to overwrite previously learned behavior.

### Ablation — Varying Ticker

The agent was also evaluated on QQQ, GLD, and BTC-USD using the same state features and training scheme.

---

## Setup & Requirements

```bash
pip install yfinance pandas numpy matplotlib seaborn
```

Data is downloaded automatically via `yfinance` — no manual downloads needed.

**Python 3.8+** is required. Run all cells top-to-bottom in order; the notebook is fully self-contained.

---

## Limitations

- Continuous indicators (RSI, momentum) are binned coarsely into 2–3 levels, discarding resolution.
- Tabular Q-learning cannot generalize to unseen state combinations the way a DQN would.
- The backtest does not account for transaction costs, slippage, or bid-ask spreads.
- Training data spans a predominantly bullish regime, biasing the learned policy toward long positions.

---

## Future Work

- Replace the tabular agent with a **Deep Q-Network (DQN)** using raw price features or OHLCV history as input.
- Train across multiple market regimes (e.g., including 2008 and 2022 bear markets) for a more robust policy.
- Extend to a **multi-asset agent** (SPY + bonds + gold) to exploit cross-asset correlation for better risk-adjusted returns.

---

## References

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518, 529–533.
- Yasin, A. S., & Gill, P. S. (2024). Reinforcement learning framework for quantitative trading. arXiv.
