# Optiver — Trading at the Close: predicting closing-auction moves

Predicting the 60-second move of each Nasdaq stock, relative to a synthetic index, during the last ten minutes before the 16:00 closing auction. Data: Optiver's [Trading at the Close](https://www.kaggle.com/competitions/optiver-trading-at-the-close) Kaggle competition (200 stocks, 481 days, one snapshot every 10 seconds of the order book and the auction book). Metric: mean absolute error (MAE).

Everything is in one notebook, [`closing_optiver.ipynb`](closing_optiver.ipynb), run top to bottom.

## Protocol

- **What is known at prediction time:** every column of every stock at the current snapshot, earlier snapshots of the same day, and earlier days. The target and any later price are never used, and missing values are never filled from later rows.
- **The independent unit is the day.** Consecutive targets share 50 of their 60 seconds (lag-1 autocorrelation about 0.8), so a stock-day holds about 11 effective observations instead of 55.
- **Split by whole days, in time order:** train days 0–389, validation 390–434, test 435–480. Scaling statistics come from train only. The test set is opened once, at the end.
- **Metrics:** MAE, and the daily information coefficient (cross-sectional correlation of prediction and target, averaged per day) with its t-stat over days and the share of positive days.

## Hypotheses tested

Each was checked with the daily IC on train, then on validation.

1. **Auction imbalance pushes the price** (imbalance size / matched size, signed). Stronger in the last five minutes.
2. **Mean reversion** of recent returns, once the shared current price is removed.
3. **Volatility predicts the size of the move**, not its sign.
4. **The gap between the auction reference price and the weighted mid** gives the direction.
5. **Order-book imbalance** adds information on top of the auction imbalance. It is the strongest single signal.

## Models and results

| Model | Validation MAE | vs. zero | Test MAE | vs. zero | Test daily IC (t-stat) |
|---|---|---|---|---|---|
| Predict zero | 6.101 | — | 5.962 | — | — |
| OLS, 6 standardised features | 6.065 | −0.6% | — | — | — |
| Quantile (median) regression, clipped features | 6.049 | −0.8% | 5.920 | −0.7% | 0.117 (22.7) |
| LightGBM, L1 objective | 6.017 | −1.4% | 5.894 | −1.1% | 0.135 (27.6) |

- Quantile regression matches the MAE metric better than OLS, which fat-tailed targets (kurtosis about 22) pull away from the median.
- LightGBM's trees are chosen by early stopping on the last 45 training days, then refit on all of train. It beats the quantile regression on every validation day (paired daily t-stat −16.7).
- About 80% of LightGBM's gain comes from book imbalance and the reference gap.

## Is it tradable?

A long-short book (predictions demeaned across stocks, gross exposure 1, rebalanced every 60 s) earns about **19 bps a day before costs** on the test days, and **loses about 58 bps a day** once each rebalance pays the quoted spread.

**Conclusion:** the signal is real and stable out of sample, but not tradable at this horizon under a simple spread-crossing cost model. The two strongest features are partly mechanical effects of where the weighted mid sits inside the spread.

## Running it

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt      # Windows: .venv\Scripts\pip install -r requirements.txt
kaggle competitions download -c optiver-trading-at-the-close
unzip optiver-trading-at-the-close.zip -d optiver-trading-at-the-close
jupyter lab closing_optiver.ipynb
```

The data is not included in this repository: it is Kaggle competition data, available after accepting the competition rules. The notebook expects `optiver-trading-at-the-close/train.csv` next to it and needs several GB of RAM.
