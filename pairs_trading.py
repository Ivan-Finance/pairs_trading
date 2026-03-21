"""
PAIRS TRADING — Statistical Arbitrage

Pairs trading is a market-neutral strategy that profits from the relative
price movements of two historically correlated securities.

If two stocks tend to move together over time, their price SPREAD should hover
around a stable mean. When the spread deviates too far from that mean, it is
likely to revert and that temporary divergence is the trading opportunity.

When the spread is unusually HIGH:
    - ticker1 is overpriced relative to ticker2
    - SHORT ticker1, LONG ticker2 (bet the gap will close)

When the spread is unusually LOW:
    - ticker1 is underpriced relative to ticker2
    - LONG ticker1, SHORT ticker2

The strategy is market-neutral because one long + one short position
largely cancels out broad market moves (e.g. if the whole market drops,
both positions lose/gain roughly equally).

The relationship between the two stocks can BREAK DOWN permanently
(called a breakdown in cointegration), causing the spread to never revert.
This is why statistical testing BEFORE trading is essential.

"""
!pip install yfinance

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

def pairs_trading(ticker1, ticker2, start="2018-01-01", end="2026-3-21"):

    # 1. DATA DOWNLOAD

    # Downloading adjusted closing prices for both tickers.
    # auto_adjust=True corrects prices for dividends and stock splits,
    # avoiding artificial jumps in the price series.
    df = yf.download([ticker1, ticker2], start=start, end=end, auto_adjust=True, progress=False)
    prices = df["Close"]
    prices.dropna(inplace=True)

    print(f"\n{'='*60}")
    print(f"  PAIRS TRADING: {ticker1} vs {ticker2}")
    print(f"  Period: {start} → {end}")
    print(f"  Trading days: {len(prices)}")
    print(f"{'='*60}")

    # 2. SPREAD

    # The spread is the difference between the two prices.
    # A positive spread means ticker1 is more expensive than ticker2.
    # The sign depends on which ticker you put first: what matters is
    # the MAGNITUDE of the deviation, not the direction.
    spread = prices[ticker1] - prices[ticker2]

    print(f"\n  Spread (last 5 days):")
    print(spread.tail())

    # 3. ROLLING STATISTICS & Z-SCORE
    
    # We use a 20-day rolling window to compute the local mean and
    # standard deviation of the spread.
    # This makes the model adaptive: it tracks slow structural drift
    # in the relationship between the two stocks.
    ma_20 = spread.rolling(20).mean()
    std_rolling = spread.rolling(20).std()

    # Z-score: how many standard deviations is today's spread
    # away from its recent 20-day mean?
    #
    #   z = (spread - mean) / std
    #
    # Interpretation:
    #   z ≈  0  → spread is at its historical average → no signal
    #   z >  2  → spread is unusually HIGH (2 std above mean) → sell ticker1, buy ticker2
    #   z < -2  → spread is unusually LOW  (2 std below mean) → buy ticker1, sell ticker2
    #
    # The threshold of ±2 is conventional: in a normal distribution,
    # values beyond ±2 std occur only ~5% of the time, making them
    # statistically "extreme" and likely to revert.
    z_series = (spread - ma_20) / std_rolling

    print(f"\n  Z-score (last 5 days):")
    print(z_series.dropna().tail())

    # 4. TRADING SIGNALS
    
    # np.where works like a vectorized if/elif/else over the entire Series.
    # Instead of looping day by day, it applies the condition to all rows at once.
    
    #   signal =  1 → spread too high → short ticker1, long ticker2
    #   signal = -1 → spread too low  → long ticker1, short ticker2
    #   signal =  0 → no trade
    signal = np.where(z_series > 2, 1, np.where(z_series < -2, -1, 0))

    print(f"\n  Signal distribution:")
    print(f"  +1 (spread high — short {ticker1}, long {ticker2}):  {(signal ==  1).sum()} days")
    print(f"  -1 (spread low  — long {ticker1}, short {ticker2}): {(signal == -1).sum()} days")
    print(f"   0 (no trade):                                       {(signal ==  0).sum()} days")

    # 5. Z-SCORE CHART
    
    plt.figure(figsize=(13, 5))

    # Plot the z-score time series
    z_series.plot(color="steelblue", linewidth=0.8, label="Z-score")

    # Reference lines at 0, +2, -2
    plt.axhline( 2, color="darkblue", linewidth=1.2, linestyle="--", label="+2 std (sell signal)")
    plt.axhline( 0, color="red",  linewidth=0.8, linestyle="-",  label="Mean")
    plt.axhline(-2, color="darkblue", linewidth=1.2, linestyle="--", label="-2 std (buy signal)")

    # Overlay the trading signals as colored dots
    plt.scatter(z_series[signal ==  1].index, z_series[signal ==  1],
                color="green", s=20, zorder=5, label=f"Long {ticker2} / Short {ticker1}")
    plt.scatter(z_series[signal == -1].index, z_series[signal == -1],
                color="red",   s=20, zorder=5, label=f"Long {ticker1} / Short {ticker2}")

    plt.title(f"Z-score of Spread — {ticker1} vs {ticker2}", fontsize=13)
    plt.xlabel("Date")
    plt.ylabel("Z-score")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

    # 6. P&L
    
    # Convert the numpy signal array to a pandas Series so we can use .shift().
    # We must use the same date index as z_series.
    signal_series = pd.Series(signal, index=z_series.index)

    # signal.shift(1) is critical to avoid look-ahead bias:
    # the signal is generated at the END of day t.
    
    # spread.diff() is the daily change in the spread (today - yesterday).
    # If you are long the spread (signal = +1) and the spread widens → profit.
    # If you are short the spread (signal = -1) and the spread narrows → profit.
    pnl = signal_series.shift(1) * spread.diff()
    pnl_cumulative = pnl.cumsum()

    print(f"\n  P&L Summary:")
    print(f"  Peak P&L:      {pnl_cumulative.max():+.2f}")   
    print(f"  Final P&L:     {pnl_cumulative.iloc[-1]:+.2f}") 
    print(f"  Best day:      {pnl.max():+.2f}")               
    print(f"  Worst day:     {pnl.min():+.2f}")               
    peak = pnl_cumulative.cummax()
    drawdown = pnl_cumulative - peak
    print(f"  Max drawdown:  {drawdown.min():+.2f}")

    plt.figure(figsize=(13, 4))
    pnl_cumulative.plot(color="steelblue", linewidth=1.2)
    plt.axhline(0, color="red", linewidth=0.8, linestyle="--")
    plt.title(f"Cumulative P&L — {ticker1} vs {ticker2}", fontsize=13)
    plt.xlabel("Date")
    plt.ylabel("Cumulative profit/loss ($)")
    plt.tight_layout()
    plt.show()

    # 7. AUGMENTED DICKEY-FULLER TEST

    # Before trusting any pairs trading strategy, you must verify that the
    # spread is STATIONARY, meaning it genuinely reverts to a stable mean
    # rather than drifting randomly. The ADF test checks this formally.
    # It estimates how strongly the spread is pulled back toward its mean
    # each day. If the pull is strong and consistent, the series is stationary.
    
    # Null hypothesis (H0): the spread is a random walk (non-stationary)
    # Alternative (H1):     the spread is stationary (mean-reverting)
    
    # Decision rule:
    #   p-value < 0.05 → reject H0 → spread is stationary → pairs trading makes sense
    #   p-value > 0.05 → fail to reject H0 → spread may drift → strategy is unreliable
    
    # The ADF Statistic should be more negative than the critical value (-2.86 at 5%).
    # The more negative, the stronger the evidence of stationarity.
    result = adfuller(spread.dropna())

    print(f"\n  Augmented Dickey-Fuller Test:")
    print(f"  ADF Statistic:  {result[0]:.4f}  (critical value at 5%: -2.86)")
    print(f"  P-value:        {result[1]:.4f}")
    print(f"  Verdict:        ", end="")

    if result[1] < 0.05:
        print("Spread is stationary: pairs trading is statistically justified.")
    else:
        print("Spread is NOT stationary: strategy may be unreliable.")

# RUN (Change the tickers to analyze any pair you want)

# Simply call pairs_trading(ticker1, ticker2) with any two stocks.
# The best candidates are companies in the SAME sector driven by the SAME
# underlying factors (oil price, interest rates, consumer demand, etc.).
# The ADF test will tell you whether the pair is statistically valid.

# Some historically correlated pairs to try:

#   ENERGY (both driven by crude oil prices)
#   pairs_trading("XOM", "CVX")     # ExxonMobil vs Chevron
#   pairs_trading("BP", "SHEL")     # BP vs Shell

#   BANKS (both sensitive to interest rates)
#   pairs_trading("GS", "MS")       # Goldman Sachs vs Morgan Stanley
#   pairs_trading("JPM", "BAC")     # JPMorgan vs Bank of America

#   TECH (both driven by ad revenue / cloud growth)
#   pairs_trading("META", "GOOGL")  # Meta vs Alphabet
#   pairs_trading("MSFT", "ORCL")   # Microsoft vs Oracle

#   BEVERAGES (same consumer sector, weak cointegration historically)
#   pairs_trading("KO", "PEP")      # Coca-Cola vs Pepsi

#   RETAIL
#   pairs_trading("WMT", "TGT")     # Walmart vs Target
#   pairs_trading("MCD", "YUM")     # McDonald's vs Yum Brands

pairs_trading("XOM", "CVX")