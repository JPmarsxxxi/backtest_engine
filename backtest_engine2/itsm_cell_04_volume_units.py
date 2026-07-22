# Cell 4 — Stage 1 amendment: volume from quote (USD) to base units (BTC).
# MarketImpact/LiquidityCap compute dollar-ADV as volume * price (components.py:65-76,
# liquidity.py:22-39), so the panel's volume must be in base units, not dollars.
volume = volume / prices  # BTC per 30m bar; indexes/columns identical by construction

panel = DataPanel(prices, volume=volume, check_outliers=False)

implied_dollar_adv = (volume["BTCUSDT"].tail(20).median() * prices["BTCUSDT"].iloc[-1])
raw_quote_median = RAW["quote_volume"].tail(20).median()
print(f"Implied $ per 30m bar (volume*price, last 20 bars): {implied_dollar_adv:,.0f}")
print(f"Raw quote_volume median        (last 20 bars): {raw_quote_median:,.0f}")
print(f"Ratio (should be ~1): {implied_dollar_adv / raw_quote_median:.3f}")
print(f"Panel has volume: {panel.has_field('volume')}")
