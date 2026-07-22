# Cell 7 — Filter Analysis: Check if the fixes actually work
# Run this after Cell 4 to analyze trades under different filter conditions

import pandas as pd
import numpy as np
from pathlib import Path

TRADE_LOG_PATH = Path("data/ou_trend_trade_log.parquet")

if not TRADE_LOG_PATH.exists():
    print("❌ Trade log not found. Run Cell 4 first!")
else:
    trades = pd.read_parquet(TRADE_LOG_PATH)
    
    # Expand entry_ctx if it's stored as dict
    if 'entry_ctx' in trades.columns:
        ctx = pd.json_normalize(trades['entry_ctx'])
        df = pd.concat([trades.drop(columns=['entry_ctx']), ctx], axis=1)
    else:
        df = trades.copy()
    
    # Add derived columns
    df['pi_width'] = df['pi_plus'] - df['pi_minus']
    df['dip_pct'] = (df['p0'] - df['m0']) / df['p0']
    
    print("="*80)
    print("BASELINE (ALL TRADES)")
    print("="*80)
    
    def analyze_subset(subset, label):
        if len(subset) == 0:
            print(f"\n{label}: NO TRADES ❌")
            return
        
        win_rate = subset['win'].mean()
        mean_ret = subset['ret_pct'].mean()
        median_ret = subset['ret_pct'].median()
        n = len(subset)
        
        # Exit breakdown
        stop_pct = (subset['exit_reason'] == 'pi_minus').mean()
        profit_pct = (subset['exit_reason'] == 'pi_plus').mean()
        timeout_pct = (subset['exit_reason'] == 'timeout').mean()
        
        print(f"\n{label}")
        print(f"  Trades:        {n:,}")
        print(f"  Win Rate:      {win_rate:.1%}")
        print(f"  Mean Return:   {mean_ret:+.2%}  ({mean_ret*1e4:+.1f} bps)")
        print(f"  Median Return: {median_ret:+.2%}")
        print(f"  Exits: STOP {stop_pct:.1%} | PROFIT {profit_pct:.1%} | TIMEOUT {timeout_pct:.1%}")
    
    analyze_subset(df, "BASELINE (all trades)")
    
    print("\n" + "="*80)
    print("INDIVIDUAL FILTERS")
    print("="*80)
    
    # Filter 1: Stronger trend
    if 'roll_trend' in df.columns:
        f1 = df[df['roll_trend'] >= 0.06]
        analyze_subset(f1, "Filter 1: Stronger trend (roll_trend >= 6%)")
    
    # Filter 2: Better OU quality
    f2 = df[df['ou_sharpe'] >= -0.05]
    analyze_subset(f2, "Filter 2: Better OU quality (ou_sharpe >= -0.05)")
    
    # Filter 3: Sane half-life
    f3 = df[(df['half_life'] >= 10) & (df['half_life'] <= 80)]
    analyze_subset(f3, "Filter 3: Sane half-life (10-80 bars)")
    
    # Filter 4: Wider exit corridor
    f4 = df[df['pi_width'] >= 5.0]
    analyze_subset(f4, "Filter 4: Wider corridor (pi_width >= 5)")
    
    # Filter 5: Lower sigma (less noise)
    f5 = df[df['sigma'] <= df['sigma'].quantile(0.75)]
    analyze_subset(f5, "Filter 5: Lower noise (sigma <= 75th percentile)")
    
    # Filter 6: Faster exit
    f6 = df[df['bars_held'] <= 72]
    analyze_subset(f6, "Filter 6: Faster exit (would have exited by 72h)")
    
    print("\n" + "="*80)
    print("COMBINED FILTERS")
    print("="*80)
    
    # COMBO 1: The Cell 6 suggestion
    combo1_conditions = [
        df.get('roll_trend', pd.Series([True]*len(df))) >= 0.06,
        df['ou_sharpe'] >= -0.05,
        (df['half_life'] >= 10) & (df['half_life'] <= 80),
        df['pi_width'] >= 5.0,
    ]
    combo1 = df[np.all(combo1_conditions, axis=0)]
    analyze_subset(combo1, "COMBO 1: Cell 6 suggestion (trend + ou + hl + width)")
    
    # COMBO 2: Add sigma filter
    combo2_conditions = combo1_conditions + [
        df['sigma'] <= df['sigma'].quantile(0.75)
    ]
    combo2 = df[np.all(combo2_conditions, axis=0)]
    analyze_subset(combo2, "COMBO 2: COMBO 1 + lower sigma")
    
    # COMBO 3: Most aggressive
    combo3_conditions = [
        df.get('roll_trend', pd.Series([True]*len(df))) >= 0.08,  # Even stronger
        df['ou_sharpe'] >= 0.0,  # Actually positive
        (df['half_life'] >= 15) & (df['half_life'] <= 60),  # Tighter range
        df['pi_width'] >= 6.0,  # Wider
        df['sigma'] <= df['sigma'].quantile(0.50),  # Bottom half noise
    ]
    combo3 = df[np.all(combo3_conditions, axis=0)]
    analyze_subset(combo3, "COMBO 3: Aggressive (very selective)")
    
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    
    # Create summary comparison
    results = []
    
    for name, subset in [
        ("Baseline", df),
        ("F1: Trend≥6%", df[df.get('roll_trend', pd.Series([True]*len(df))) >= 0.06] if 'roll_trend' in df.columns else pd.DataFrame()),
        ("F2: OU≥-0.05", f2),
        ("F3: HL 10-80", f3),
        ("F4: Width≥5", f4),
        ("F5: Low σ", f5),
        ("COMBO 1", combo1),
        ("COMBO 2", combo2),
        ("COMBO 3", combo3),
    ]:
        if len(subset) > 0:
            results.append({
                'Filter': name,
                'N': len(subset),
                'Win%': f"{subset['win'].mean():.1%}",
                'Mean Ret': f"{subset['ret_pct'].mean():+.2%}",
                'Stop%': f"{(subset['exit_reason']=='pi_minus').mean():.1%}",
                'Timeout%': f"{(subset['exit_reason']=='timeout').mean():.1%}",
            })
    
    summary = pd.DataFrame(results)
    print(summary.to_string(index=False))
    
    print("\n" + "="*80)
    print("INTERPRETATION GUIDE")
    print("="*80)
    print("""
Compare each filter to BASELINE:
- Higher Win% = filter cuts losers
- Higher Mean Ret = filter improves P&L per trade
- Lower Stop% = fewer stop-outs (good if stops were false)
- Lower Timeout% = fewer dead trades

If COMBO 1, 2, or 3 show big improvements:
→ Add those filters to Cell 2 (_try_enter method)
→ Re-run Cell 4 walk-forward to see if OOS performance improves

Trade-off: More filters = fewer trades = less capital deployed
Check N (number of trades) - if COMBO 3 only gives you 50 trades across 5 years,
that's too selective (not enough sample size).

Sweet spot: 2-3x better metrics with N still > 500-1000 trades
    """)