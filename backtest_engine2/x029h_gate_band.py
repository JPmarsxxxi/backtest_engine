"""#029h - stack the two turnover levers: spread GATE (skip wide-spread bars) + no-trade BAND (skip small
rebalances), on real FTMO spreads. Question (user): the gate already cuts turnover - does ADDING the band
on top further improve net? They attack different turnover sources so it's not obviously redundant."""
import numpy as np
import pandas as pd
import x029g_ftmo_gate as g   # reuses half_ftmo (real FTMO per-bar spread), Wc, rets, cols, idx

half_ftmo, Wc, rets = g.half_ftmo, g.Wc, g.rets
BARS, ANN = g.BARS, g.ANN

def gate_band(target, lag, cap_bp, band):
    cap = cap_bp / 1e4
    gpos = target.where(half_ftmo <= cap).ffill().fillna(0.0)     # spread gate
    if band > 0:                                                   # + no-trade band
        arr = gpos.values; T, n = arr.shape; held = np.zeros(n); out = np.zeros((T, n))
        for t in range(T):
            mv = np.abs(arr[t] - held) > band
            held = np.where(mv, arr[t], held); out[t] = held
        gpos = pd.DataFrame(out, index=gpos.index, columns=gpos.columns)
    gross = (gpos.shift(lag) * rets).sum(axis=1)
    dpos = (gpos - gpos.shift(1)).abs()
    net = gross - (dpos * half_ftmo).sum(axis=1)
    turn = dpos.sum(axis=1).mean() * BARS
    def sh(r): r = r.dropna(); return r.mean()/r.std()*ANN if r.std() > 0 else 0
    def pct(r): return r.dropna().mean()*BARS*252*100
    return sh(net), pct(net), sh(gross), turn

print('\n#029h gate + band stacked, REAL FTMO spreads (cap=0.15bp fixed, sweep band)')
for lag in (1, 2):
    print(f'lag{lag} ({"idealized" if lag==1 else "realistic"}):')
    for band in (0.0, 0.05, 0.10, 0.20, 0.35, 0.50):
        ns, npct, gs, turn = gate_band(Wc, lag, 0.15, band)
        print(f'  band {band:.2f}: NET Sh {ns:+.2f} ({npct:+.2f}%/yr)  gross Sh {gs:+.2f}  turn {turn:.1f}/d')
