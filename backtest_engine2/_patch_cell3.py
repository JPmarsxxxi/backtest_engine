"""Patch _cell_2.py → _cell_2_v2.py with the hold-through-regime change."""
src = open('_cell_2.py', 'r', encoding='utf-8').read()

# 1. Remove _close_all_for_regime_off method
start_marker = "\n    def _close_all_for_regime_off(self, prices, t):"
end_marker   = "\n    def generate_weights(self, data, t):"
s = src.find(start_marker)
e = src.find(end_marker)
if s != -1 and e != -1 and e > s:
    src = src[:s] + end_marker + src[e + len(end_marker):]
    print("Removed _close_all_for_regime_off")
else:
    print(f"WARNING: markers not found  s={s} e={e}")

# 2. Replace the old regime_off action block in generate_weights
old = (
    "        if regime_off:\n"
    "            if self.open_trades:\n"
    "                self._close_all_for_regime_off(data.prices, t)\n"
    "            self._event_this_bar = True   # bypass hold-positions logic in apply_risk\n"
    "            return pd.Series(0.0, index=data.prices.columns)\n"
    "\n"
)
new = (
    "        if regime_off:\n"
    "            # Block new entries; existing trades run to their natural OU exit\n"
    "            _saved = self.max_new_entries_per_bar\n"
    "            self.max_new_entries_per_bar = 0\n"
    "            weights = super().generate_weights(data, t)\n"
    "            self.max_new_entries_per_bar = _saved\n"
    "            return weights\n"
    "\n"
)
if old in src:
    src = src.replace(old, new)
    print("Replaced regime_off block")
else:
    print("WARNING: old regime_off block not found")

# 3. Update docstring
old_doc = (
    '    """\n'
    '    OuTrendPullbackStrategy gated by a BTC DVOL expanding-percentile macro regime filter.\n'
    '    DVOL > expanding P(dvol_percentile) of all DVOL seen so far → regime OFF: close all\n'
    '    positions immediately and block new entries until DVOL falls back below the threshold.\n'
    '    """\n'
)
new_doc = (
    '    """\n'
    '    OuTrendPullbackStrategy gated by a BTC DVOL expanding-percentile macro regime filter.\n'
    '    DVOL > expanding P(dvol_percentile): block new entries only.\n'
    '    Existing trades run to their natural OU exit (pi_plus / pi_minus / timeout).\n'
    '    """\n'
)
if old_doc in src:
    src = src.replace(old_doc, new_doc)
    print("Updated docstring")
else:
    print("WARNING: docstring not found (non-blocking)")

# 4. Fix comment in regime ON section
src = src.replace(
    "        # ── Regime ON: delegate to OU trend+pullback layer (Layer 2) ─────────",
    "        # ── Regime ON: full strategy ──────────────────────────────────────────────────"
)

open('_cell_2_v2.py', 'w', encoding='utf-8').write(src)
print(f"Written _cell_2_v2.py  ({src.count(chr(10))} lines)")
print(f"_close_all_for_regime_off present: {'_close_all_for_regime_off' in src}")
print(f"max_new_entries_per_bar = 0 present: {'max_new_entries_per_bar = 0' in src}")
