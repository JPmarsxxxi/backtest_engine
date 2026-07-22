# diagnostic only (not a notebook cell): why are only 3 coins live at the last bar?
print(prices_skew.notna().sum(axis=1).tail(5))
print()
print("last valid date per coin:")
print(prices_skew.apply(lambda s: s.last_valid_index()).sort_values())
