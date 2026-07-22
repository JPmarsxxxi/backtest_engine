import sys
sys.path.insert(0, r'C:\Users\User\backtest_engine\backtest_engine2')

if __name__ == '__main__':
    from _mood_helpers import calibrate_mood_threshold
    MOOD_THRESHOLD = calibrate_mood_threshold()
    print(f"\nMOOD_THRESHOLD: {MOOD_THRESHOLD:.6f}")
