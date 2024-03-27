import sys

import pickle
import numpy as np

if __name__ == "__main__":
    sys.path.append("build")
    import trade_env
    vec_trade = trade_env.VecTrade(7)

