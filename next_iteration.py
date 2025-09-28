# monte_carlo_option_pricing.py
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from dataclasses import dataclass
from math import log, sqrt, exp, erf
import argparse
import sys

def norm_cdf(x):
    """Standard normal CDF via erf (avoids SciPy dependency)."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

@dataclass
class Option:
    S0: float   # initial stock price
    K: float    # strike price
    T: float    # time to maturity (in years)
    r: float    # risk-free rate
    sigma: float  # volatility

class MonteCarloPricer:
    def __init__(self, option: Option, n_simulations: int = 100000, seed: int | None = None):
        self.option = option
        self.n_simulations = int(n_simulations)
        self.rng = np.random.default_rng(seed)

    def simulate_terminal_price(self) -> np.ndarray:
        """
        Simulate terminal stock prices using Geometric Brownian Motion (GBM).
        Returns an array of terminal prices (size = n_simulations).
        """
        Z = self.rng.standard_normal(self.n_simulations)
        ST = self.option.S0 * np.exp(
            (self.option.r - 0.5 * self.option.sigma**2) * self.option.T +
            self.option.sigma * np.sqrt(self.option.T) * Z
        )
        return ST

    def price(self, option_type: str = "call") -> float:
        """
        Price a European Call or Put using Monte Carlo.
        """
        ST = self.simulate_terminal_price()
        if option_type.lower() == "call":
            payoffs = np.maximum(ST - self.option.K, 0.0)
        elif option_type.lower() == "put":
            payoffs = np.maximum(self.option.K - ST, 0.0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")

        discounted_price = np.exp(-self.option.r * self.option.T) * np.mean(payoffs)
        return float(discounted_price)

def black_scholes(option: Option, option_type: str = "call") -> float:
    """Closed-form Black-Scholes price for European options (no dividends)."""
    S, K, T, r, sigma = option.S0, option.K, option.T, option.r, option.sigma
    if T <= 0 or sigma <= 0:
        # fallback: payoff at maturity / immediate
        if option_type == "call":
            return max(S - K, 0.0)
        else:
            return max(K - S, 0.0)
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    if option_type == "call":
        return S * norm_cdf(d1) - K * exp(-r * T) * norm_cdf(d2)
    else:
        return K * exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)

def get_stock_data(ticker: str, period="1y"):
    try:
        data = yf.download(ticker, period=period, progress=False)
    except Exception as e:
        print("Error downloading data:", e, file=sys.stderr)
        raise
    if data.empty:
        raise ValueError("No data returned by yfinance for ticker " + ticker)
    data["LogRet"] = np.log(data["Close"] / data["Close"].shift(1))
    sigma = float(data["LogRet"].std(skipna=True) * np.sqrt(252))  # annualized volatility
    S0 = float(data["Close"].iloc[-1])
    return S0, sigma, data

def plot_distribution(ST: np.ndarray, option: Option, save_path: str | None = None):
    plt.figure(figsize=(8,5))
    plt.hist(ST, bins=50, density=True, alpha=0.6)
    plt.axvline(option.K, color="red", linestyle="--", label=f"Strike Price {option.K}")
    plt.title(f"Simulated Distribution of Stock Price at Maturity (T={option.T} yr)")
    plt.xlabel("Stock Price at Maturity")
    plt.ylabel("Density")
    plt.legend()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Monte Carlo European Option Pricing")
    parser.add_argument("--ticker", default="BMW.DE", help="Ticker for yfinance (default: BMW.DE)")
    parser.add_argument("--K", type=float, default=150.0, help="Strike price")
    parser.add_argument("--T", type=float, default=1.0, help="Time to maturity (years)")
    parser.add_argument("--r", type=float, default=0.03, help="Risk-free rate (annual)")
    parser.add_argument("--sims", type=int, default=100000, help="Number of Monte Carlo simulations")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (optional)")
    parser.add_argument("--save-plot", default=None, help="If provided, saves the histogram to this path (e.g. sample_plot.png)")
    args = parser.parse_args()

    S0, sigma, data = get_stock_data(args.ticker)
    print(f"Latest {args.ticker} price: {S0:.2f}, Estimated volatility: {sigma:.2%}")

    option = Option(S0=S0, K=args.K, T=args.T, r=args.r, sigma=sigma)
    mc = MonteCarloPricer(option, n_simulations=args.sims, seed=args.seed)

    call_mc = mc.price("call")
    put_mc = mc.price("put")
    call_bs = black_scholes(option, "call")
    put_bs = black_scholes(option, "put")

    print(f"Monte Carlo {args.ticker} Call Price: {call_mc:.4f}")
    print(f"Monte Carlo {args.ticker} Put  Price: {put_mc:.4f}")
    print(f"Black-Scholes {args.ticker} Call Price: {call_bs:.4f}")
    print(f"Black-Scholes {args.ticker} Put  Price: {put_bs:.4f}")

    ST = mc.simulate_terminal_price()
    plot_distribution(ST, option, save_path=args.save_plot)

if __name__ == "__main__":
    main()
