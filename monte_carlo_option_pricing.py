import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class Option:
    S0: float   # initial stock price
    K: float    # strike price
    T: float    # time to maturity (in years)
    r: float    # risk-free rate
    sigma: float  # volatility

class MonteCarloPricer:
    def __init__(self, option: Option, n_simulations: int = 100000):
        self.option = option
        self.n_simulations = n_simulations

    def simulate_terminal_price(self) -> np.ndarray:
        """
        Simulate terminal stock prices using Geometric Brownian Motion.
        """
        Z = np.random.standard_normal(self.n_simulations)
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
            payoffs = np.maximum(ST - self.option.K, 0)
        elif option_type.lower() == "put":
            payoffs = np.maximum(self.option.K - ST, 0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")

        discounted_price = np.exp(-self.option.r * self.option.T) * np.mean(payoffs)
        return discounted_price

def get_stock_data(ticker: str, period="1y"):
    data = yf.download(ticker, period=period)
    data["LogRet"] = np.log(data["Close"] / data["Close"].shift(1))
    sigma = float(data["LogRet"].std() * np.sqrt(252))  # annualized volatility
    S0 = float(data["Close"].iloc[-1])
    return S0, sigma, data


def plot_distribution(ST: np.ndarray, option: Option):
    plt.figure(figsize=(8,5))
    plt.hist(ST, bins=50, density=True, alpha=0.6, color="skyblue")
    plt.axvline(option.K, color="red", linestyle="--", label=f"Strike Price {option.K}")
    plt.title(f"Simulated Distribution of Stock Price at Maturity (T={option.T} yr)")
    plt.xlabel("Stock Price at Maturity")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # 1. Get stock data
    ticker = "BMW.DE"
    S0, sigma, data = get_stock_data(ticker)
    print(f"Latest {ticker} price: {S0:.2f}, Estimated volatility: {sigma:.2%}")

    # 2. Define option parameters
    option = Option(S0=S0, K=150, T=1.0, r=0.03, sigma=sigma)

    # 3. Initialize pricer
    mc = MonteCarloPricer(option, n_simulations=100000)

    # 4. Price call & put
    call_price = mc.price("call")
    put_price = mc.price("put")

    print(f"Monte Carlo {ticker} European Call Price: {call_price:.4f}")
    print(f"Monte Carlo {ticker} European Put Price:  {put_price:.4f}")

    # 5. Plot simulated distribution
    ST = mc.simulate_terminal_price()
    plot_distribution(ST, option)
