import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def generate_growth_data(days=90, seed=42):
    """
    Synthesizes a realistic business growth trajectory by blending a
    deterministic trend, weekly seasonal curves, and random daily noise.
    """
    np.random.seed(seed)
    
    # Establish a chronological timeline starting in early 2026
    date_range = pd.date_range(start="2026-01-01", periods=days, freq="D")
    
    # 1. Base Trend: Linear growth combined with a slight quadratic acceleration
    time_index = np.arange(days)
    baseline_growth = 500 + (12 * time_index) + (0.1 * (time_index ** 2))
    
    # 2. Weekly Seasonality: Sinusoidal wave mapped across a 7-day period
    # This simulates a typical business cycle (e.g., lower weekend engagement)
    weekly_pattern = 80 * np.sin(2 * np.pi * date_range.dayofweek / 7)
    
    # 3. Environmental Noise: Random fluctuations pulled from a normal distribution
    noise = np.random.normal(loc=0, scale=25, size=days)
    
    # Merge the components into our final performance metric
    daily_users = baseline_growth + weekly_pattern + noise
    
    # Pack the numpy arrays into a cleanly indexed Pandas DataFrame
    df = pd.DataFrame({"Users": daily_users}, index=date_range)
    return df

def forecast_growth(df, horizon=30):
    """
    Applies the Holt-Winters Exponential Smoothing model to isolate 
    the trend and seasonality, then projects growth 30 days into the future.
    """
    print("--- Spinning up the forecasting engine ---")
    
    # Fit the model: additive trend, additive seasonality (7-day period)
    # We use 'add' because our cycles are stable in magnitude.
    model = ExponentialSmoothing(
        df["Users"], 
        trend="add", 
        seasonal="add", 
        seasonal_periods=7
    ).fit()
    
    print("Model parameters successfully optimized.")
    
    # Forecast the next 30 days
    forecast = model.forecast(steps=horizon)
    
    print(f"Forecast complete. Projected users in {horizon} days: {forecast.iloc[-1]:.0f}")
    print("-" * 50)
    
    return forecast, model

if __name__ == "__main__":
    # 1. Generate the historical "messy" data
    data = generate_growth_data()
    
    # 2. Run the math to predict the future
    future, fitted_model = forecast_growth(data)
    
    # 3. Render the Visual Verification Plot
    plt.figure(figsize=(12, 6))
    
    # Plot historical data
    plt.plot(data.index, data["Users"], label="Actual History (The Messy Reality)", color="#2c3e50", linewidth=1.5)
    
    # Plot the forecast
    plt.plot(future.index, future, label="30-Day Forecast (The Math)", color="#27ae60", linestyle="--", linewidth=2)
    
    # Formatting the chart
    plt.title("Infrastructure Capacity & User Growth Forecast", fontsize=14)
    plt.xlabel("Timeline", fontsize=12)
    plt.ylabel("Active User Count", fontsize=12)
    plt.legend(loc="upper left")
    plt.grid(True, linestyle=":", alpha=0.6)
    
    # Save the output visualization directly to the project folder
    plt.savefig("growth_forecast.png")
    print("Success! Visualization saved as 'growth_forecast.png'.")
    plt.show()