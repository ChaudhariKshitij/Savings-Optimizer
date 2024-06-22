import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
import matplotlib.pyplot as plt
import seaborn as sns

# Load the complete list of stocks with their ticker symbols
stocks_df = pd.read_csv('all_nse_stocks.csv')  # Make sure to have this CSV file

# Create a dictionary mapping stock names to ticker symbols
stock_dict = pd.Series(stocks_df.Ticker.values, index=stocks_df.Name).to_dict()
reverse_stock_dict = {v: k for k, v in stock_dict.items()}


# Welcome message with different fonts
st.markdown("""
    <h1 style='font-family:Arial; color:gray;'>Welcome to Your Savings Optimizer!!</h1>
    <h2 style='font-family:Times New Roman; color:darkgray;'>Optimize your investments based on historical data and modern portfolio theory.</h2>
    """, unsafe_allow_html=True)

# Display persistent warning about market risks
st.warning("Note:This project is done for study purpose only.This allocation is suggested based on calculations aimed at maximizing returns. However, stock markets are always subject to market risks. Make your investment wisely.")

# Sidebar for selecting stocks and entering total portfolio value
st.sidebar.header("Stock Selection")
selected_stocks = st.sidebar.multiselect(
    'Select stocks:',
    options=stocks_df.Name.values.tolist(),
    default=None
)
total_portfolio_value = st.sidebar.number_input(
    "Enter the total amount to invest (₹):",
    min_value=1000,
    value=100000
)

# Create a submit button
submit_button = st.sidebar.button("Submit")

if submit_button:
    if not selected_stocks:
        st.write("Please select at least one stock.")
    else:
        assets = [stock_dict[name] for name in selected_stocks if name in stock_dict]
        stockStartDate = datetime(2013, 1, 1)
        today = datetime.today().strftime('%Y-%m-%d')
        df = pd.DataFrame()

        for stock in assets:
            try:
                data = yf.download(stock, start=stockStartDate, end=today)
                if not data.empty:
                    df[stock] = data['Adj Close']
                else:
                    st.warning(f"No data found for {stock}.")
            except Exception as e:
                st.error(f"Could not download data for {stock}: {e}")

        if not df.empty:
            df.dropna(inplace=True)
            if not df.empty:
                returns = df.pct_change().dropna()
                annual_cov_matrix = returns.cov() * 252

                # Calculate weights
                mu = expected_returns.mean_historical_return(df)
                s = risk_models.sample_cov(df)
                ef = EfficientFrontier(mu, s)
                weights = ef.max_sharpe()

                cleaned_weights = ef.clean_weights()
                weight_array = np.array(list(cleaned_weights.values()))

                port_variance = np.dot(weight_array.T, np.dot(annual_cov_matrix, weight_array))
                port_volatility = np.sqrt(port_variance)
                port_annual_return = np.sum(returns.mean() * weight_array) * 252

                # Calculate discrete allocation
                latest_prices = get_latest_prices(df)
                da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=total_portfolio_value)
                allocation, leftover = da.lp_portfolio()

                # Create allocation DataFrame
                allocation_df = pd.DataFrame.from_dict(allocation, orient='index', columns=['Number of Shares'])
                allocation_df['Stock Name'] = allocation_df.index.map(reverse_stock_dict)
                allocation_df['Latest Price (₹)'] = allocation_df.index.map(latest_prices)
                allocation_df['Total Value (₹)'] = allocation_df['Number of Shares'] * allocation_df['Latest Price (₹)']
                allocation_df = allocation_df[['Stock Name', 'Number of Shares', 'Latest Price (₹)', 'Total Value (₹)']]
                allocation_df.reset_index(drop=True, inplace=True)

                # Portfolio metrics
                portfolio_metrics = {
                    'Metric': ['Expected Annual Return', 'Annual Volatility/Risk', 'Annual Variance'],
                    'Value (%)': [port_annual_return * 100, port_volatility * 100, port_variance * 100]
                }
                portfolio_metrics_df = pd.DataFrame(portfolio_metrics)

                # Create a grid layout for displaying charts and tables side by side
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Suggested Stocks Allocation")
                    st.table(allocation_df)
                    st.write(f"You are still remaining with ₹{leftover:.2f}")

                    # Plot bar chart for portfolio allocation
                    fig, ax = plt.subplots(figsize=(10, 6))
                    fig.patch.set_facecolor('none')  # Make the figure background transparent
                    ax.set_facecolor('none')  # Make the axis background transparent
                    bars = ax.bar(allocation_df['Stock Name'], allocation_df['Total Value (₹)'], color=sns.color_palette("viridis", len(allocation_df)))
                    ax.set_xlabel('Stock Name', fontsize=12)
                    ax.set_ylabel('Total Value (₹)')
                    ax.set_title('Suggested Stocks Allocation', fontsize=14)
                    plt.xticks(rotation=45, ha='right', fontsize=10)  # Rotate x-axis labels and adjust font size
                    for bar in bars:
                        yval = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom')
                    st.pyplot(fig)

                with col2:
                    st.subheader("Metrics")
                    st.table(portfolio_metrics_df)

                    # Plot bar chart for portfolio metrics
                    fig2, ax2 = plt.subplots(figsize=(10, 6))
                    fig2.patch.set_facecolor('none')  # Make the figure background transparent
                    ax2.set_facecolor('none')  # Make the axis background transparent
                    bars = ax2.bar(portfolio_metrics_df['Metric'], portfolio_metrics_df['Value (%)'], color=sns.color_palette("viridis", len(portfolio_metrics_df)))
                    ax2.set_xlabel('Metric', fontsize=12)
                    ax2.set_ylabel('Value (%)')
                    ax2.set_title('Metrics', fontsize=14)
                    for bar in bars:
                        yval = bar.get_height()
                        ax2.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom')
                    st.pyplot(fig2)

            else:
                st.warning("No valid data after cleaning. Please select different stocks.")
        else:
            st.warning("No valid data retrieved. Please select different stocks.")
