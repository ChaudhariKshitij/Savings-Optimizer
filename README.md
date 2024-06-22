
## Welcome to Your Savings Optimizer!! 💰📈

This project extends my previous work on [Portfolio Optimization](https://github.com/ChaudhariKshitij/Data-Visualization-Data-Analysis/tree/fa3de79f14c66de616909ed79008c5092a4ac6b3/Portfolio%20Optimization). It allows you to optimize your investments using historical data and modern portfolio theory principles.

### Features ✨
- **Interactive Stock Selection:** Choose your preferred stocks from a comprehensive list.
- **Investment Value Input:** Enter the total amount you wish to invest.
- **Optimized Allocation:** Calculate the optimal allocation of your investment across selected stocks.
- **Detailed Metrics:** View important portfolio metrics such as expected annual return, volatility, and variance.
- **Visual Representations:** Charts and tables to help visualize your portfolio allocation and metrics.

### Usage Instructions 📘
1. Open the application in your browser.
2. Select your desired stocks from the sidebar.
3. Enter the total amount you wish to invest.
4. Click the "Submit" button to get your optimized portfolio allocation.
5. View the suggested stock allocations and portfolio metrics in the main panel.
   
### Explaination
1. **User Interface:** Utilizes Streamlit to create an interactive web application for stock selection and input of investment amount.
2. **Data Download and Cleaning:** Fetches historical stock data from Yahoo Finance, handles missing data, and calculates returns and covariance.
3. **Portfolio Optimization:** Uses `PyPortfolioOpt` to optimize the stock allocation based on the Sharpe ratio.
4. **Discrete Allocation:** Determines the exact number of shares to buy for each stock within the total investment amount.
5. **Visualization:** Displays the allocation and metrics in tables and bar charts.

### Important Note ⚠️
This project is done for study purpose only.
**This allocation is suggested based on calculations aimed at maximizing returns. However, stock markets are always subject to market risks. Make your investment decisions wisely.**

### Contributing 🤝
Feel free to fork the project and make your contributions. Any suggestions or improvements are welcome!
