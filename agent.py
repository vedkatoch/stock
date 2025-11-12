from google.adk.tools import google_search
import yfinance as yf
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from niftystocks import ns
import financedatabase as fd
from google.adk.agents import LlmAgent
from google.adk.tools.agent_tool import AgentTool
import jugaad_data as jd
from jugaad_data.nse import NSELive

# 




sid_obj = SentimentIntensityAnalyzer()


def get_stock_advice(ticker: str) -> str:
    """
    Provides a concise financial analysis and advice for a given stock using live market data.

    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL', 'TCS.NS').

    Returns:
        str: Professional-style financial advice summary.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        name = info.get("shortName", ticker.upper())
        sector = info.get("sector", "N/A")
        pe = info.get("trailingPE")
        forward_pe = info.get("forwardPE")
        pb = info.get("priceToBook")
        eps_growth = info.get("earningsQuarterlyGrowth")
        debt_to_equity = info.get("debtToEquity")
        beta = info.get("beta")
        dividend_yield = info.get("dividendYield")

        lines = [f"📊 **Financial Advice for {name} ({ticker.upper()})**"]
        lines.append(f"🏢 Sector: {sector}")

        # Valuation
        if pe and forward_pe:
            lines.append(f"💰 Valuation:")
            lines.append(f"   - Current P/E: {pe:.2f}")
            lines.append(f"   - Forward P/E: {forward_pe:.2f}")
            if pe < 15:
                lines.append("   ✅ Stock appears undervalued compared to the market.")
            elif pe > 30:
                lines.append(
                    "   ⚠️ Stock is trading at a high multiple. Consider if growth justifies this."
                )
        else:
            lines.append("   ℹ️ P/E data not available.")

        # Price to Book
        if pb:
            lines.append(f"   - P/B Ratio: {pb:.2f}")
            if pb < 1:
                lines.append("   ✅ Stock is trading below its book value.")
            elif pb > 3:
                lines.append("   ⚠️ Premium valuation relative to book value.")

        # Growth
        if eps_growth is not None:
            lines.append(f"📈 Earnings Growth: {eps_growth * 100:.2f}% YoY")
            if eps_growth > 0.15:
                lines.append("   ✅ Strong earnings momentum.")
            elif eps_growth < 0:
                lines.append("   ⚠️ Negative growth. Investigate reasons.")

        # Risk
        if beta is not None:
            lines.append(f"📉 Beta: {beta:.2f}")
            if beta > 1.2:
                lines.append(
                    "   ⚠️ High volatility. May not suit risk-averse investors."
                )
            elif beta < 0.8:
                lines.append("   ✅ Lower volatility. More defensive in nature.")

        # Debt
        if debt_to_equity is not None:
            lines.append(f"💳 Debt-to-Equity: {debt_to_equity:.2f}")
            if debt_to_equity > 100:
                lines.append("   ⚠️ High leverage. Check debt servicing ability.")
            else:
                lines.append("   ✅ Reasonable debt level.")

        # Dividends
        if dividend_yield:
            lines.append(f"💵 Dividend Yield: {dividend_yield * 100:.2f}%")
            if dividend_yield > 2:
                lines.append("   ✅ Attractive for income-focused investors.")

        # Final recommendation
        lines.append("\n🔎 Summary:")
        if pe and pe < 20 and eps_growth and eps_growth > 0.1:
            lines.append(
                "➡️ This stock shows signs of value and growth. Consider adding to a diversified portfolio."
            )
        elif pe and pe > 30 and eps_growth < 0:
            lines.append(
                "⛔ Caution: Expensive and shrinking earnings. Better opportunities may exist."
            )
        else:
            lines.append(
                "📌 Further research recommended based on your risk profile and investment goals."
            )

        return "\n".join(lines)

    except Exception as e:
        return f"❌ Error fetching data or providing advice: {e}"


def get_livestock():
    """
    Fetches and returns the latest live data for all stocks listed in the NIFTY 50 index.

    Returns:
        List[dict]: A list of dictionaries containing the symbol, company name (if available),
                    and last traded price for each stock in the NIFTY 50 index.
    """
    n = NSELive()
    nifty_data = n.live_index("NIFTY 50")

    stock_list = []

    for stock in nifty_data["data"]:
        stock_list.append(
            {
                "symbol": stock.get("symbol"),
                "company_name": stock.get("companyName", ""),
                "last_price": stock.get("lastPrice"),
            }
        )

    return stock_list


get_news = LlmAgent(
    name="get_news",
    model="gemini-2.0-flash",
    description=(
        "An agent that fetches and summarizes the latest stock market news, expert financial advice, "
        "and provides investment suggestions for a given company or ticker symbol."
    ),
    instruction=(
        "You are a financial assistant helping users make informed stock investment decisions. For a given ticker:\n\n"
        "1. Search for and summarize the most recent and relevant news from the past trending news.\n"
        "2. Include any available insights from financial advisors, market analysts, or credible reports.\n"
        "3. Based on the findings, assess the overall sentiment (Positive / Neutral / Negative).\n"
        "4. Provide a clear recommendation on whether the user should BUY, HOLD, or AVOID the stock.\n"
        "5. If recommending HOLD or AVOID, suggest up to 2 alternative stocks at a similar price point "
        "that may offer better potential based on market conditions, fundamentals, or analyst consensus.\n\n"
        "Make the output concise, clear, and actionable. Use the available tools to gather accurate and recent data."
    ),
    tools=[
        google_search,
    ],
)


def get_all_ticker_name():
    """
    Fetches a combined list of ticker symbols from Indian and global markets.

    Returns:
        List[str]: Combined list of unique ticker symbols.
    """
    symbols = []

    if True:
        try:
            india_symbols = ns.get_nifty_50_with_ns()
            symbols.extend(india_symbols)
        except Exception as e:
            print(f"Error fetching Indian tickers: {e}")

    if True:
        try:
            global_symbols = fd.select_equities(country="USA")
            symbols.extend(global_symbols)
        except Exception as e:
            print(f"Error fetching global tickers: {e}")

    return list(dict.fromkeys(symbols))


def calculate_intrinsic_value(ticker: str, wacc: float = 0.10) -> float:
    """
    Estimate the intrinsic value per share of a stock using a simplified DCF (Discounted Cash Flow) model.

    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL' for Apple Inc., 'GOOGL' for Alphabet Inc.).
        wacc (float, optional): The Weighted Average Cost of Capital to discount future cash flows.
            Default is 0.10 (10%).

    Returns:
        float: The estimated intrinsic value per share of the company.

    Raises:
        ValueError: If any required financial data is missing for the calculation.
        RuntimeError: For any other unexpected errors during the process.
    """
    stock = yf.Ticker(ticker)
    try:
        # Get financial data
        cash_flow = stock.cashflow
        balance_sheet = stock.balance_sheet
        info = stock.info

        # Transpose to access recent year as index 0
        cash_flow = cash_flow.T
        balance_sheet = balance_sheet.T

        # Access latest values safely
        operating_cash_flow = cash_flow[
            "Cash Flow From Continuing Operating Activities"
        ].iloc[0]
        capital_expenditures = cash_flow["Capital Expenditure"].iloc[0]
        total_debt = balance_sheet["Total Debt"].iloc[0]
        cash_and_equivalents = balance_sheet["Cash And Cash Equivalents"].iloc[0]
        shares_outstanding = info["sharesOutstanding"]

        # Calculate FCF and Net Debt
        free_cash_flow = operating_cash_flow - capital_expenditures
        net_debt = total_debt - cash_and_equivalents

        # Compute values
        enterprise_value = free_cash_flow / wacc
        equity_value = enterprise_value - net_debt
        intrinsic_value_per_share = equity_value / shares_outstanding

        return round(intrinsic_value_per_share, 2)

    except KeyError as e:
        raise ValueError(f"Missing required financial data: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error: {e}")


def is_stock_undervalued(ticker: str) -> str:
    """
    Determine whether a stock is undervalued, overvalued, or fairly valued.

    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL' for Apple Inc., 'GOOGL' for Alphabet Inc.).

    Returns:
        str: Valuation result - "Undervalued", "Overvalued", or "Fairly Valued".
    """
    stock = yf.Ticker(ticker)
    intrinsic_value = calculate_intrinsic_value(ticker)

    try:
        current_price = stock.info["currentPrice"]
    except Exception as e:
        raise ValueError(f"Unable to fetch current price: {e}")

    threshold = 0.10  # 10% tolerance

    if current_price < intrinsic_value * (1 - threshold):
        return "Undervalued"
    elif current_price > intrinsic_value * (1 + threshold):
        return "Overvalued"
    else:
        return "Fairly Valued"


def get_sentiment_analysis(sentence: str) -> dict:
    """
    Analyzes the sentiment of a given sentence using VADER sentiment analysis.

    Args:
        sentence (str): The input text to analyze.

    Returns:
        dict: A dictionary containing:
            - 'positive': Percentage of positive sentiment
            - 'neutral' : Percentage of neutral sentiment
            - 'negative': Percentage of negative sentiment
            - 'compound': Compound score from VADER (-1 to 1)
            - 'overall_sentiment': One of 'Positive', 'Neutral', or 'Negative'
    """
    sentiment_dict = sid_obj.polarity_scores(sentence)

    compound = sentiment_dict["compound"]
    overall_sentiment = (
        "Positive"
        if compound >= 0.05
        else "Negative" if compound <= -0.05 else "Neutral"
    )

    result = {
        "positive": f"{sentiment_dict['pos'] * 100:.2f}%",
        "neutral": f"{sentiment_dict['neu'] * 100:.2f}%",
        "negative": f"{sentiment_dict['neg'] * 100:.2f}%",
        "compound": compound,
        "overall_sentiment": overall_sentiment,
    }

    return result



def get_Historical_data(stock: str, time_period: str):
    """
    Fetch historical stock market data for a given stock symbol over a specified time period.

    Args:
        stock (str): The stock ticker symbol (e.g., 'AAPL' for Apple Inc., 'GOOGL' for Alphabet Inc.).
        time_period (str): The period of historical data to retrieve.
                           Accepts values like '1d', '5d', '1mo', '3mo', '6mo',
                           '1y', '2y', '5y', '10y', 'ytd', or 'max'.

    Returns:
       csv: A csv containing historical market data with the following columns:
                      - 'Open': Opening price of the stock for each interval.
                      - 'High': Highest price during the interval.
                      - 'Low': Lowest price during the interval.
                      - 'Close': Closing price of the stock for each interval.
                      - 'Volume': Number of shares traded during the interval.
    """
    ticker = yf.Ticker(stock)
    historical_data = ticker.history(period=time_period)
    return historical_data[["Open", "High", "Low", "Close", "Volume"]].to_csv()


root_agent = LlmAgent(
    name="stock_market_analysis",
    model="gemini-2.0-flash",
    description=(
        "An agent that analyzes stock data, predicts the next day's price, and evaluates whether the stock is worth buying."
    ),
    instruction=(
        "You are a helpful agent who provides comprehensive stock market insights. For a given stock, perform the following tasks:\n\n"
        "1. Use the `get_Historical_data` tool to fetch the last 20 days of historical price data.\n"
        "2. Use the `get_news` tool to gather relevant news headlines from the past trending and latest news, including expert commentary or financial advisor perspectives.\n"
        "3. Use the `get_sentiment_analysis` tool to analyze the sentiment of those news headlines (positive, neutral, or negative).\n"
        "4. Use the `calculate_intrinsic_value` tool to estimate the intrinsic value per share of the stock using financial data.\n"
        "5. Use the `is_stock_undervalued` tool to determine if the stock is undervalued, fairly valued, or overvalued based on its current price.\n"
        "6. Use the `get_all_ticker_name` tool to fetch a combined list of stock ticker symbols from Indian (Nifty 50) and global markets, with filtering options.\n"
        "7. Use the `get_livestock` tool to retrieve real-time market data for NIFTY 50 stocks including symbol, company name, and last traded price.\n"
        "8. Use the `get_stock_advice` tool to generate stock-specific financial advice based on valuation, growth, risk, and profitability metrics.\n"
        "9. Based on all the above analysis, provide a clear recommendation on whether the user should BUY, HOLD, or AVOID the stock.\n"
        "10. If the recommendation is HOLD or AVOID, suggest 1–2 alternative stocks at a similar price point that may offer better potential based on valuation, sentiment, or fundamentals.\n\n"
        "Make the summary concise, data-driven, and actionable for investors."
        "Note -: Only run those tool which is necessary prevent token limit"
    ),
    tools=[
        get_Historical_data,
        get_sentiment_analysis,
        calculate_intrinsic_value,
        is_stock_undervalued,
        get_all_ticker_name,
        get_livestock,
        get_stock_advice,
        AgentTool(agent=get_news),
    ],
)
