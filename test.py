from crawl import *

# top_200_ticket = get_200_ticket(file_path_ticket)
#
# all_aggs = {}  # Dictionary to store data for each symbol
#
# for ticker in top_200_ticket:
#     print(ticker)
#     aggs = []  # List to store aggregates for current symbol
#     for day in client.get_aggs(ticker=ticker, multiplier=1, timespan="day", from_="2024-09-01", to="2024-09-05"):
#         aggs.append(day)
#         print(day)
#     all_aggs[ticker] = aggs  # Store aggregates for current symbol in dictionary
#     time.sleep(13)  # Wait for 13 seconds between each API call to avoid hitting the rate limit

# Create an empty list to store the dataframes
dfs = []

# Iterate through the keys (tickers) in all_aggs
for ticker, aggs in all_aggs.items():
    # Create an empty list for the current ticker's data
    data = []

    # Iterate through the aggregates for the current ticker
    for agg in aggs:
        # Append the data to the list
        data.append({
            "date": agg.timestamp,
            "ticker": ticker,
            # "open": agg.open,
            # "low": agg.low,
            # "high": agg.high,
            "close": agg.close,
            "volume": agg.volume,
            # "transactions": agg.transactions,
            # "vwap": agg.vwap
        })

    # Create a dataframe from the list of dictionaries
    df = pd.DataFrame(data)

    # Append the dataframe to the list of dataframes
    dfs.append(df)

# Concatenate all dataframes into one
new_data = pd.concat(dfs, ignore_index=True)

