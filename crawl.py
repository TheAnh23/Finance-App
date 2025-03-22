import pandas as pd
from polygon import RESTClient
import time

file_path_ticket = pd.read_csv('databases/ticket-in-US.csv')
client = RESTClient('tRIBGoySz4hZB0sB6hPeG6FePhGpFxMH')
current_stock_data = pd.read_csv('databases/current_stock_data.csv')


def get_200_ticket(file):
    filtered_tickets = file[(file['Country'] == 'United States') & (file['Sector'] == 'Technology')]
    sorted_tickets = filtered_tickets.sort_values('Market Cap', ascending=False)
    top_200 = sorted_tickets.head(200)['Symbol'].tolist()
    return top_200


def crawl_ticket(tickets, from_date, to_date):
    all_aggs = {}  # Dictionary to store data for each symbol

    for ticker in tickets:
        aggs = []  # List to store aggregates for current symbol
        for day in client.get_aggs(ticker=ticker, multiplier=1, timespan="day", from_=from_date, to=to_date):
            aggs.append(day)
        all_aggs[ticker] = aggs  # Store aggregates for current symbol in dictionary
        time.sleep(13)  # Wait for 13 seconds between each API call to avoid hitting the rate limit

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

    return new_data


def combine_data(new_data, existing_data):
    combined_data = pd.concat([existing_data, new_data], ignore_index=True)
    combined_data = combined_data.drop_duplicates(subset=["date", "ticker"], keep='last')
    combined_data.index = pd.to_datetime(combined_data.date, dayfirst=True).dt.date
    combined_data = combined_data[["date", "ticker", "close", "volume"]]
    combined_data.to_csv('databases/combined_data.csv', index=False)
    return combined_data


# top_200_ticket = get_200_ticket(file_path_ticket)
# new_crawl_ticket = crawl_ticket(top_200_ticket, "2024-09-01", "2024-09-30")
# combine_data = combine_data(new_crawl_ticket, current_stock_data)