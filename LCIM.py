import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import subprocess
import logging
import os

# Setting up logging for better debugging and monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Label Encoder instance
le = LabelEncoder()

# File paths (modify them as per your directory structure)
input_path = 'databases/stock.txt'
output_path = 'databases/output.txt'
csv_file = 'databases/current_stock_data.csv'
class_file = "MainTestLCIM"
java_file = r"D:\0. STUDY\HK8\Khóa luận\investment_recommendation\models\MainTestLCIM.java"
spmf_jar = r"D:\0. STUDY\HK8\Khóa luận\investment_recommendation\models\spmf_modified.jar"


# Function to write input data to text file
def write_input_into_txt(input_path, df):
    with open(input_path, 'w') as f:
        for _, row in df.iterrows():
            tickers = row['ticker'] if isinstance(row['ticker'], list) else [row['ticker']]
            tickers_str = ' '.join(map(str, tickers))
            sum_utility = row['utility']
            costs = row['cost'] if isinstance(row['cost'], list) else [row['cost']]
            costs_str = ' '.join(map(str, costs))
            f.write(f"{tickers_str}:{sum_utility}:{costs_str}\n")


# Function to calculate utility and cost
def calculate_utility_and_cost(df):
    df['price_change'] = df.groupby('ticker_encoded')['close'].pct_change().fillna(0)
    df['log_volume'] = np.log(df['volume'].replace(0, np.nan) + 1).fillna(0)  # Add 1 to avoid log(0)

    # Adjusting the utility and cost calculations
    df['utility'] = np.where(df['price_change'] > 0,
                             df['price_change'] * (1 + df['log_volume']), 0)
    df['cost'] = np.where(df['price_change'] <= 0,
                          abs(df['price_change']) * (1 - df['log_volume']), 0)
    return df


# Function to prepare input for LCIM model
def file_input_LCIM(df, input_path):
    try:
        df['ticker_encoded'] = le.fit_transform(df['ticker'])
        df['date'] = pd.to_datetime(df['date'], dayfirst=True)  # Fix for date parsing
        df = df.sort_values(by='date').reset_index(drop=True)

        # Calculate utility and cost
        df = calculate_utility_and_cost(df)

        # Grouping data for input to LCIM
        result = (
            df.groupby('date', group_keys=False, as_index=False)
            .apply(lambda group: pd.DataFrame({
                'Trans': [group['date'].iloc[0]] * len(group),
                'ticker': group['ticker_encoded'].tolist(),
                'utility': [group['utility'].sum()] * len(group),
                'cost': group['cost'].tolist()
            }))
            .reset_index(drop=True)
        )

        write_input_into_txt(input_path, result)

        # Calculate thresholds
        utility_threshold = result['utility'].quantile(0.75)
        cost_flat = pd.Series([item if isinstance(item, float) else i for sublist in result['cost']
                               for i in (sublist if isinstance(sublist, list) else [sublist])
                               for item in [i]])
        cost_threshold = abs(cost_flat).quantile(0.25)

        return utility_threshold, cost_threshold

    except Exception as e:
        logging.error(f"Error in file_input_LCIM: {e}")
        raise


# Function to compile and run the LCIM model
def compile_and_run_java(input_file, output_file, minutil, maxcost, minsup):
    try:
        # Compile Java file
        compile_command = ["javac", "-cp", spmf_jar, java_file]
        logging.info(f"Compiling Java file: {' '.join(compile_command)}")
        compile_process = subprocess.run(compile_command, check=True)

        # Run Java program if compilation is successful
        if compile_process.returncode == 0:
            run_command = [
                "java",
                "-cp", f".;{spmf_jar}",  # Use ';' for Windows, ':' for Unix/Mac
                class_file,
                input_file,
                output_file,
                str(minutil),
                str(maxcost),
                str(minsup)
            ]
            logging.info(f"Running Java model: {' '.join(run_command)}")
            subprocess.run(run_command, check=True)

    except subprocess.CalledProcessError as e:
        logging.error(f"Java subprocess failed: {e}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise


# Main execution flow
if __name__ == '__main__':
    try:
        # Check if CSV file exists
        if not os.path.exists(csv_file):
            logging.error(f"CSV file not found: {csv_file}")
            raise FileNotFoundError(f"{csv_file} not found.")

        # Load data
        df = pd.read_csv(csv_file)

        # Process input and get thresholds
        utility_threshold, cost_threshold = file_input_LCIM(df, input_path)

        # Run LCIM Model
        compile_and_run_java(input_path, output_path, utility_threshold, cost_threshold, 0.8)

        logging.info("LCIM model execution completed successfully.")

    except Exception as e:
        logging.error(f"An error occurred in the main flow: {e}")
