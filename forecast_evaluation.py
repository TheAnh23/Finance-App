import re
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

le = LabelEncoder()
data = pd.read_csv("databases/database.csv")
output_LCIM = pd.read_csv("databases/output_LCIM.csv")

def distionaty_stock(data):
  data['Ticket'] = le.fit_transform(data['ticker'])
  df = data[['Ticket','date','close','volume','ticker']]
  df.loc[:, 'date'] = pd.to_datetime(df['date'])
  df = df.sort_values(by='date').reset_index(drop=True)
  return df
def split_funds(df):
  new_rows = []
  for index, row in df.iterrows():
    funds = row['Ticket'].split(" ")  # Split the 'Fund' column by whitespace
    for fund in funds:
      new_row = row.copy()  # Create a copy of the original row
      new_row['Ticket'] = fund  # Replace the 'Fund' value with the individual fund
      new_rows.append(new_row)  # Append the new row to the list
  return pd.DataFrame(new_rows)
def process_funds(df):
  df_sorted = df.sort_values(by=['Ticket', 'A_Utility', 'A_Cost', 'Transaction'])
  df_processed = df_sorted.drop_duplicates(subset='Ticket', keep='first')
  df_processed = df_processed.sort_values(by=['A_Utility', 'A_Cost'], ascending=True).reset_index(drop=True)
  return df_processed
def get_recommendation(output_file, df_distionaty_stock):
  df = output_file[output_file['Ticket'].str.contains(" ", na=False)]
  list_ticket = df.sort_values(by=['A_Utility','A_Cost'], ascending=True)\
      .reset_index(drop=True)
  list_ticket['Level'] = list_ticket["Ticket"].apply(lambda x: len(re.findall(r'\b\d+\b', x)))
  list_ticket['Transaction'] = range(len(list_ticket))
  df_splitted = split_funds(list_ticket)
  df_result = process_funds(df_splitted)
  df_result['Ticket'] = df_result['Ticket'].astype(int)
  df_distionaty_stock_unique = df_distionaty_stock[['ticker', 'Ticket']].drop_duplicates()
  df_result = df_result.merge(df_distionaty_stock_unique, on='Ticket', how='left')
  df_result.rename(columns={'ticker': 'Ticket Name'}, inplace=True)
  df_result['Visualize'] = 1/df_result['A_Cost']
  return df_result
def process_output_LCIM_model(output_file, ticket, df_distionaty_stock):
  if ticket not in df_distionaty_stock['ticker'].values:
    print(f"Warning: Ticket '{ticket}' not found in df_distionaty_stock.")
    return pd.DataFrame()
  filtered_rows = df_distionaty_stock[df_distionaty_stock['ticker'] == ticket]['Ticket'].unique()
  ticket_encoded = filtered_rows[0] if len(filtered_rows) > 0 else None
  df = output_file[output_file['Ticket'].str.contains(" ", na=False)]
  list_ticket = df[df['Ticket'].str.contains(str(ticket_encoded), na=False)] \
    .sort_values(by=['A_Cost'], ascending=True) \
    .reset_index(drop=True)
  list_ticket['Level'] = list_ticket["Ticket"].apply(lambda x: len(re.findall(r'\b\d+\b', x)))
  list_ticket['Transaction'] = range(len(list_ticket))
  df_splitted = split_funds(list_ticket)
  df_result = process_funds(df_splitted)
  df_result['Ticket'] = df_result['Ticket'].astype(int)
  df_distionaty_stock_unique = df_distionaty_stock[['ticker', 'Ticket']].drop_duplicates()
  df_result = df_result.merge(df_distionaty_stock_unique, on='Ticket', how='left')
  df_result.rename(columns={'ticker': 'Ticket Name'}, inplace=True)
  df_result['Visualize'] = 1 / df_result['A_Cost']
  return df_result
def visualize_data(df):
  fig = px.treemap(df,
                   path=['Ticket Name'],
                   values='Visualize',
                   color='Visualize',
                   hover_data=['A_Cost'],
                   color_continuous_scale=['#ff0000', '#0000ff', '#ffa500', '#ffff00', '#008000']
                   )
  # Cập nhật layout
  fig.update_layout(
    template="plotly_white",
    coloraxis_colorbar=dict(
      title="Price",  # Đổi tên thanh chú thích
      orientation="h",  # Chuyển sang nằm ngang
      x=0.5,  # Căn giữa theo chiều ngang
      xanchor="center",
      y=-0.2  # Đưa thanh chú thích xuống dưới biểu đồ
    ),
    margin=dict(t=0, b=0, l=0, r=0)
  )
  st.plotly_chart(fig)
def get_df_top10_stock(df, df_result):
  top_10_stock = df_result.nlargest(10, 'Visualize')['Ticket Name'].unique()
  # Check if target stock is in top 10
  if stock not in top_10_stock:
    # If not, find the closest stock based on correlation to include
    df_temp = df.pivot(index='date', columns='ticker', values='close')
    correlation_with_target = df_temp.corr()[stock]
    closest_stock = correlation_with_target[~correlation_with_target.index.isin(top_10_stock)].idxmax()
    # Add the closest stock to top 10
    top_10_stock = np.append(top_10_stock, closest_stock)
  table_data = []
  for date in df['date'].unique():
    row = {'date': date}
    for current_stock in top_10_stock:
      stock_data = df[(df['ticker'] == current_stock) & (df['date'] == date)]
      if not stock_data.empty:
        close_value = stock_data['close'].iloc[0]
        row[current_stock] = close_value
    table_data.append(row)
  table_df = pd.DataFrame(table_data)
  table_df = table_df.fillna(0)
  table_df = table_df.set_index('date')
  return table_df
def get_df_correlation_stock(df, stock):
  df_correlation = df.corr()[stock]
  high_corr_columns = df_correlation[df_correlation > 0.90].index
  return df[high_corr_columns]
def create_sequences(X, y, time_step):
  X_seq, y_seq = [], []
  for i in range(len(X) - time_step):
    X_seq.append(X[i:i + time_step])
    y_seq.append(y[i + time_step])
  return np.array(X_seq), np.array(y_seq)
def LSTM_model(time_step, num_features):
  model = Sequential([
      LSTM(50, return_sequences=True, input_shape=(time_step, num_features)),
      LSTM(50),
      Dense(1)
  ])
  model.compile(optimizer='adam', loss='mean_squared_error')
  return model
def predict_future(model, last_sequence, scaler_X, scaler_y, future_days=15):
  future_predictions = []
  current_sequence = last_sequence.copy()
  for _ in range(future_days):
    pred_input = current_sequence.reshape(1, current_sequence.shape[0], current_sequence.shape[1])
    next_pred = model.predict(pred_input)
    next_pred_actual = scaler_y.inverse_transform(next_pred)[0][0]
    future_predictions.append(next_pred_actual)
    next_pred_scaled = scaler_y.transform(np.array(next_pred_actual).reshape(-1, 1))
    next_pred_scaled = next_pred_scaled.reshape(1, -1)  # Reshape to (1, 1)
    if current_sequence.shape[1] > 1:
      next_pred_scaled = np.repeat(next_pred_scaled, current_sequence.shape[1], axis=1)
    current_sequence = np.vstack([current_sequence[1:], next_pred_scaled])  # Shift the sequence
  return future_predictions
def LSTM_proposed_model(target_stock, table_df, time_step, future_days):
  X_data = table_df.drop(columns=[target_stock])
  y_data = table_df[target_stock]
  scaler_X = MinMaxScaler(feature_range=(0, 1))
  scaler_y = MinMaxScaler(feature_range=(0, 1))
  X_scaled = scaler_X.fit_transform(X_data)
  y_scaled = scaler_y.fit_transform(y_data.values.reshape(-1, 1))
  X, y = create_sequences(X_scaled, y_scaled, time_step)
  train_size = int(len(X) * 0.75)
  X_train, y_train = X[:train_size], y[:train_size]
  X_test, y_test = X[train_size:], y[train_size:]
  num_features = X_train.shape[2]
  model_others = LSTM_model(time_step, num_features)
  model_others.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))
  predictions_others = model_others.predict(X_test)
  predictions_others = scaler_y.inverse_transform(predictions_others)
  y_test_actual = scaler_y.inverse_transform(y_test)
  last_sequence = X_test[-1]  # Get last known sequence
  future_predictions = predict_future(model_others, last_sequence, scaler_X, scaler_y, future_days)
  return y_test_actual, predictions_others, future_predictions
def LSTM_traditional_model(target_stock, table_df, time_step, future_days):
  scaler_target = MinMaxScaler(feature_range=(0, 1))
  target_scaled = scaler_target.fit_transform(table_df[target_stock].values.reshape(-1, 1))
  X_target, y_target = create_sequences(target_scaled, target_scaled, time_step)
  train_size_target = int(len(X_target) * 0.75)
  X_train_target, y_train_target = X_target[:train_size_target], y_target[:train_size_target]
  X_test_target, y_test_target = X_target[train_size_target:], y_target[train_size_target:]
  model_self = LSTM_model(time_step, 1)
  model_self.fit(X_train_target, y_train_target, epochs=50, batch_size=16,
                 validation_data=(X_test_target, y_test_target))
  predictions_self = model_self.predict(X_test_target)
  predictions_self = scaler_target.inverse_transform(predictions_self)
  y_test_actual_self = scaler_target.inverse_transform(y_test_target)
  last_sequence = X_test_target[-1]  # Last known sequence
  future_predictions = predict_future(model_self, last_sequence, scaler_target, scaler_target, future_days)
  return y_test_actual_self, predictions_self, future_predictions

def LSTM_visulaize(target_stock, y_test_actual, predictions_others, predictions_self):
  plt.figure(figsize=(15, 6))
  plt.plot(y_test_actual, label='Actual Stock Price')
  plt.plot(predictions_others, label='Hybrid LCIM+LSTM Model', linestyle='dashed')
  plt.plot(predictions_self, label='Tranditrional LSTM Model', linestyle='dotted')
  plt.legend()
  plt.title(f"Stock Price Prediction for {target_stock} using LSTM")
  # plt.show()
  st.pyplot(plt)

def evaluation_metrics(y_test_actual, predictions_others, predictions_self):
  mse_others = mean_squared_error(y_test_actual, predictions_others)
  mae_others = mean_absolute_error(y_test_actual, predictions_others)
  mse_self = mean_squared_error(y_test_actual, predictions_self)
  mae_self = mean_absolute_error(y_test_actual, predictions_self)
  metrics_df = pd.DataFrame({
    "Model": ["Others", "Self"],
    "MSE": [mse_others, mse_self],
    "MAE": [mae_others, mae_self],
    "RMSE": [np.sqrt(mse_others), np.sqrt(mse_self)]
  })
  return metrics_df
def visualize_prediction(actual, predict, future):
  df_visualize = pd.DataFrame()

  df_visualize['actual'] = np.concatenate(
    [np.array(actual).flatten(), np.array(future).flatten()])
  df_visualize['predict'] = np.concatenate(
    [np.array(predict).flatten(), np.array(future).flatten()])

  x = np.arange(len(df_visualize))

  # Create interactive Plotly figure
  fig = go.Figure()

  # Actual values (Blue Line)
  fig.add_trace(go.Scatter(
    x=x[:-len(future)], y=df_visualize['actual'][:-len(future)],
    mode='lines', name='Actual Value', line=dict(color='blue')
  ))

  # Historical predictions (Yellow Dashed Line)
  fig.add_trace(go.Scatter(
    x=x[:-len(future)], y=df_visualize['predict'][:-len(future)],
    mode='lines', name='Historical Prediction',
    line=dict(color='yellow', dash='dash')
  ))

  # Future predictions (Red Line)
  fig.add_trace(go.Scatter(
    x=x[-len(future):], y=df_visualize['predict'][-len(future):],
    mode='lines', name='Future Prediction', line=dict(color='red')
  ))

  # Layout settings
  fig.update_layout(
    title="📈 LSTM Actual Vs Prediction",
    xaxis_title="Time Steps",
    yaxis_title="Stock Price",
    template="plotly_white",  # Giao diện sáng
    plot_bgcolor="rgba(0,0,0,0)",  # Không có nền
    paper_bgcolor="rgba(0,0,0,0)",  # Không có nền
    hovermode="x",
    legend=dict(
      orientation="h",  # Đặt ngang
      yanchor="bottom",
      y=-0.3,  # Điều chỉnh vị trí bên dưới
      xanchor="center",
      x=0.5,
      bordercolor="grey",  # Viền xém
      borderwidth=1,
      tracegroupgap=50
    )
  )

  # Display in Streamlit
  st.plotly_chart(fig, use_container_width=True)
def demonstration(cheap, expensive, data):
  cheap_list = [c.strip() for c in cheap.split(',')]
  selected_tickers = cheap_list + [expensive]
  data_filtered = data[data['ticker'].isin(selected_tickers)]
  if data_filtered.empty:
    st.warning("No data available for selected stocks.")
    return
  price_data = data_filtered.pivot(index='date', columns='ticker', values='close')
  if price_data.isnull().values.any():
    st.warning("The data has a missing value, please check again.")
    return
  returns = price_data.pct_change().dropna()
  # mean_returns = returns.mean()
  # volatility = returns.std()
  # sharpe_ratios = mean_returns / volatility
  mean_returns = returns.mean().to_frame(name="Mean Returns")
  volatility = returns.std().to_frame(name="Volatility")
  sharpe_ratios = (mean_returns["Mean Returns"] / volatility["Volatility"]).to_frame(name="Sharpe Ratio")
  cheap_returns = returns[cheap_list].mean(axis=1)
  expensive_returns = returns[expensive]
  diff_returns = cheap_returns - expensive_returns
  cumulative_diff = (1 + diff_returns).cumprod()
  fig = go.Figure()

  # Thêm đường lợi nhuận tích lũy
  fig.add_trace(go.Scatter(
    x=cumulative_diff.index, y=cumulative_diff,
    mode='lines', name=f"{', '.join(cheap_list)} vs {expensive}",
    line=dict(color='blue', width=3)
  ))

  # Thêm đường baseline
  fig.add_trace(go.Scatter(
    x=cumulative_diff.index, y=[1] * len(cumulative_diff),
    mode='lines', name='Baseline',
    line=dict(color='red', width=2, dash='dash')
  ))

  # Cấu hình layout
  fig.update_layout(
    title=dict(
      text=f"Comparison: {', '.join(cheap_list)} vs {expensive}",
      font=dict(size=18, family="Arial", color="white"),
    ),
    xaxis=dict(
      title="Date",
      tickangle=30,
      showgrid=True, gridcolor="grey",
      title_font=dict(size=14, family="Arial", color="white"),
      tickfont=dict(size=12, color="white")
    ),
    yaxis=dict(
      title="Cumulative Return Difference",
      showgrid=True, gridcolor="grey",
      title_font=dict(size=14, family="Arial", color="white"),
      tickfont=dict(size=12, color="white")
    ),
    legend=dict(
      orientation="h",
      yanchor="bottom", y=-0.4,  # Cách xa biểu đồ
      xanchor="left", x=0.25,  # Kéo dài legend
      bgcolor='rgba(0,0,0,0)',  # Nền trong suốt
      bordercolor="grey", borderwidth=1.5,  # Viền xám
      font=dict(size=13, color="white")
    ),
    paper_bgcolor='rgba(0,0,0,0)',  # Nền trong suốt
    plot_bgcolor='rgba(0,0,0,0)',  # Biểu đồ trong suốt
    margin=dict(l=40, r=40, t=60, b=80)  # Cách lề dưới rộng hơn
  )

  # Hiển thị trên Streamlit
  st.plotly_chart(fig, use_container_width=True)
  col1, col2, col3 = st.columns(3)
  with col1:
    st.subheader("Mean Returns:")
    st.write(mean_returns)
  with col2:
    st.subheader("Volatility:")
    st.write(volatility)
  with col3:
    st.subheader("Sharpe Ratios:")
    st.write(sharpe_ratios)

df_distionaty_stock = distionaty_stock(data)

st.title('Welcome to STOCK INVESTMENT RECOMMENDATION APP 💰')
st.write("""Our app helps you discover low-cost, high-interest stocks in the US, giving you access to the best investment opportunities. With advanced price prediction technology, we provide insights into future stock trends, helping you make informed decisions with confidence.
Start investing smarter today—let data-driven insights guide your journey to financial growth! 🚀📈""")

df_recommendation = get_recommendation(output_LCIM, df_distionaty_stock)

st.markdown("<h1 style='font-size: 24px;'>1. The stock list have high interest and low cost:</h1>", unsafe_allow_html=True)
visualize_data(df_recommendation)

st.markdown("<p>To prove the displayed results are correct, please check as follows: <i>With the same investment amount, please choose a group of cheap stocks and an expensive stock, based on actual data, we will analyze their profitability for you.</i></p>", unsafe_allow_html=True)
col1, col2 = st.columns([2, 1])  # Tỉ lệ độ rộng 2:1 để căn chỉnh đẹp hơn
with col1:
    cheap = st.text_input("Cheap stocks (Ex: AAA, BBB, CCC):")
with col2:
    expensive = st.text_input("Expensive stock:")
if cheap and expensive:
  demonstration(cheap, expensive, data)

st.markdown("<h1 style='font-size: 24px;'>2. Stock Information:</h1>", unsafe_allow_html=True)
stock = st.selectbox('Choose a stock ticker symbol:', [""] + list(df_recommendation['Ticket Name'].unique()))
if stock:
  df_result = process_output_LCIM_model(output_LCIM, stock, df_distionaty_stock)
  st.write('The stock sets of ',stock,':')
  visualize_data(df_result)
  st.write('Push the below button to display the prediction in the next 15 days of ',stock,' stock.')
  st.markdown(
    """
    <style>
    .stButton>button {
        display: flex;
        justify-content: center;
        align-items: center;
        background: linear-gradient(135deg, #007BFF, #00C6FF);
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 14px 28px;
        border-radius: 12px;
        border: none;
        cursor: pointer;
        transition: all 0.3s ease;
        text-align: center;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2);
        width: 180px;
        margin: 0 auto;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #0056b3, #0099cc);
        transform: scale(1.08);
        box-shadow: 0px 6px 10px rgba(0, 0, 0, 0.3);
    }
    </style>
    """,
    unsafe_allow_html=True)
  if st.button("🚀 Prediction"):

    # Call your prediction functions
    df_top10_stock = get_df_top10_stock(data, df_result)
    df_correlation = get_df_correlation_stock(df_top10_stock, stock)

    if df_correlation.shape[1] != 1:
      actual, predictions, future = LSTM_proposed_model(stock, df_correlation, 10, 15)
    else:
      actual, predictions, future = LSTM_traditional_model(stock, df_correlation, 10, 15)
    st.write("")
    visualize_prediction(actual, predictions, future)

# Run:  streamlit run forecast_evaluation.py --server.port 8502
