import os
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Load data from CSV file
file_path = os.path.join(os.path.dirname(__file__), "data.csv")
data = pd.read_csv(file_path)

# Convert date column to datetime format and set as index
data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date')

def main():
    st.title("Khmer(riel) Stock Price Prediction")

    # Add sidebar for date range input
    st.sidebar.subheader("Select date range")
    start_date = st.sidebar.date_input("Start date", data.index.min())
    end_date = st.sidebar.date_input("End date", data.index.max())

    # Filter the data based on user input
    filtered_data = data.loc[start_date:end_date]

    # Display the filtered data in a table
    st.subheader("Khmer(riel) Stock Price Dataset")
    st.dataframe(filtered_data)

    # Fit an ETS model to the filtered data
    model = ExponentialSmoothing(filtered_data['Price'], seasonal_periods=7, trend='add', seasonal='add')
    model_fit = model.fit()

    # Allow the user to select the number of days to predict
    st.subheader("ETS Model Predictions")
    num_days = st.slider("Select the number of days to predict:", 1, 30, 7)

    # Create future dates for prediction
    last_date = filtered_data.index.max()
    future_dates = pd.date_range(start=last_date, periods=num_days + 1, freq=filtered_data.index.freq)[1:]

    # Make predictions using the ETS model
    predictions = model_fit.predict(start=len(filtered_data), end=len(filtered_data) + num_days - 1)

    # Create a dataframe for the predicted prices
    predicted_prices = pd.DataFrame({"Date": future_dates, "Predicted Price": predictions})

    # Concatenate the original and predicted dataframes
    combined_data = pd.concat([filtered_data, predicted_prices.set_index('Date')])

    # Plot the actual and predicted prices
    fig, ax = plt.subplots()
    combined_data['Price'].plot(ax=ax, label='Actual Price')
    combined_data['Predicted Price'].plot(ax=ax, label='Predicted Price')
    ax.legend()

    # Show the predicted prices
    predicted_prices_table = pd.concat([filtered_data.tail(1)['Price'], predicted_prices['Predicted Price']])
    predicted_prices_table = predicted_prices_table.rename({predicted_prices_table.index[0]: 'Last Actual Price'})
    st.write("Predicted Prices")
    st.write(predicted_prices_table)

    # Display the line chart
    st.pyplot(fig)

if __name__ == "__main__":
    main()