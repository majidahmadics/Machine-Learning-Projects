# Asset Pricing Using Different Deep Learning Models

## Table of Contents
- [Asset Pricing Prediction with RNN](#introduction)
- [Asset Pricing Prediction with LSTM](#installation)

# Asset Pricing Prediction with RNN

This project demonstrates how to build a simple Recurrent Neural Network (RNN) model for asset pricing prediction using historical stock price data. The RNN model is implemented using TensorFlow's Keras API.

## Requirements

- Python 3.x
- TensorFlow
- pandas
- NumPy
- scikit-learn
- matplotlib

## Import Libraries:

Import necessary libraries such as NumPy, pandas, matplotlib, scikit-learn, and TensorFlow. These libraries are used for data manipulation, visualization, preprocessing, and building neural network models.

## Load and Preprocess the Data:

Read the dataset file 'stock_data.csv' containing historical stock price data using pandas read_csv function.
Extract the 'Close' prices from the dataset and reshape them into a column vector using NumPy's reshape function.
Scale the prices to a range between 0 and 1 using MinMaxScaler from scikit-learn. This preprocessing step helps improve the convergence and performance of the neural network model.

## Prepare the Data for Training:

Define a window size (e.g., 20) indicating the number of previous time steps to consider as input features for each prediction.
Create input sequences (X) and corresponding target values (y) by iterating over the scaled prices data. For each iteration, extract a window of previous prices as input features (X) and the next price as the target value (y).
Convert the input sequences (X) and target values (y) into NumPy arrays for compatibility with TensorFlow.

## Define the RNN Model:

Construct a Sequential model using TensorFlow's Keras API. A Sequential model allows you to build neural network architectures layer by layer.
Add two SimpleRNN layers to the model with 50 units each. The return_sequences=True argument in the first layer indicates that it should return the full sequence of outputs instead of just the last output. This is necessary when stacking recurrent layers.
Introduce dropout regularization with a rate of 0.2 after each RNN layer to prevent overfitting.
Add a Dense output layer with a single unit, which will output the predicted price.

## Compile the Model:

Compile the model using the Adam optimizer and mean squared error loss function. The Adam optimizer is an efficient optimization algorithm commonly used for training neural networks.
## Train the Model:

Train the model on the input sequences (X) and target values (y) for 50 epochs with a batch size of 32. During training, the model learns to minimize the mean squared error between the actual and predicted prices.
## Make Predictions:

Use the trained model to make predictions on the input sequences (X) to obtain the predicted prices.

## Inverse Transform the Predictions:

Inverse transform the predicted prices back to their original scale using the MinMaxScaler's inverse_transform method. This step restores the predicted prices to their original range for better interpretation and comparison.

## Visualize the Results:

Plot the actual prices (in blue) and predicted prices (in red) over time using matplotlib. This visualization allows you to visually compare the model's predictions with the actual prices.
By following these steps, the code demonstrates how to build, train, and evaluate a simple RNN model for asset pricing prediction using historical stock price data.

# Asset Pricing Prediction with LSTM

## 1. Data Collection:

Import necessary libraries: yfinance, numpy, pandas, MinMaxScaler from sklearn.preprocessing, mean_squared_error from sklearn.metrics, Sequential and Dense from keras.models, LSTM from keras.layers, and plotly.graph_objects as go.
Set the parameters: ticker (stock ticker symbol), start_date, and end_date for the desired time period.
Define a function fetch_stock_data() to download stock data using Yahoo Finance API (yf.download()). Return the downloaded data.

## 2. Data Preprocessing:

Define a function preprocess_data() to preprocess the data.
Normalize the stock data using MinMaxScaler() from sklearn.preprocessing.
Create sequences of data for training and testing.
Split the data into input features (X) and target labels (y).
Split the data into training and testing sets.
Return preprocessed data, training, and testing sets along with the scaler object.

## 3. Model Building:

Define a function build_lstm_model() to create an LSTM model using Sequential() from keras.models.
Add an LSTM layer with specified units and input shape.
Add a Dense layer with 1 unit.
Compile the model using adam optimizer and mean_squared_error loss function.
Return the compiled model.

## 4. Model Training:

Define a function train_model() to train the LSTM model.
Fit the model to the training data with specified epochs and batch size.

## 5. Model Evaluation:

Define a function evaluate_model() to evaluate the trained model.
Make predictions on the test data.
Inverse transform the predictions and actual values using the scaler object.
Calculate the Root Mean Squared Error (RMSE) between the actual and predicted values.
Return RMSE, predictions, and actual values.

## 6. Main Function:

Define a main function main() to orchestrate the entire process.
Call the data collection function to fetch stock data.
Define the sequence length (seq_length) for creating sequences of data.
Preprocess the data using the defined function.
Build the LSTM model using the defined function.
Train the model using the defined function.
Evaluate the model's performance using the defined function.
Print the test RMSE and accuracy.
Plot the actual and predicted stock prices using Plotly.

## 7. Execute the Script:

Check if the script is being run directly (__name__ == "__main__").
Call the main function to execute the entire process.

Note: Ensure that you have installed all the required libraries (yfinance, numpy, pandas, scikit-learn, keras, and plotly) before executing the script. Additionally, make sure you have a stable internet connection to fetch stock data using Yahoo Finance API.
