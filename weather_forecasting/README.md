# Delhi Climate Analysis and Weather Forecasting

This repository contains a Python script for analyzing historical climate data in Delhi and forecasting future weather using the Facebook Prophet model. The dataset used in this analysis is named "DailyDelhiClimateTrain.csv."

## Table of Contents
- [Libraries and Dataset](#libraries-and-dataset)
- [Descriptive Statistics](#descriptive-statistics)
- [Column Information](#column-information)
- [Mean Temperature Over the Years](#mean-temperature-over-the-years)
- [Temperature-Humidity Relationship](#temperature-humidity-relationship)
- [Analyzing Temperature Change](#analyzing-temperature-change)
- [Weather Forecasting using Prophet](#weather-forecasting-using-prophet)

## Libraries and Dataset

The necessary Python libraries—Pandas, NumPy, Matplotlib, Seaborn, and Plotly Express—are imported to facilitate data analysis and visualization. The dataset, "DailyDelhiClimateTrain.csv," is loaded using Pandas.
```python

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

data = pd.read_csv("DailyDelhiClimateTrain.csv")
```
## Descriptive Statistics

Descriptive statistics of the dataset, including measures such as mean, standard deviation, minimum, and maximum, are displayed using Pandas.
```python

print(data.describe())
```

## Column Information

Information about all the columns in the dataset is printed to provide an overview of the available data.
```python

print(data.info())
```

## Mean Temperature Over the Years

A Plotly Express line chart is created to visualize the mean temperature in Delhi over the years.

```python
figure = px.line(data, x="date", y="meantemp", title='Mean Temperature in Delhi Over the Years')
figure.show()
```

## Temperature-Humidity Relationship

Plotly Express scatter plot with a trendline is generated to explore the relationship between temperature and humidity.

```python

figure = px.scatter(data_frame=data, x="humidity", y="meantemp", size="meantemp", trendline="ols", title="Relationship Between Temperature and Humidity")
figure.show()
```

## Analyzing Temperature Change

The script converts the date column to datetime format and extracts the year and month data for further analysis. A Seaborn line plot is created to visualize temperature changes in Delhi over the years.

```python

data["date"] = pd.to_datetime(data["date"], format='%Y-%m-%d')
data['year'] = data['date'].dt.year
data["month"] = data["date"].dt.month

plt.style.use('fivethirtyeight')
plt.figure(figsize=(15, 10))
plt.title("Temperature Change in Delhi Over the Years")
sns.lineplot(data=data, x='month', y='meantemp', hue='year')
plt.show()
```

## Weather Forecasting using Prophet

The script prepares the data for weather forecasting using the Facebook Prophet model. The data is converted into the required format, and the Prophet model is trained to make future predictions.

```python

forecast_data = data.rename(columns={"date": "ds", "meantemp": "y"})

from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly

model = Prophet()
model.fit(forecast_data)
forecasts = model.make_future_dataframe(periods=365)
predictions = model.predict(forecasts)
plot_plotly(model, predictions)
```

Feel free to explore the code, analyze the insights, and use it as a reference for similar analyses or forecasting tasks. If you have any questions or suggestions, feel free to open an issue or reach out through the provided contact information.
