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

## Descriptive Statistics
Descriptive statistics of the dataset, including measures such as mean, standard deviation, minimum, and maximum, are displayed using Pandas.

print(data.describe())

## Column Information
Information about all the columns in the dataset is printed to provide an overview of the available data.

print(data.info())
