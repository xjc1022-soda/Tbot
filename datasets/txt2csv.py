import pandas as pd

data_ecl = pd.read_csv('./datasets/electricity.txt', parse_dates=True, sep=';', decimal=',', index_col=0)
data_ecl.to_csv('./datasets/electricity.csv')