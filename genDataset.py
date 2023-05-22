import pandas as pd
from torch import tensor


def genDataset(csv_path):
    df = pd.read_csv(csv_path)
    df = df[['Rainfall', 'Humidity9am', 'Humidity3pm', 'RainToday',
             'Pressure3pm', 'Temp9am', 'Temp3pm', 'Pressure9am', 'RainTomorrow']]
    df = df.dropna(how='any')

    df.RainToday[df.RainToday == 'Yes'] = 1.0
    df.RainToday[df.RainToday == 'No'] = 0.0
    df.RainToday = pd.to_numeric(df.RainToday)
    df.RainTomorrow[df.RainTomorrow == 'Yes'] = 1.0
    df.RainTomorrow[df.RainTomorrow == 'No'] = 0.0
    df.RainTomorrow = pd.to_numeric(df.RainTomorrow)

    Y, X = df[['RainTomorrow']], df.drop(
        ['RainTomorrow'], axis=1, inplace=False)

    X, Y = tensor(X.values), tensor(Y.values)

    X = X.float()
    Y = Y.float().squeeze()

    return X, Y
