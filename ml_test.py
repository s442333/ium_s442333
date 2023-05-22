import torch
import pandas as pd
from onnx2torch import convert
from genDataset import genDataset
import sys

from os.path import isfile

build_id = sys.argv[1] if len(sys.argv) >= 2 else 101

Xtest, Ytest = genDataset("weather.csv.test")

model = convert("model.onnx")

total = 0
correct = 0
dic = {
    "real_value": [],
    "predicted_value": []
}

for x, y in zip(Xtest, Ytest):
    result = 1 if model(x).squeeze() >= 0.5 else 0
    result_true = int(y.item())

    if result == result_true:
        correct += 1
    total += 1

    dic['real_value'].append(result_true)
    dic['predicted_value'].append(result)

df = pd.DataFrame(dic)
df.to_csv('ml_result.csv', index=False)

m = None
if isfile('metrics.csv'):
    m = pd.read_csv("metrics.csv")
    m = pd.concat([m, pd.DataFrame(
        {"build": [build_id], "accuracy": [100 * correct/total]})])
else:
    m = pd.DataFrame({"build": [build_id], "accuracy": [
                     100*correct/total]})

m.to_csv('metrics.csv', index=False)


print(f'Accuracy: {100*correct/total}')
