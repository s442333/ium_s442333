import torch
import torch.nn as nn

import sys

from genDataset import genDataset

import mlflow

Xtrain, Ytrain = genDataset("weather.csv.train")

def calculate_accuracy(y_true, y_pred):
    predicted = y_pred.ge(.5).view(-1)
    return (y_true == predicted).sum().float() / len(y_true)


def round_tensor(t, decimal_places=6):
    return round(t.item(), decimal_places)

class Weather(nn.Module):
    def __init__(self, n_features):
        super(Weather, self).__init__()

        self.fc1 = nn.Linear(n_features, 32)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(16, 1)
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)

        out = self.fc2(out)
        out = self.relu2(out)

        out = self.fc3(out)
        out = self.sigm(out)

        return out


lr = 0.001
epochs = int(sys.argv[1]) if len(sys.argv) == 2 else 501

mlflow.set_experiment("s442333")

def main(lr, epochs, run):
    epochs = int(epochs)
    n_features = 8
    model = Weather(n_features)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs):
        y_pred = model(Xtrain)

        y_pred = torch.squeeze(y_pred)
        train_loss = criterion(y_pred, Ytrain)

        if epoch % 10 == 0 or epoch == 500:
            train_acc = calculate_accuracy(Ytrain, y_pred)

            mlflow.log_metric("train.loss", round_tensor(train_loss))
            mlflow.log_metric("train.acc", round_tensor(train_acc))

            print(f'''epoch {epoch}
                Train set - loss: {round_tensor(train_loss)}, accuracy: {round_tensor(train_acc)}''')

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

    mlflow.log_param("epochs", epochs)
    mlflow.log_param("lr", lr)


    dummy_input = torch.randn(1, n_features)
    torch.onnx.export(model, dummy_input, "model.onnx")

with mlflow.start_run() as run:
    main(lr, epochs, run)