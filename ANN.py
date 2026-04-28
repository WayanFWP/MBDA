import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

from module import Architecture

from module.Trainner import Trainner
from module.DataLoader import DataLoader

data = DataLoader(path="./data/ANN", type="EMNIST", batch=64)
architecture = Architecture.ANN(input=28*28, hidden_size=[128, 64, 32], output=47)
trainer = Trainner(model=architecture, optimizer="adam", lr=1e-3, criterion=nn.CrossEntropyLoss())

train, test = data.Loader()
test_data, test_labels = data.extract(test)

trainer.fit(train, epochs=5, verbose=True)
loss, accuracy = trainer.loss_history, trainer.accuracy_history

test_accuracy, prediction, true_labels, x_test = trainer.evaluate(test)