# # import library
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn import svm, datasets
# from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split
# from sklearn.utils import shuffle
# from sklearn.metrics import classification_report,accuracy_score
# import pandas as pd
# from sklearn.datasets import load_iris

# download dataset

from medmnist import PneumoniaMNIST,PathMNIST
dataset = PneumoniaMNIST(split="train", download=True)
dataset = PneumoniaMNIST(split="val", download=True)
dataset = PneumoniaMNIST(split="test", download=True)

dataset = PathMNIST(split="train", download=True)
dataset = PathMNIST(split="val", download=True)
dataset = PathMNIST(split="test", download=True)

print(dataset)