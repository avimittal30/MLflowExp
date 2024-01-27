import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import os

def preprocessed_data(df):
    df.set_index(['CustomerId'])
    train,test=train_test_split(df,test_size=0.2, random_state=40)
    return train, test