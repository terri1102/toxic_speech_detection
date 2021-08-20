import torch
import random
import numpy as np
import os
import pandas as pd
from sklearn.metrics import classification_report
from build_dataset import main.test
#import build_dataset.test


path_to_output_file = './results/output.txt'  # path to the file with prediction probabilities


#현재 문제: build_dataset에서 만든 test 변수(데이터프레임)를 여기로 불러 올 수 없음
#1. test를 csv로 저장하기
#2. 모듈 간에 통용되는 변수 만들기...?

labels_test = test['label']  # true labels

probs_test = pd.read_csv(path_to_output_file, header=None)[0]  # prediction probabilities
threshold = 0.5   # you can adjust this threshold for your own dataset
preds_test=(probs_test>=threshold).astype('uint8') # predicted labels using the above fixed threshold

print(classification_report(labels_test, preds_test))
