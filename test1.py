

######### เอาไว้ดู data เฉยๆ ##########
import pickle
import numpy as np

with open('data_left_hand.pickle', 'rb') as file:
    data = pickle.load(file)

print(data)