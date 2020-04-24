import pickle
import numpy as np

file1 = open("weightvector_offensive.save", "rb")
weight_matrix_offensive = pickle.load(file1)

file2 = open("weightvector_defensive.save", "rb")
weight_matrix_defensive = pickle.load(file2)

print("offensive weight vector:")
print(weight_matrix_offensive)

print("defensive weight vector:")
print(weight_matrix_defensive)



