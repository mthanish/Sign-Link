import pickle

with open("data_processing/isl_model.pkl", "rb") as f:
    model = pickle.load(f)

print(type(model))
