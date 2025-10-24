# import pickle
# from sklearn.ensemble import RandomForestClassifier
# import numpy as np

# # ------------------ Dummy training data ------------------
# # Each "hand landmark" vector will have 42 features (21 landmarks * 2 for x, y)
# # We'll just generate random data for testing
# X_dummy = np.random.rand(10, 42)  # 10 sample hands
# y_dummy = ['A','B','C','D','E','F','G','H','I','J']  # dummy labels

# # ------------------ Train a simple model ------------------
# model = RandomForestClassifier()
# model.fit(X_dummy, y_dummy)

# # ------------------ Save the model ------------------
# MODEL_PATH = r'D:\Zaids Work\SignLink\model\isl_model.pkl'
# with open(MODEL_PATH, 'wb') as f:
#     pickle.dump(model, f)

# print(f"Dummy model saved at: {MODEL_PATH}")
