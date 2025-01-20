import joblib
from sklearn.linear_model import LinearRegression

model = joblib.load('model/linear_regression_model.pkl')
print(model)
