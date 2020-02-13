from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from keras.losses import *

param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    # {'n_estimators': [30, 180, 220], 'max_features': [16, 32, 64, 128]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'epochs': [100, 50, 200], 'batch_size': [64, 32, 128], 'dropout_rate': [0.005, 0.01], 'l': [0.01, 0.0001, 0.05], 'lr': [0.001, 0.01, 0.003],
     'loss': [hinge, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error]},
]
# forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training
grid_search = GridSearchCV(model, param_grid, cv=5,
                           scoring='neg_mean_squared_error', return_train_score=True, n_jobs=-1)
