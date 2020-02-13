

from keras.wrappers.scikit_learn import KerasRegressor
import math
from keras import backend as K
from keras.losses import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from keras.losses import logcosh
from keras import regularizers
from keras.layers.core import Dropout
from keras.models import Sequential
from keras.layers import Activation, Dense


def rmse(y_true, y_pred):
    from keras import backend
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

# coefficient of determination (R^2) for regression  (only for Keras tensors)


def r_square(y_true, y_pred):
    from keras import backend as K
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return abs(1 - SS_res/(SS_tot + K.epsilon()))


def create_model(epochs=100, batch_size=32, dropout_rate=0.005, l=0.0001, lr=0.001, loss=mean_squared_error):
    NN_model = Sequential()
    # The Input Layer :
    NN_model.add(Dense(6, kernel_initializer='normal', input_dim=235, activation='relu'))
    # The Hidden Layers :
    #NN_model.add(Dense(1, kernel_initializer='normal',activation='relu',kernel_regularizer=regularizers.l2(l)))
    #NN_model.add(Dense(1, kernel_initializer='normal',activation='relu',kernel_regularizer=regularizers.l2(l)))
    #NN_model.add(Dropout(rate =dropout_rate))
    #NN_model.add(Dense(16, kernel_initializer='normal',activation='relu',kernel_regularizer=regularizers.l2(l)))
    NN_model.add(Dense(4, kernel_initializer='normal', activation='relu',
                       kernel_regularizer=regularizers.l2(l)))
    # The Output Layer :
    NN_model.add(Dense(1, kernel_initializer='normal', activation='linear'))

    from keras import optimizers
    adam = optimizers.Adam(lr=lr)

    # Compile the network :
    NN_model.compile(loss=loss, optimizer=adam, metrics=['mse'])

    # NN_model.summary()

    return NN_model


def fun():
    NN_model = create_model()
    NN_model.summary()

    model = KerasRegressor(build_fn=create_model)
    param_grid = [
        # try 12 (3×4) combinations of hyperparameters
        # {'n_estimators': [30, 180, 220], 'max_features': [16, 32, 64, 128]},
        # then try 6 (2×3) combinations with bootstrap set as False
        {'epochs': [100, 50, 200], 'batch_size': [64, 32, 128], 'dropout_rate': [0.005, 0.01], 'l': [0.01, 0.0001, 0.05], 'lr': [0.001, 0.01, 0.003],
         'loss': [hinge, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error]},
    ]
    # forest_reg = RandomForestRegressor(random_state=42)
    # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
    return GridSearchCV(model, param_grid, cv=5,
                        scoring='neg_mean_squared_error', return_train_score=True, n_jobs=-1)
