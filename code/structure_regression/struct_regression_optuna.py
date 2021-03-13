import pandas as pd
import numpy as np
import numpy.matlib
import math
import random
import json
from json import JSONEncoder

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.figure as figure

from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn import metrics
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel, Matern, DotProduct
from sklearn.cross_decomposition import PLSRegression
from sklearn import model_selection
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from dcekit.validation import double_cross_validation
from dcekit.generative_model import GMR

import optuna
import optuna.integration.lightgbm as lgb
import lightgbm

path = Path('__file__').parent
dir_name = path / '..\data'
save_dir = path / '..\save_struct_0313'

struct_data = pd.read_csv(dir_name/'struct_info_sbu_org.csv', index_col=0)
discriptor_data = pd.read_csv(dir_name/'struct_disc_sbu_org.csv', index_col=0).fillna(0)

aim_y = struct_data.columns
dataset_x = discriptor_data.copy()

model_names = ['PLS', 'Ridge', 'LASSO', 'EN', 'SVR(Linear)', 'SVR(rbf)', 'RandomForest', 'GP', 'LightGBM']

fold_number = 10
max_number_of_components = 20
do_autoscaling = True
random_state=3

score_metrics = pd.DataFrame(np.zeros((len(model_names), 3)), index=model_names, columns=['r2', 'MAE', 'RMSE'])

def objective_pls(trial):
    n_components = trial.suggest_int('n_components', 1, max_number_of_components)

    regr = PLSRegression(n_components=n_components)
    score = cross_val_score(regr, autoscaled_x_train, autoscaled_y_train, cv=fold_number, scoring='r2')
    r2_mean = score.mean()
    print(r2_mean)

    return r2_mean

def optimised_pls(study):
    optimised_regr = PLSRegression(n_components=study.best_params['n_components'])
    return optimised_regr

def objective_ridge(trial):
    alpha = trial.suggest_float('alpha', 0.0, 2.0)

    regr = Ridge(alpha=alpha)
    score = cross_val_score(regr, autoscaled_x_train, autoscaled_y_train, cv=fold_number, scoring='r2')
    r2_mean = score.mean()
    print(r2_mean)

    return r2_mean

def optimised_ridge(study):
    optimised_regr = Ridge(alpha=study.best_params['alpha'])
    return optimised_regr

def objective_lasso(trial):
    alpha = trial.suggest_float('alpha', 0.0, 2.0)

    regr = Lasso(alpha=alpha)
    score = cross_val_score(regr, autoscaled_x_train, autoscaled_y_train, cv=fold_number, scoring='r2')
    r2_mean = score.mean()
    print(r2_mean)

    return r2_mean

def optimised_lasso(study):
    optimised_regr = Lasso(alpha=study.best_params['alpha'])
    return optimised_regr

def objective_en(trial):
    alpha = trial.suggest_float('alpha', 0.0, 2.0)
    l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)

    regr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    score = cross_val_score(regr, autoscaled_x_train, autoscaled_y_train, cv=fold_number, scoring='r2')
    r2_mean = score.mean()
    print(r2_mean)

    return r2_mean

def optimised_en(study):
    optimised_regr = ElasticNet(alpha=study.best_params['alpha'], l1_ratio=study.best_params['l1_ratio'])
    return optimised_regr


def objective_svr_lin(trial):
    svr_c = trial.suggest_loguniform('C', 1e-2, 1e2)
    epsilon = trial.suggest_loguniform('epsilon', 1e-2, 1e2)

    regr = SVR(kernel='linear', C=svr_c, epsilon=epsilon)
    score = cross_val_score(regr, autoscaled_x_train, autoscaled_y_train, cv=fold_number, scoring='r2')
    r2_mean = score.mean()
    print(r2_mean)

    return r2_mean

def optimised_svr_lin(study):
    optimised_regr = SVR(kernel='linear', C=study.best_params['C'], epsilon=study.best_params['epsilon'])
    return optimised_regr

def objective_svr_rbf(trial):
    svr_c = trial.suggest_loguniform('C', 1e-2, 1e2)
    epsilon = trial.suggest_loguniform('epsilon', 1e-2, 1e2)
    gamma = trial.suggest_loguniform('gamma', 1e-2, 1e2)

    regr = SVR(kernel='rbf', C=svr_c, epsilon=epsilon, gamma=gamma)
    score = cross_val_score(regr, autoscaled_x_train, autoscaled_y_train, cv=fold_number, scoring='r2')
    r2_mean = score.mean()
    print(r2_mean)

    return r2_mean

def optimised_svr_rbf(study):
    optimised_regr = SVR(kernel='rbf', C=study.best_params['C'], epsilon=study.best_params['epsilon'], gamma=study.best_params['gamma'])
    return optimised_regr


def objective_rf(trial):
    max_depth = trial.suggest_int('max_depth', 1, 15)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 7)
    max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 1, 50)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    

    regr = RandomForestRegressor(max_depth=max_depth, min_samples_split=min_samples_split, max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf)
    score = cross_val_score(regr, autoscaled_x_train, autoscaled_y_train, cv=fold_number, scoring='r2')
    r2_mean = score.mean()
    print(r2_mean)

    return r2_mean

def optimised_rf(study):
    optimised_regr = RandomForestRegressor(
        max_depth=study.best_params['max_depth'],
        min_samples_split=study.best_params['min_samples_split'],
        max_leaf_nodes=study.best_params['max_leaf_nodes'],
        min_samples_leaf=study.best_params['min_samples_leaf']
        )
    return optimised_regr

def objective_gp(trial):
    gp_kernels = [
        ConstantKernel() * DotProduct() + WhiteKernel(),
        ConstantKernel() * RBF() + WhiteKernel(),
        ConstantKernel() * RBF() + WhiteKernel() + ConstantKernel() * DotProduct(),
        # ConstantKernel() * Matern(nu=1.5) + WhiteKernel(),
        # ConstantKernel() * Matern(nu=1.5) + WhiteKernel() + ConstantKernel() * DotProduct(),
        # ConstantKernel() * Matern(nu=0.5) + WhiteKernel(),
        # ConstantKernel() * Matern(nu=0.5) + WhiteKernel() + ConstantKernel() * DotProduct(),
        # ConstantKernel() * Matern(nu=2.5) + WhiteKernel(),
        # ConstantKernel() * Matern(nu=2.5) + WhiteKernel() + ConstantKernel() * DotProduct()
    ]
    kernel = trial.suggest_categorical('kernel', gp_kernels)

    regr = GaussianProcessRegressor(alpha=0, kernel=kernel)
    score = cross_val_score(regr, autoscaled_x_train, autoscaled_y_train, cv=fold_number, scoring='r2')
    r2_mean = score.mean()
    print(r2_mean)

    return r2_mean

def optimised_gp(study):
    optimised_regr = GaussianProcessRegressor(alpha=0, kernel=study.best_params['kernel'])
    return optimised_regr

def objective_lgb(x, y):

    x_train_a, x_valid_a, y_train, y_valid = train_test_split(x, y, train_size=0.8, random_state=5, shuffle=True)

    drop_col = x_train_a.std(axis=0, ddof=1)!=0
    x_train = x_train_a.loc[:, drop_col] 
    x_valid = x_valid_a.loc[:, drop_col] 

    autoscaled_x_train = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0, ddof=1)
    autoscaled_y_train = (y_train - y_train.mean()) / y_train.std(ddof=1)
    autoscaled_x_valid = (x_valid - x_train.mean(axis=0)) / x_train.std(axis=0, ddof=1)
    autoscaled_y_valid = (y_valid - y_train.mean()) / y_train.std(ddof=1)

    trains = lightgbm.Dataset(autoscaled_x_train.values, autoscaled_y_train.values)
    valids = lightgbm.Dataset(autoscaled_x_valid.values, autoscaled_y_valid.values)

    params = {
        'objective':'mean_squared_error',
        'metric':'rmse'
    }
    best_params, history = {}, []
    model = lgb.train(
        params, trains, valid_sets=valids,
        verbose_eval=False,
        # num_boost_round=100,
        # early_stopping_rounds=5,
        # best_params=best_params,
        # tuning_history=history,
        # force_row_wise=True
    )
    best_params=model.params
    return best_params

def optimised_lgb(autoscaled_x_train, autoscaled_y_train, autoscaled_x_test, autoscaled_y_test, best_params):
    trains = lightgbm.Dataset(autoscaled_x_train.values, autoscaled_y_train.values)
    
    model = lightgbm.train(best_params, trains, num_boost_round=100)
    estimated_y_train = model.predict(autoscaled_x_train.values)
    estimated_y_test = model.predict(autoscaled_x_test.values)

    return estimated_y_train, estimated_y_test

def evaluation(raw_y, estimated_y, data_type, savefig=False):
    raw_y_array = raw_y.values
    estimated_y_array = estimated_y.reshape((estimated_y.shape[0]))
    # y_dataframe = pd.DataFrame(np.c_[raw_y_array, estimated_y], index=raw_y.index, columns=['actual', 'esti'])

    plt.rcParams['font.size'] = 15
    plt.figure(figsize=(5, 5))
    plt.scatter(raw_y, estimated_y_array)
    y_max = np.max(np.array([raw_y_array, estimated_y_array]))
    y_min = np.min(np.array([raw_y_array, estimated_y_array]))
    plt.plot(
            [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
            [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 
            'k-'
            )
    plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
    plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
    plt.xlabel('actual y')
    plt.ylabel('estimated y')
    plt.title('{0} {1}  R2={2}'.format(col, data_type, round(metrics.r2_score(raw_y, estimated_y_array), 3)))
    plt.savefig(
                save_dir/'image\\{}_{}_{}.png'.format(col, data_type, model_names[i]), 
                bbox_inches='tight'
                )
    plt.show()

    # r2dcv, RMSEdcv, MAEdcv
    r2 = round(metrics.r2_score(raw_y, estimated_y_array), 5)
    mae = round(metrics.mean_absolute_error(raw_y, estimated_y_array), 5)
    rmse = round(np.sqrt(metrics.mean_squared_error(raw_y, estimated_y_array)), 5)

    evaluation_index = [r2, mae, rmse]
    evaluation_index = pd.DataFrame(
        np.array(evaluation_index).T,
        index=['r2', 'MAE', 'RMSE'],
        columns=['{}'.format(model_names[i])]
        ).T
    print('{} score: \n'.format(model_names[i]), evaluation_index)

    return evaluation_index

# objective_def = [objective_pls, objective_ridge, objective_lasso, objective_en, objective_svr_lin, objective_svr_rbf, objective_rf, objective_gp, objective_lgb]
# optimised_def = [optimised_pls, optimised_ridge, optimised_lasso, optimised_en, optimised_svr_lin, optimised_svr_rbf, optimised_rf, optimised_gp, optimised_lgb]
objective_def = [objective_svr_rbf]
optimised_def = [optimised_svr_rbf]

estimated_train_alls = {}
estimated_test_alls = {}
for col in aim_y:
    print(f'calculating for {col}')

    score_metrics_train = score_metrics.copy()
    score_metrics_test = score_metrics.copy()

    y_data = struct_data.loc[:, col]

    x_train_a, x_test_a, y_train, y_test = train_test_split(dataset_x, y_data, train_size=0.8, random_state=5, shuffle=True)

    drop_col = x_train_a.std(axis=0, ddof=1)!=0
    x_train = x_train_a.loc[:, drop_col] 
    x_test = x_test_a.loc[:, drop_col] 

    autoscaled_x_train = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0, ddof=1)
    autoscaled_y_train = (y_train - y_train.mean()) / y_train.std(ddof=1)
    autoscaled_x_test = (x_test - x_train.mean(axis=0)) / x_train.std(axis=0, ddof=1)

    for i, (obj_model, opt_model) in enumerate(zip(objective_def, optimised_def)):
        print(model_names[i])
        if obj_model != objective_lgb:
            # optuna study
            study = optuna.create_study(direction='maximize')
            study.optimize(obj_model, n_trials=100)

            # optuna optimise
            optimised_model = opt_model(study)

            # predict
            optimised_model.fit(autoscaled_x_train, autoscaled_y_train)

            estimated_y_train = optimised_model.predict(autoscaled_x_train)
            estimated_y_test = optimised_model.predict(autoscaled_x_test)
        else:
            best_params = obj_model(x_train, y_train)
            estimated_y_train, estimated_y_test = opt_model(autoscaled_x_train, autoscaled_y_train, autoscaled_x_test, y_test, best_params)

        estimated_y_train = estimated_y_train * y_train.std(axis=0, ddof=1) + y_train.mean(axis=0)
        estimated_train_alls[col+model_names[i]] = estimated_y_train

        estimated_y_test = estimated_y_test * y_train.std(axis=0, ddof=1) + y_train.mean(axis=0)
        estimated_test_alls[col+model_names[i]] = estimated_y_test

        score_train = evaluation(
            y_train,
            estimated_y_train,
            data_type='train',
            savefig=True
        )
        score_metrics_train.loc[model_names[i], :] = score_train.values

        score_test = evaluation(
            y_test,
            estimated_y_test,
            data_type='test',
            savefig=True
        )
        score_metrics_test.loc[model_names[i], :] = score_test.values
    score_metrics_train.to_csv(save_dir/'predict_scores_train_{}.csv'.format(col))
    score_metrics_test.to_csv(save_dir/'predict_scores_test_{}.csv'.format(col))

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

with open(save_dir/'estimated_y_train.json', 'w') as f:
    json.dump(estimated_train_alls, f, cls=NumpyArrayEncoder)
with open(save_dir/'estimated_y_test.json', 'w') as f:
    json.dump(estimated_test_alls, f, cls=NumpyArrayEncoder)

y_test.to_csv(save_dir/'sample_names.csv')

# feature_importance = pd.DataFrame(index=dataset_x.drop(columns=['organic_Ipc']).columns, columns=aim_y)
# for i in aim_y:
#     model_feature = RandomForestRegressor()
#     model_feature.fit(dataset_x.drop(columns=['organic_Ipc']), struct_data.loc[:, i])
#     feature_importance[i] = model_feature.feature_importances_
# feature_importance.to_csv(save_dir/'feature_importance.csv')