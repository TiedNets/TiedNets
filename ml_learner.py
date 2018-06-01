import os
import csv
import json
import random
import logging.config
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn import tree
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.externals import joblib
from matplotlib.mlab import griddata

__author__ = 'Agostino Sturaro'

# global variable
logger = None


def setup_logging(log_conf_path):
    global logger
    with open(log_conf_path, 'rt') as f:
        config = json.load(f)
    logging.config.dictConfig(config)
    logger = logging.getLogger(__name__)


# cost function is used in some plots to better visualize the prediction error and its standard deviation
def calc_my_cost(X, y, predictor):
    predictions = predictor(X)
    errors = predictions - y
    cost = np.mean(np.abs(errors))  # average of absolute error
    error_std_dev = np.std(errors, 0, ddof=1)

    return cost, error_std_dev


# normal equations used to solve the multivariate linear regression problem finding an optimal solution
def normal_equation(X, y):
    # using Octave's notation, this would be pinv(X'*X)*X'*y
    return np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)


def setup_2d_axes(xlabel, ylabel, xlim=None, ylim=None, xticks=None, yticks=None):
    fig, ax = plt.subplots()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)

    return fig, ax


def setup_3d_axes(xlabel, ylabel, zlabel, xlim=None, ylim=None, zlim=None):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if zlim is not None:
        ax.set_zlim(zlim)

    return fig, ax


# lines is a list of dictionaries [{x, y, line_style, line_label}, {...}, ...]
def plot_2d_lines(lines, ax):
    for line in lines:
        plt.plot(line['x'], line['y'], line['style'], label=line['label'])

    ax.grid(linestyle='-', linewidth=0.5)
    ax.legend()
    plt.tight_layout()  # make sure everything is showing
    plt.show()


def plot_scenario_performances(x_values, results, predictions, xlabel, ylabel):
    fig, ax = setup_2d_axes(xlabel, ylabel)

    ax.plot(x_values, results, 'g-o', label='results')
    ax.plot(x_values, predictions, 'r-o', label='predictions')
    ax.legend()

    plt.tight_layout()  # make sure everything is showing
    plt.show()


# get two 2d ndarrays, the second dimension of the arrays is like a line-wrap to represent the grid rows
# the first point of the grid is (x_grid[0], y_grid[0])
# the second one (same row, one step to the right) is (x_grid[1], y_grid[0])
def make_uniform_grid_xy(x_vec, y_vec, res_X, res_Y):
    linspace_col_1 = np.linspace(x_vec.min(), x_vec.max(), num=res_X, endpoint=False)
    linspace_col_2 = np.linspace(y_vec.min(), y_vec.max(), num=res_Y, endpoint=False)
    x_grid, y_grid = np.meshgrid(linspace_col_1, linspace_col_2)

    return x_grid, y_grid


# Takes 3 vectors and interpolates them to return a uniformly spaced 3d grid, represented using three 2d ndarrays
def make_uniform_grid_xyz(x_vec, y_vec, z_vec, res_X, res_Y):
    xs = np.linspace(np.min(x_vec), np.max(x_vec), num=res_X, endpoint=False)
    ys = np.linspace(np.min(y_vec), np.max(y_vec), num=res_Y, endpoint=False)
    x_grid, y_grid = np.meshgrid(xs, ys)

    # use the matplotlib function, not numpy's
    z_grid = griddata(x_vec, y_vec, z_vec, x_grid, y_grid, interp='linear')

    return x_grid, y_grid, z_grid


def plot_3d_no_interpolate(ax_x_vec, ax_y_vec, ax_z_vec, ax_x_label, ax_y_label, ax_z_label, scatter=True,
                           project=False, surface=False):
    if len(ax_x_vec.shape) != 1 or len(ax_y_vec.shape) != 1:
        raise ValueError('The parameters "xs" and "ys" must be 1D ndarrays, '
                         'xs.shape={}, ys.shape={}'.format(ax_x_vec.shape, ax_y_vec.shape))

    min_x, max_x = np.min(ax_x_vec), np.max(ax_x_vec)
    min_y, max_y = np.min(ax_y_vec), np.max(ax_y_vec)
    min_z, max_z = np.min(ax_z_vec), np.max(ax_z_vec)

    fig, ax = setup_3d_axes(ax_x_label, ax_y_label, ax_z_label,
                            xlim=(min_x, max_x), ylim=(min_y, max_y), zlim=(min_z, max_z))

    if surface is True:
        surf = ax.plot_trisurf(ax_x_vec, ax_y_vec, ax_z_vec, cmap=cm.jet, linewidth=0, alpha=0.8)
        plt.colorbar(surf)

    if scatter is True:
        ax.scatter(ax_x_vec, ax_y_vec, ax_z_vec)

    if project is True:
        ax.plot(ax_y_vec, ax_z_vec, 'g+', zdir='x', zs=min_x)
        ax.plot(ax_x_vec, ax_z_vec, 'r+', zdir='y', zs=max_y)
        ax.plot(ax_x_vec, ax_y_vec, 'k+', zdir='z', zs=min_z)

    plt.tight_layout()
    plt.show()


def plot_3d_lots(ax_x_label, ax_y_label, ax_z_label, ax_x_vec, ax_y_vec, ax_z_vec,
                 x_grid, y_grid, z_grid, surface=False, contour=True, scatter=True, project=True):
    if len(ax_x_vec.shape) != 1 or len(ax_y_vec.shape) != 1:
        raise ValueError('The parameters "xs" and "ys" must be 1D ndarrays, '
                         'xs.shape={}, ys.shape={}'.format(ax_x_vec.shape, ax_y_vec.shape))

    min_x, max_x = np.min(ax_x_vec), np.max(ax_x_vec)
    min_y, max_y = np.min(ax_y_vec), np.max(ax_y_vec)
    min_z, max_z = np.min(ax_z_vec), np.max(ax_z_vec)

    fig, ax = setup_3d_axes(ax_x_label, ax_y_label, ax_z_label,
                            xlim=(min_x, max_x), ylim=(min_y, max_y), zlim=(min_z, max_z))

    if surface is True:
        ax.plot_surface(x_grid, y_grid, z_grid, cmap=cm.jet, linewidth=0)

    if contour is True:
        ax.contour(x_grid, y_grid, z_grid, extend3d=True, cmap=cm.coolwarm)
        ax.contour(x_grid, y_grid, z_grid, zdir='x', offset=min_x, cmap=cm.jet)
        ax.contour(x_grid, y_grid, z_grid, zdir='y', offset=max_y, cmap=cm.jet)
        ax.contour(x_grid, y_grid, z_grid, zdir='z', offset=min_z, cmap=cm.jet)

    if scatter is True:
        ax.scatter(ax_x_vec, ax_y_vec, ax_z_vec)

    if project is True:
        ax.plot(ax_y_vec, ax_z_vec, 'g+', zdir='x', zs=min_x)
        ax.plot(ax_x_vec, ax_z_vec, 'r+', zdir='y', zs=max_y)
        ax.plot(ax_x_vec, ax_y_vec, 'k+', zdir='z', zs=min_z)

    plt.tight_layout()
    plt.show()


# create a mask array to filter the rows of an array having the desired values on a set of columns
# values_by_col is a dictionary of the form {col_number: desired_value, another_col_number: another_value}
def create_mask_for_rows(array, values_by_col):
    if not isinstance(values_by_col, dict) or len(values_by_col) == 0:
        raise ValueError('The parameter "values_by_col" should be a non-empty dictionary')

    cur_mask = None
    for col_num in values_by_col:
        col_val = values_by_col[col_num]
        mask = array[:, col_num] == col_val
        if cur_mask is not None:
            cur_mask &= mask
        else:
            cur_mask = mask

    return cur_mask


# by scenario we mean a series of simulations representing cumulative attacks on the same network
# each simulations of a scenario hits the same node as the simulation before, plus a new one, e.g.:
# (A1), (A1, A2), (A1, A2, A3), ...
# we can identify a scenario by filtering rows having the same (seed, instance)
# the number of attacks is considered the independent variable (the only one we change) in the scenario
# the parameter "atks_cnt_col" is the number of the column within the "data_info" array that holds the number of attacks
# for each simulation, we want to compare the result we got with the prediction generated by our learning algorithm
def find_scenario_results_and_predictions(data_X, results_y, data_info, info_filter, atks_cnt_col, predictor):
    # filter simulations (rows) relevant to our scenario
    mask = create_mask_for_rows(data_info, info_filter)
    subset_X = data_X[mask]
    subset_y = results_y[mask]
    subset_info = data_info[mask]
    if subset_X.shape[0] == 0:
        raise ValueError('No simulation in the dataset passed by parameter "data_X" matches the conditions specified '
                         'by the parameter "info_filter"\n{}'.format(info_filter))

    # we are under the assumption that each scenario only has a simulation for each number of attacks
    atk_counts = subset_info[:, atks_cnt_col]
    if atk_counts.shape != (np.unique(atk_counts)).shape:
        raise RuntimeError('This function assumes that, for a given scenario, only one simulation was done for each '
                           'quantity of attacks. Check the value of the parameter "atks_cnt_col" and your dataset.')

    # these 3 vectors should be treated as a single matrix and never sorted separately
    sorted_atk_counts = np.zeros(atk_counts.shape, dtype=atk_counts.dtype)
    results = np.zeros(atk_counts.shape)
    predictions = np.zeros(atk_counts.shape)

    # for each simulation (row), save the actual result and compute a prediction
    sort_index = np.argsort(atk_counts)  # use this to iterate the array picking the elements in the correct order
    for i, data_row_num in enumerate(sort_index):
        sorted_atk_counts[i] = atk_counts[data_row_num]
        results[i] = subset_y[data_row_num]  # actual result of the simulation
        predictions[i] = predictor(subset_X[data_row_num, None, :])  # predicted result of the simulation

    return sorted_atk_counts, results, predictions


# read a text file (e.g. tab separated values) and only load columns with the given names
def load_named_cols(input_fpath, col_names, file_header):
    col_nums = [file_header.index(col_name) for col_name in col_names]
    data_array = np.loadtxt(input_fpath, delimiter='\t', skiprows=1, usecols=col_nums)
    return data_array


def apply_row_filter(X, y, info, X_col_names, y_col_name, info_col_names, filter_conf):
    col_name = filter_conf['col_name']
    filter_values = filter_conf['col_values']

    if col_name in X_col_names:
        col_names = X_col_names
        data = X
    elif col_name == y_col_name:
        data = y
    elif col_name in info_col_names:
        col_names = info_col_names
        data = info
    else:
        raise ValueError('The column "col_name" of "filter" must be one of the columns in: '
                         '"X_col_names", "y_col_name", "info_col_names".')

    relevant_idx = np.zeros(data.shape[0], dtype=bool)
    if data.ndim == 1:
        for i in range(0, data.shape[0]):
            relevant_idx[i] = data[i] in filter_values
    else:
        col_num = col_names.index(col_name)
        for i in range(0, data.shape[0]):
            relevant_idx[i] = data[i, col_num] in filter_values

    if True not in relevant_idx:
        raise ValueError('None of the values specified in "col_values" is present in column "{}"'.format(col_name))

    return X[relevant_idx], y[relevant_idx], info[relevant_idx]


# possible improvement, accept dtypes and use genfromtxt
def load_dataset(dataset_fpath, X_col_names, y_col_name, info_col_names, filter_conf=None):
    global logger

    # from each file we load load 3 sets of data, the examples (X), the labels (y) and related information (info)
    with open(dataset_fpath, 'r') as data_file:
        csvreader = csv.reader(data_file, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        header = csvreader.next()

    result_col = header.index(y_col_name)
    X = load_named_cols(dataset_fpath, X_col_names, header)
    y = np.loadtxt(dataset_fpath, delimiter='\t', skiprows=1, usecols=[result_col])
    info = load_named_cols(dataset_fpath, info_col_names, header)

    logger.debug('Loading dataset at {}'.format(dataset_fpath))
    logger.debug('X (first 2 rows)\n{}'.format(X[range(2), :]))
    logger.debug('y (first 2 rows)\n{}'.format(y[range(2)]))
    logger.debug('info (first 2 rows)\n{}'.format(info[range(2), :]))

    if filter_conf is None:
        return X, y, info
    else:
        return apply_row_filter(X, y, info, X_col_names, y_col_name, info_col_names, filter_conf)


# Returns costs and std dev of costs for each group of examples it finds
# group_by_col is a column of values with the same number of rows as data_X,
# usually it is one of the info columns from the same dataset as data_X
# For each unique value found on the column "group_by_col", a separate cost value is returned.
# To calculate each cost value, we use the examples that have the same value on the column "group_by_col".
def calc_cost_group_by(data_X, data_y, group_by_col, predictor):
    unique_vals = np.sort(np.unique(group_by_col))
    var_costs = np.zeros(unique_vals.size)
    error_std_devs = np.zeros(unique_vals.size)
    for i, val in enumerate(unique_vals):
        relevant_idx = group_by_col == val
        relevant_data_X = data_X[relevant_idx, :]
        relevant_data_y = data_y[relevant_idx]
        var_costs[i], error_std_devs[i] = calc_my_cost(relevant_data_X, relevant_data_y, predictor)
    return unique_vals, var_costs, error_std_devs


# Returns accuracies and F1 scores for each group of examples it finds
# group_by_col is a column of values with the same number of rows as data_X,
# usually it is one of the info columns from the same dataset as data_X
# To calculate each cost value, we use the examples that have the same value on the column "group_by_col".
def calc_scores_group_by(data_X, data_y, group_by_col, predictor):
    unique_vals = np.sort(np.unique(group_by_col))
    accuracies = np.zeros(unique_vals.size)
    f1_scores = np.zeros(unique_vals.size)
    for i, val in enumerate(unique_vals):
        relevant_idx = group_by_col == val
        relevant_data_X = data_X[relevant_idx, :]
        relevant_data_y = data_y[relevant_idx]
        predictions = predictor(relevant_data_X)
        accuracies[i] = accuracy_score(relevant_data_y, predictions)
        # change if this is not a binary classification
        f1_scores[i] = f1_score(relevant_data_y, predictions, pos_label='low', average='binary')
    return unique_vals, accuracies, f1_scores


# returns three arrays:
# 1) unique_vals, the unique values found on the group_by_col, from lowest to highest
#    e.g. the list of attack sizes
# 2) avg_labels, the average labels of the examples of each group
#    e.g. for each attack size, the average fraction of dead nodes
# 3) avg_preds, the average predicted values of each group
#    e.g. for each attack size, the average predicted fraction of dead nodes
# The returned arrays should never be sorted separately, because
# unique_vals[i], avg_labels[i] and avg_preds[i] work together
def avg_labels_and_preds_group_by(data_X, data_y, group_by_col, predictor):
    # pick unique values on the column with the number of attacks
    unique_vals = np.sort(np.unique(group_by_col))
    avg_labels = np.zeros(unique_vals.size)
    if predictor is not None:
        avg_preds = np.zeros(unique_vals.size)
    else:
        avg_preds = None
    for i, val in enumerate(unique_vals):
        relevant_idx = group_by_col == val
        relevant_data_X = data_X[relevant_idx, :]
        relevant_data_y = data_y[relevant_idx]
        avg_labels[i] = np.mean(relevant_data_y)
        if predictor is not None:
            avg_preds[i] = np.mean(predictor(relevant_data_X))
    return unique_vals, avg_labels, avg_preds


# iteratively apply SelectFromModel.transform changing the threshold until we get the desired number of features
# If you have a test set, just use the returned fitted_sfm to transform it
def iterate_sfm_transform(fitted_sfm, unfitted_X, max_feature_cnt, max_rounds, base_thresh, thresh_incr):
    temp_train_X = fitted_sfm.transform(unfitted_X)
    sel_feature_cnt = temp_train_X.shape[1]

    if sel_feature_cnt > max_feature_cnt:
        rounds = 0
        fitted_sfm.threshold = base_thresh
        temp_train_X = fitted_sfm.transform(unfitted_X)
        sel_feature_cnt = temp_train_X.shape[1]
        while sel_feature_cnt > max_feature_cnt and rounds < max_rounds:
            fitted_sfm.threshold += thresh_incr
            temp_train_X = fitted_sfm.transform(unfitted_X)
            sel_feature_cnt = temp_train_X.shape[1]
            rounds += 1
    fitted_X = temp_train_X

    return fitted_X, fitted_sfm


# We pick an estimator (e.g. linear regression) and put it aside.
# We apply some preprocessing on the dataset (e.g. standardization), without using the estimator.
# Then, if we want to do feature selection, we pick a selector (e.g. Recursive Feature Elimination).
# The selector uses the estimator and gets fit with the dataset,
# then we use the selector to transform the dataset, performing the actual feature selection.
# Finally, we fit (train) the estimator (model) using the prepared dataset (preprocessed and feature-selected).
# This model can be used to make predictions immediately on the transformed data used to train it,
# or to make predictions on other datasets (e.g. the test set), provided we apply the same transformations first.
def train_regr_model(train_X, train_y, X_col_names, model_conf):
    global logger
    # make a local copy of the objects we received as parameters and might change
    train_X = train_X.copy()  # make a local copy of the array
    X_col_names = list(X_col_names)  # make a local copy of the list

    model_name = model_conf['model']['name'].lower()
    model_kwargs = model_conf['model']['kwargs']
    logger.info('Model name: {}'.format(model_name))

    # LinearRegression works as polynomial regression if we apply PolynomialFeatures to the dataset.
    # RidgeCV, LassoCV and ElasticNetCV are linear regression models with different regularizations.
    # Regularization is used to lower the weight given to less important and useless features.
    # They have built-in cross validation (cv) that is used to tune the hyperparameters, like alpha.
    if model_name == 'linearregression':
        clf = linear_model.LinearRegression(**model_kwargs)
    elif model_name == 'ridgecv':
        clf = linear_model.RidgeCV(**model_kwargs)
    elif model_name == 'lassocv':
        clf = linear_model.LassoCV(**model_kwargs)
    elif model_name == 'elasticnetcv':
        clf = linear_model.ElasticNetCV(**model_kwargs)
    elif model_name == 'decisiontreeregressor':
        clf = tree.DecisionTreeRegressor(**model_kwargs)
    elif model_name == 'mlpregressor':
        clf = MLPRegressor(**model_kwargs)
    else:
        raise ValueError('Unsupported model name: "{}"'.format(model_name))

    # objects for preprocessing and feature selection, in the order they were used
    transformers = []

    # used to log more readable standardization info
    scaling_by_col = {}
    scaled = False

    steps = model_conf['steps']
    for step_num, step in enumerate(steps):
        step_name = step['name'].lower()
        step_kwargs = step['kwargs']
        selector = None
        logger.info('Step {}: {}'.format(step_num, step_name))

        if step_name == 'variancethreshold':
            base_feature_cnt = train_X.shape[1]
            vt_sel = VarianceThreshold(**step_kwargs)
            train_X = vt_sel.fit_transform(train_X)
            vt_feature_mask = vt_sel.get_support()
            X_col_names = [item for item_num, item in enumerate(X_col_names) if vt_feature_mask[item_num]]
            sel_feature_cnt = train_X.shape[1]
            logger.debug('VarianceThreshold removed {} features'.format(base_feature_cnt - sel_feature_cnt))
            transformers.append(vt_sel)

        # create polynomial features, interactions allowing us to learn a more complex prediction function
        elif step_name == 'polynomialfeatures':
            poly = preprocessing.PolynomialFeatures(**step_kwargs)
            train_X = poly.fit_transform(train_X)
            X_col_names = poly.get_feature_names(X_col_names)
            logger.debug('Polynomial X_col_names = {}'.format(X_col_names))
            logger.debug('train_X with polynomial features (first 2 rows)\n{}'.format(train_X[range(2), :]))
            transformers.append(poly)

        # apply a standardization step
        elif step_name == 'standardscaler':
            scaler = preprocessing.StandardScaler(**step_kwargs)
            train_X = scaler.fit_transform(train_X)
            if scaled is True:
                logger.warning('You are have more than one standardization step!')
            scaled = True
            means = scaler.mean_
            stddevs = scaler.scale_
            for col_num, col_name in enumerate(X_col_names):
                scaling_by_col[col_name] = {'mean': means[col_num], 'std': stddevs[col_num]}
            transformers.append(scaler)

        elif step_name == 'rfe':
            selector = RFE(clf, **step_kwargs)
        elif step_name == 'rfecv':
            selector = RFECV(clf, **step_kwargs)
        elif step_name == 'selectfrommodel':
            selector = SelectFromModel(clf, **step_kwargs)
        else:
            raise ValueError('Unsupported step name: "{}"'.format(step_name))

        if selector is not None:
            selector.fit(train_X, train_y)

            if step_name == 'selectfrommodel':
                train_X, selector = iterate_sfm_transform(selector, train_X, 20, 100, 0.01, 0.01)
            else:
                train_X = selector.transform(train_X)

            sel_feature_mask = selector.get_support()
            X_col_names = [item for item_num, item in enumerate(X_col_names) if sel_feature_mask[item_num]]
            logger.info('Selected features = {}'.format(X_col_names))
            logger.debug('After transform, train_X.shape[1] = {}'.format(train_X.shape[1]))
            transformers.append(selector)

    clf.fit(train_X, train_y)
    if model_name in ['linearregression', 'ridgecv', 'lassocv', 'elasticnetcv']:
        learned_eq = '{:+.3f}'.format(clf.intercept_)
        coefficients = clf.coef_
        for i in range(0, len(coefficients)):
            learned_eq += ' {:+.3f} {}'.format(coefficients[i], X_col_names[i])
        logger.info('Learned equation = {}'.format(learned_eq))
    if model_name in ['ridgecv', 'lassocv', 'elasticnetcv']:
        logger.info('alpha = {}'.format(clf.alpha_))

    if scaled is True:
        scaling_by_col = {col_name: scaling_by_col[col_name] for col_name in X_col_names}
        logger.info('Scaling of selected features: {}'.format(scaling_by_col))

    logger.debug('Learning completed')

    return clf, transformers, train_X, X_col_names


def check_prediction_bounds(plain_X, info, X_col_names, info_col_names, predictions, lb, include_lb, ub, include_ub,
                            print_examples=False):
    if plain_X.shape[1] != len(X_col_names):
        raise ValueError('plain_X and X_col_names should have the same number of columns')
    if info.shape[1] != len(info_col_names):
        raise ValueError('info and info_col_names should have the same number of columns')

    if include_lb is True:
        below_mask = np.less(predictions, lb)  # only strictly below the bound
    else:
        below_mask = np.less_equal(predictions, lb)
    if print_examples and True in below_mask:
        logger.info('Prediction below lower bound for the following {} examples'.format(np.count_nonzero(below_mask)))
        logger.info('{}\n{}'.format(X_col_names, plain_X[below_mask]))
        logger.info('{}\n{}'.format(info_col_names, info[below_mask]))

    if include_ub is True:
        over_mask = np.greater(predictions, ub)
    else:
        over_mask = np.greater_equal(predictions, ub)
    if print_examples and True in over_mask:
        logger.info('Prediction over upper bound for the following {} examples'.format(np.count_nonzero(over_mask)))
        logger.info('{}\n{}'.format(X_col_names, plain_X[over_mask]))
        logger.info('{}\n{}'.format(info_col_names, info[over_mask]))


def interpolate_xy_predict_z_plot_3d(X, y, transformers, predictor, ax_x_label, ax_y_label, ax_z_label, res_x, res_y):
    global logger
    if X.shape[1] != 2:
        raise ValueError('This function only works for datasets with exactly 2 features.')

    # create a dataset with uniformly spaced points
    x_grid, y_grid = make_uniform_grid_xy(X[:, 0], X[:, 1], res_x, res_y)

    # this is why we need X to only have 2 features, we need to recreate it
    uniform_X = np.c_[x_grid.ravel(), y_grid.ravel()]

    for transformer in transformers:
        logger.debug('Applying {}.transform'.format(type(transformer).__name__))
        uniform_X = transformer.transform(uniform_X)

    predictions = predictor(uniform_X)
    z_grid = np.reshape(predictions, x_grid.shape)

    ax_x_vec, ax_y_vec, ax_z_vec = X[:, 0], X[:, 1], y
    plot_3d_lots(ax_x_label, ax_y_label, ax_z_label, ax_x_vec, ax_y_vec, ax_z_vec, x_grid, y_grid, z_grid)


def plot_rnd_scenarios(X, y, info, info_col_names, predictor, xlabel, ylabel, rnd_inst_cnt, rnd_seed_cnt, seed=None):
    global logger

    # TODO: rework these hardcoded names
    atks_cnt_col = info_col_names.index('#atkd_a')
    instances_col = info_col_names.index('instance')
    seeds_col = info_col_names.index('seed')

    instances = np.sort(np.unique(info[:, instances_col]))
    sim_seeds = np.sort(np.unique(info[:, seeds_col]))
    logger.debug('All instances = {}\nAll sim seeds {}'.format(instances, sim_seeds))

    my_random = random.Random(seed)
    my_random.shuffle(instances)
    my_random.shuffle(sim_seeds)

    for cur_inst in instances[0:rnd_inst_cnt]:
        logger.info('instance = {}'.format(cur_inst))
        for cur_seed in sim_seeds[0:rnd_seed_cnt]:
            logger.info('seed = {}'.format(cur_seed))
            info_filter = {instances_col: cur_inst, seeds_col: cur_seed}
            indep_var_vals, results, predictions = \
                find_scenario_results_and_predictions(X, y, info, info_filter, atks_cnt_col, predictor)
            plot_scenario_performances(indep_var_vals, results, predictions, xlabel, ylabel)


# draw a graph showing the mean error for each #attacks, and the standard deviation of this error
def plot_cost_by_atk_size(atk_sizes, costs, std_devs):
    fig, ax = setup_2d_axes('# of attacked power nodes', 'Measured fraction', ylim=(0, 0.5))
    line_1 = {'x': atk_sizes, 'y': costs, 'style': 'b-o', 'label': 'Avg abs prediction error'}
    line_2 = {'x': atk_sizes, 'y': std_devs, 'style': 'r-o', 'label': 'Standard deviation'}
    plot_2d_lines([line_1, line_2], ax)


def plot_deaths_and_preds_by_atk_size(atk_sizes, avg_deaths, avg_preds):
    line_1 = {'x': atk_sizes, 'y': avg_deaths, 'style': 'g-o', 'label': 'Actual'}
    line_2 = {'x': atk_sizes, 'y': avg_preds, 'style': 'b-o', 'label': 'Predicted'}
    fig, ax = setup_2d_axes('# of attacked power nodes', 'Average fraction of dead nodes')
    plot_2d_lines([line_1, line_2], ax)


def plot_actual_and_pred_cnts_by_atk_size(atk_sizes, actual_cnt, pred_cnt):
    ind = np.arange(len(atk_sizes))  # the x locations for the groups
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, actual_cnt, width, color='r')
    rects2 = ax.bar(ind + width, pred_cnt, width, color='b')

    ax.set_xlabel('#attacked nodes')
    ax.set_ylabel('#simulations')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(atk_sizes)

    ax.set_title('Occurrences of massive destruction ordered by attack size')
    ax.legend((rects1[0], rects2[0]), ('Actual', 'Predicted'), fontsize=10)
    plt.tight_layout()
    plt.show()


def train_model_on_dataset(config, model_num):
    model_conf = config['model_trainings'][model_num]
    dataset_num = model_conf['dataset_num']
    model_name = model_conf['model']['name'].lower()

    dataset = config['datasets'][dataset_num]
    dataset_fpath = dataset['fpath']
    X_col_names = dataset['X_col_names']
    y_col_name = dataset['y_col_name']
    info_col_names = dataset['info_col_names']
    filter_conf = dataset['filter'] if 'filter' in dataset else None

    # find columns
    train_X, train_y, train_info = load_dataset(dataset_fpath, X_col_names, y_col_name, info_col_names, filter_conf)

    model, transformers, train_X, transf_X_col_names = \
        train_regr_model(train_X, train_y, X_col_names, model_conf)

    if 'output_dir' in model_conf:
        output_dir = os.path.normpath(model_conf['output_dir'])
        if os.path.isabs(output_dir) is False:
            output_dir = os.path.abspath(output_dir)

        # save the learned model and what is needed to adapt the data to it
        learned_stuff_fpath = os.path.join(output_dir, 'model_{}.pkl'.format(model_num))
        learned_stuff = {'model': model, 'transformers': transformers}
        joblib.dump(learned_stuff, learned_stuff_fpath)
        logger.info('Saved model to {}'.format(learned_stuff_fpath))

        # save a representation of the learned decision tree, to draw it, install graphviz and run
        # dot -T png decision_tree.dot -o decision_tree.png
        if 'tree' in model_name:
            tree_repr_fpath = os.path.join(output_dir, 'model_{}_tree.dot'.format(model_num))
            with open(tree_repr_fpath, 'w') as out_file:
                tree.export_graphviz(model, out_file, feature_names=transf_X_col_names)
            logger.info('Saved tree representation to {}'.format(tree_repr_fpath))

    return model, transformers, transf_X_col_names


def pick_group_by_col(plain_ds_X, X_col_names, ds_info, info_col_names, config):
    group_by_col_name = config['group_by_col_name']
    if group_by_col_name in X_col_names:
        group_by_col = plain_ds_X[:, X_col_names.index(group_by_col_name)]
    elif group_by_col_name in info_col_names:
        group_by_col = ds_info[:, info_col_names.index(group_by_col_name)]
    else:
        raise ValueError('Parameter "group_by_col_name" must be one of the columns specified by parameter '
                         '"X_col_names" or by parameter "info_col_names".')
    return group_by_col


# Type of plots by name:
# - features_xy_results_z, 3D plot, represents two features of a dataset, one on X and one on Y, and the corresponding
# simulation results on Z. Useful to check how any two features affect simulation results.
# - features_xy_predictions_z, 3D plot, same as features_xy_results_z, but with predicted results on Z. Useful to check
# how any two features affect predicted results.
# - interpolate_xy_predict_z, 3D plot, similar to features_xy_predictions_z, but only works for predictions that need
# exactly two features. It interpolates points on X and Y to get a smoother drawing.
# - plot_rnd_scenarios, 2D plot, picks a few random scenarios from a dataset, and for each one, it plots two lines,
# one with the simulation results, and the other with the predicted results. Useful to test different predictors and
# find out how they perform on individual scenarios, rather than evaluating performances on a whole dataset.
# - cost_by_atk_size, 2D plot, draws a single line for a given dataset. Values on the X axis are the initial attack
# sizes, while values on Y are measures of the prediction error calculated by a cost function. The line has error bars
# representing the standard deviation of the error. Useful to visualize prediction performances on a whole dataset.
# - cost_by_atk_size_many, 2D plot, same as cost_by_atk_size, but also works for multiple datasets, drawing a different
# line for each dataset.
# - deaths_and_preds_by_atk_size, 2D plot, plots two lines, one representing the average simulation results and one
# representing the predicted results for different initial attack sizes. Values on the X axis represent the fraction of
# attacked nodes, while values on Y represent the resulting fraction of failed nodes.
# - deaths_and_preds_by_atk_size_many, 2D plot, similar to deaths_and_preds_by_atk_size_many, but also works for
# multiple datasets, drawing a different pair of lines for each dataset. Additionally, it can plot only the simulation
# results for a dataset. Just omit the model_num option for that dataset.
def make_plots(config, models):

    plots = config['plots']
    for plot_conf in plots:
        plot_name = plot_conf['name']

        if 'dataset_num' in plot_conf:
            dataset_num = plot_conf['dataset_num']
            dataset = config['datasets'][dataset_num]
            dataset_fpath = dataset['fpath']
            X_col_names = dataset['X_col_names']
            y_col_name = dataset['y_col_name']
            info_col_names = dataset['info_col_names']
            filter_conf = dataset['filter'] if 'filter' in dataset else None

            plain_ds_X, ds_y, ds_info =\
                load_dataset(dataset_fpath, X_col_names, y_col_name, info_col_names, filter_conf)

            if 'model_num' in plot_conf:
                model_num = plot_conf['model_num']
                model = models[model_num]['model']
                transformers = models[model_num]['transformers']
                ds_X = plain_ds_X.copy()
                for transformer in transformers:
                    logger.debug('Applying {}.transform'.format(type(transformer).__name__))
                    ds_X = transformer.transform(ds_X)
                predictor = lambda x: model.predict(x)
            else:
                model, transformers, ds_X, predictor = None, None, None, None
                if plot_name != 'features_xy_results_z':
                    raise ValueError('Plot {} needs a "model_num" configuration parameter'.format(plot_name))

        # visualize how two features, on axis x and y, affect the results and the predictions on axis z
        if plot_name in ['features_xy_results_z', 'features_xy_predictions_z']:
            ax_x_vec = plain_ds_X[:, X_col_names.index(plot_conf['ax_x_feature'])]
            ax_y_vec = plain_ds_X[:, X_col_names.index(plot_conf['ax_y_feature'])]
            if plot_name == 'features_xy_results_z':
                ax_z_vec = ds_y
            else:
                ax_z_vec = predictor(ds_X)
            ax_x_label = plot_conf['ax_x_label']
            ax_y_label = plot_conf['ax_y_label']
            ax_z_label = plot_conf['ax_z_label']
            plot_3d_no_interpolate(ax_x_vec, ax_y_vec, ax_z_vec, ax_x_label, ax_y_label, ax_z_label,
                                   plot_conf['scatter'], plot_conf['project'], plot_conf['surface'])

        # make some plots comparing results and predictions on different scenarios
        if plot_name == 'plot_rnd_scenarios':
            ax_x_label = plot_conf['ax_x_label']
            ax_y_label = plot_conf['ax_y_label']
            plot_rnd_scenarios(ds_X, ds_y, ds_info, info_col_names, predictor, ax_x_label, ax_y_label,
                               plot_conf['rnd_inst_cnt'], plot_conf['rnd_seed_cnt'], plot_conf.get('seed'))

        # this only works for datasets with only 2 features
        if plot_name == 'interpolate_xy_predict_z':
            ax_x_label = plot_conf['ax_x_label']
            ax_y_label = plot_conf['ax_y_label']
            ax_z_label = plot_conf['ax_z_label']
            res_x = plot_conf['res_x']
            res_y = plot_conf['res_y']
            interpolate_xy_predict_z_plot_3d(plain_ds_X, ds_y, transformers, predictor,
                                             ax_x_label, ax_y_label, ax_z_label, res_x, res_y)

        # add plot type that plots datasets together on the same figure
        if plot_name == 'cost_by_atk_size':
            group_by_col = pick_group_by_col(plain_ds_X, X_col_names, ds_info, info_col_names, plot_conf)
            atk_sizes, costs, error_stdevs = calc_cost_group_by(ds_X, ds_y, group_by_col, predictor)
            plot_cost_by_atk_size(atk_sizes, costs, error_stdevs)

        if plot_name == 'deaths_and_preds_by_atk_size':
            group_by_col = pick_group_by_col(plain_ds_X, X_col_names, ds_info, info_col_names, plot_conf)
            atk_sizes, avg_deaths, avg_preds = \
                avg_labels_and_preds_group_by(ds_X, ds_y, group_by_col, predictor)
            plot_deaths_and_preds_by_atk_size(atk_sizes, avg_deaths, avg_preds)

        if 'overlays' in plot_conf:
            overlays = plot_conf['overlays']

            ax_x_label = plot_conf['ax_x_label']
            ax_y_label = plot_conf['ax_y_label']
            fig, ax = setup_2d_axes(ax_x_label, ax_y_label)

            if 'ax_x_lim' in plot_conf:
                ax_x_lim = plot_conf['ax_x_lim']
                ax.set_xlim(**ax_x_lim)
            if 'ax_y_lim' in plot_conf:
                ax_y_lim = plot_conf['ax_y_lim']
                ax.set_ylim(**ax_y_lim)
            ax.grid(linestyle='-', linewidth=0.5)

            for overlay in overlays:
                data_label = overlay['label']
                dataset_num = overlay['dataset_num']
                dataset = config['datasets'][dataset_num]
                X_col_names = dataset['X_col_names']
                y_col_name = dataset['y_col_name']
                info_col_names = dataset['info_col_names']
                dataset_fpath = dataset['fpath']
                filter_conf = dataset['filter'] if 'filter' in dataset else None

                plain_ds_X, ds_y, ds_info =\
                    load_dataset(dataset_fpath, X_col_names, y_col_name, info_col_names, filter_conf)

                x_multiplier = 1
                if 'x_multiplier' in overlay:
                    x_multiplier = overlay['x_multiplier']
                y_multiplier = 1
                if 'y_multiplier' in overlay:
                    y_multiplier = overlay['y_multiplier']

                if 'model_num' in overlay:
                    model_num = overlay['model_num']
                    model = models[model_num]['model']
                    transformers = models[model_num]['transformers']
                    ds_X = plain_ds_X.copy()
                    for transformer in transformers:
                        logger.debug('Applying {}.transform'.format(type(transformer).__name__))
                        ds_X = transformer.transform(ds_X)
                    predictor = lambda x: model.predict(x)
                else:
                    model, transformers, ds_X, predictor = None, None, None, None

                if plot_name == 'cost_by_atk_size_many':
                    group_by_col = pick_group_by_col(plain_ds_X, X_col_names, ds_info, info_col_names, plot_conf)
                    atk_sizes, costs, error_stdevs = calc_cost_group_by(ds_X, ds_y, group_by_col, predictor)
                    color = overlay['color']
                    fmt = overlay['fmt']
                    if x_multiplier != 1 or y_multiplier != 1:
                        ax.errorbar(x_multiplier * atk_sizes, y_multiplier * costs, y_multiplier * error_stdevs,
                                    fmt=fmt, color=color, linewidth=1, capsize=3, label=data_label)
                    else:
                        ax.errorbar(atk_sizes, costs, error_stdevs, fmt=fmt, color=color, linewidth=1, capsize=3,
                                    label=data_label)

                if plot_name == 'deaths_and_preds_by_atk_size_many':
                    style = overlay['style']
                    group_by_col = pick_group_by_col(plain_ds_X, X_col_names, ds_info, info_col_names, plot_conf)

                    if 'model_num' in overlay:
                        atk_sizes, avg_deaths, avg_preds = \
                            avg_labels_and_preds_group_by(ds_X, ds_y, group_by_col, predictor)
                        if x_multiplier != 1 or y_multiplier != 1:
                            plt.plot(x_multiplier * atk_sizes, y_multiplier * avg_preds, style, label=data_label)
                        else:
                            plt.plot(atk_sizes, avg_preds, style, label=data_label)
                    else:
                        atk_sizes, avg_deaths, avg_preds = \
                            avg_labels_and_preds_group_by(plain_ds_X, ds_y, group_by_col, predictor)
                        if x_multiplier != 1 or y_multiplier != 1:
                            plt.plot(x_multiplier * atk_sizes, y_multiplier * avg_deaths, style, label=data_label)
                        else:
                            plt.plot(atk_sizes, avg_deaths, style, label=data_label)
                            # if avg_deaths.shape[0] == 1:
                            #     plt.axhline(avg_deaths[0], color='red', linestyle='--', label=data_label)
                            # else:
                            #     plt.plot(atk_sizes, avg_deaths, style, label=data_label)

            ax.legend()  # Create the plot legend considering all overlays

            if 'fig_fpath' in plot_conf:
                fig_fpath = os.path.normpath(plot_conf['fig_fpath'])
                if os.path.isabs(fig_fpath) is False:
                    fig_fpath = os.path.abspath(fig_fpath)
                fig.savefig(fig_fpath, bbox_inches="tight")
                logger.info('Figure saved to {}'.format(fig_fpath))

            plt.tight_layout()
            plt.show()


def run():
    conf_fpath = './dataset.json'
    with open(conf_fpath) as conf_file:
        config = json.load(conf_file)

    models = []
    model_trainings = config['model_trainings']
    for model_num in range(0, len(model_trainings)):
        model, transformers, transf_X_col_names = train_model_on_dataset(config, model_num)
        models.append({'model': model, 'transformers': transformers})

    make_plots(config, models)


setup_logging('logging_base_conf.json')
run()
