import os
import csv
import json
import random
import logging.config
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn import tree
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


def transform_and_predict(X, poly_feat, scaler, clf):
    transformed_X = poly_feat.transform(X)
    transformed_X = scaler.transform(transformed_X)
    return clf.predict(transformed_X)


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


# possible improvement, accept dtypes and use genfromtxt
def load_dataset(dataset_fpath, X_col_names, y_col_name, info_col_names):
    global logger

    # from each file we load load 3 sets of data, the examples (X), the labels (y) and related information
    # for each array we remember the correlation column name -> column number
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

    return X, y, info


def filter_dataset(X, y, info, X_col_names, info_col_names, filter_conf):
    filter_col_group = filter_conf['col_group']
    filter_values = filter_conf['col_values']

    if filter_col_group == 'X_col_names':
        col_names = X_col_names
        data = X
    elif filter_col_group == 'info_col_names':
        col_names = info_col_names
        data = info
    elif filter_col_group == 'y_col_name':
        data = y
    else:
        raise ValueError('The possible values for the "col_group" option of a filter are: '
                         '"X_col_names", "info_col_names", "y_col_name".')

    relevant_idx = np.zeros(data.shape[0], dtype=bool)
    if data.ndim == 1:
        for i in range(0, data.shape[0]):
            relevant_idx[i] = data[i] in filter_values
    else:
        filter_col_name = filter_conf['col_name']
        col_num = col_names.index(filter_col_name)
        for i in range(0, data.shape[0]):
            relevant_idx[i] = data[i, col_num] in filter_values

    return X[relevant_idx], y[relevant_idx], info[relevant_idx]


# Returns costs and std dev of costs for each group of examples it finds
# group_by_col is the index of a column of data_info
# For each unique value found on the column "group_by_col", a separate cost value is returned.
# To calculate each cost value, we use the examples that have the same value on the column "group_by_col".
def calc_cost_group_by(data_X, data_y, data_info, group_by_col, predictor):
    unique_vals = np.sort(np.unique(data_info[:, group_by_col]))
    var_costs = np.zeros(unique_vals.size)
    error_std_devs = np.zeros(unique_vals.size)
    # create the mask (row y/n) using the info matrix, but apply it to the data matrix
    # this way we can filter data rows using information not provided to the learning alg
    for i, val in enumerate(unique_vals):
        relevant_idx = data_info[:, group_by_col] == val
        relevant_data_X = data_X[relevant_idx, :]
        relevant_data_y = data_y[relevant_idx]
        var_costs[i], error_std_devs[i] = calc_my_cost(relevant_data_X, relevant_data_y, predictor)
    return unique_vals, var_costs, error_std_devs


# Returns accuracies and F1 scores for each group of examples it finds
# group_by_col is the index of a column of data_info
# For each unique value found on the column "group_by_col", a separate cost value is returned.
# To calculate each cost value, we use the examples that have the same value on the column "group_by_col".
def calc_scores_group_by(data_X, data_y, data_info, group_by_col, predictor):
    unique_vals = np.sort(np.unique(data_info[:, group_by_col]))
    accuracies = np.zeros(unique_vals.size)
    f1_scores = np.zeros(unique_vals.size)
    # create the mask (row y/n) using the info matrix, but apply it to the data matrix
    # this way we can filter data rows using information not provided to the learning alg
    for i, val in enumerate(unique_vals):
        relevant_idx = data_info[:, group_by_col] == val
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
# 3) avg_preds,
#    e.g. for each attack size, the average predicted fraction of dead nodes
# The returned arrays should never be sorted separately, because
# unique_vals[i], avg_labels[i] and avg_preds[i] work together
def avg_labels_and_preds_group_by(data_X, data_y, data_info, group_by_col, predictor):
    # pick unique values on the column with the number of attacks
    unique_vals = np.sort(np.unique(data_info[:, group_by_col]))
    avg_labels = np.zeros(unique_vals.size)
    if predictor is not None:
        avg_preds = np.zeros(unique_vals.size)
    else:
        avg_preds = None
    # create the mask (row y/n) using the info matrix, but apply it to the data matrix
    # this way we can filter data rows using information not provided to the learning alg
    for i, val in enumerate(unique_vals):
        relevant_idx = data_info[:, group_by_col] == val
        relevant_data_X = data_X[relevant_idx, :]
        relevant_data_y = data_y[relevant_idx]
        avg_labels[i] = np.mean(relevant_data_y)
        if predictor is not None:
            avg_preds[i] = np.mean(predictor(relevant_data_X))
    return unique_vals, avg_labels, avg_preds


# for each group of examples it finds, returns the occurrences of the specified label in the actual results
# and in the predicted results
def count_label_in_labels_and_preds_group_by(data_X, data_y, data_info, group_by_col, predictor, label):
    unique_vals = np.sort(np.unique(data_info[:, group_by_col]))
    actual_cnt = np.zeros(unique_vals.size)
    pred_cnt = np.zeros(unique_vals.size)
    # create the mask (row y/n) using the info matrix, but apply it to the data matrix
    # this way we can filter data rows using information not provided to the learning alg
    for i, val in enumerate(unique_vals):
        relevant_idx = data_info[:, group_by_col] == val
        relevant_data_X = data_X[relevant_idx, :]
        relevant_data_y = data_y[relevant_idx]
        predictions = predictor(relevant_data_X)
        actual_cnt[i] = np.count_nonzero(relevant_data_y == label)
        pred_cnt[i] = np.count_nonzero(predictions == label)
    return unique_vals, actual_cnt, pred_cnt


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


def train_regr_model(train_X, train_y, X_col_names, var_thresh, poly_feat, standardize, model_name, feat_sel_name):
    global logger
    # make a local copy of the objects we received as parameters and might change
    train_X = train_X.copy()  # make a local copy of the array
    X_col_names = list(X_col_names)  # make a local copy of the list
    model_name = model_name.lower()

    # objects for preprocessing and feature selection, in the order they were used
    transformers = []

    if var_thresh is True:
        base_feature_cnt = train_X.shape[1]
        vt_sel = VarianceThreshold()
        train_X = vt_sel.fit_transform(train_X)
        vt_feature_mask = vt_sel.get_support()
        X_col_names = [item for item_num, item in enumerate(X_col_names) if vt_feature_mask[item_num]]
        sel_feature_cnt = train_X.shape[1]
        logger.debug('VarianceThreshold removed {} features'.format(base_feature_cnt - sel_feature_cnt))
        transformers.append(vt_sel)

    # create polynomial features, interactions allowing us to learn a more complex prediction function
    if poly_feat is True:
        poly = preprocessing.PolynomialFeatures(4, interaction_only=False)
        train_X = poly.fit_transform(train_X)
        X_col_names = poly.get_feature_names(X_col_names)
        logger.debug('Polynomial X_col_names = {}'.format(X_col_names))
        logger.debug('train_X with polynomial features (first 2 rows)\n{}'.format(train_X[range(2), :]))
        transformers.append(poly)

    # apply a standardization step
    if standardize is True:
        scaler = preprocessing.StandardScaler()
        train_X = scaler.fit_transform(train_X)
        transformers.append(scaler)

    if model_name == 'linearregression':
        # plain linear regression can work as polynomial regression too
        clf = linear_model.LinearRegression()

    # # other polynomial regression models with built-in regularization and cross validation
    # # regularization is used to lower the weight given to less important and useless features
    # # cross validation is used to tune the hyperparameters, like alpha
    elif model_name == 'ridgecv':
        alphas = (0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 25.0, 50.0)
        clf = linear_model.RidgeCV(alphas)
    elif model_name == 'lassocv':
        clf = linear_model.LassoCV(max_iter=100000)
    elif model_name == 'elasticnetcv':
        l1_ratios = [.01, .05, .1, .3, .5, .7, .9, .95, .99, 1]
        clf = linear_model.ElasticNetCV(l1_ratio=l1_ratios, cv=5, max_iter=40000)
    elif model_name == 'decisiontreeregressor':
        # other useful parameters are max_depth and min_samples_leaf
        clf = tree.DecisionTreeRegressor(criterion='mse', max_depth=3)
    else:
        raise ValueError('Unsupported value for parameter model_name')

    # model selection by feature selection
    if feat_sel_name not in [None, '']:
        feat_sel_name = feat_sel_name.lower()
        if feat_sel_name == 'rfe':
            selector = RFE(clf, step=1, n_features_to_select=20)
        elif feat_sel_name == 'rfecv':
            selector = RFECV(clf, step=1, cv=5, scoring='neg_mean_absolute_error')
        elif feat_sel_name == 'selectfrommodel':
            selector = SelectFromModel(clf)
        else:
            raise ValueError('Unsupported value for parameter feat_sel_name')

        selector.fit(train_X, train_y)

        if feat_sel_name == 'selectfrommodel':
            train_X, selector = iterate_sfm_transform(selector, train_X, 20, 100, 0.01, 0.01)
        else:
            train_X = selector.transform(train_X)

        sel_feature_mask = selector.get_support()
        X_col_names = [item for item_num, item in enumerate(X_col_names) if sel_feature_mask[item_num]]
        logger.info('Selected features = {}'.format(X_col_names))
        logger.debug('After final transform, train_X.shape[1] = {}'.format(train_X.shape[1]))

        # TODO: make sure this is correct, maybe prediction can only be done with selector and not with clf
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

    logger.debug('learning done')

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
    fig, ax = setup_2d_axes('# of attacked power nodes', 'Measured fraction', ylim=(0, 1))
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


# TODO: use bottom=prev_class to make stacked bars, [(label, [(atk_size, label_cnt)])]
# TODO: add size checks on arrays and better explanation
# height_lists is a list of lists, each list contains the heights of bars of the same type (with the same color)
def plot_label_cnts_by_atk_size(labels, atk_sizes, label_cnts):
    # if not len(labels) == len(atk_sizes) == len(label_cnts):
    #     raise ValueError('The 3 parameters should be arrays of the same size')
    bar_width = 0.2  # the width of the bars
    space = 0.4  # space between the visual groups of bars
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

    labels_cnt = len(labels)  # number of bars in a group
    groups_cnt = len(atk_sizes)  # number of visual groups of bars

    # this works like [(label, [(atk_size, position)])]
    positions_by_label = []  # the x locations for each bar of each type
    for i in range(0, labels_cnt):
        positions_by_label.append([])

    cur_pos = 0
    cur_label_idx = 0
    middles = []
    for i in range(0, groups_cnt):
        group_middle = cur_pos + (bar_width * labels_cnt / 2)
        middles.append(group_middle)
        for j in range(0, labels_cnt):
            # the second bar of the first visual group is the first representation of the second label type
            positions_by_label[cur_label_idx].append(cur_pos)
            cur_pos += bar_width
            cur_label_idx = (cur_label_idx + 1) % labels_cnt
        cur_pos += space

    fig, ax = plt.subplots()
    first_rects = []  # contains the first bar of each label type
    for j in range(0, labels_cnt):
        positions = positions_by_label[j]
        heights = label_cnts[j]
        first_rects.append(ax.bar(positions, heights, bar_width, color=colors[j])[0])

    ax.set_xlabel('#attacked nodes')
    ax.set_ylabel('#simulations')
    ax.set_xticks(middles)
    ax.set_xticklabels(atk_sizes)

    ax.set_title('Occurrences of massive destruction ordered by attack size')
    ax.legend(first_rects, labels, fontsize=10)
    plt.tight_layout()
    plt.show()


def train_model_on_dataset(config, model_num):
    model_conf = config['model_trainings'][model_num]
    dataset_num = model_conf['dataset_num']
    output_dir = model_conf['output_dir']
    model_name = model_conf['model_name']
    var_thresh = model_conf['variance_threshold']
    poly_feat = model_conf['polynomial_features']
    standardize = model_conf['standardize']
    feat_sel_name = model_conf['feature_selection']

    dataset = config['datasets'][dataset_num]
    dataset_fpath = dataset['fpath']
    X_col_names = dataset['X_col_names']
    y_col_name = dataset['y_col_name']
    info_col_names = dataset['info_col_names']

    # find columns
    train_X, train_y, train_info = load_dataset(dataset_fpath, X_col_names, y_col_name, info_col_names)

    model, transformers, train_X, transf_X_col_names = \
        train_regr_model(train_X, train_y, X_col_names, var_thresh, poly_feat, standardize, model_name, feat_sel_name)

    if 'tree' in model_name.lower():
        # save a representation of the learned decision tree, to plot it, install graphviz
        # it can be turned into an image by running a command like the following
        # dot -T png decision_tree.dot -o decision_tree.png
        with open(os.path.join(output_dir, 'decision_tree_{}_.dot'.format(model_num)), 'w') as f:
            tree.export_graphviz(model, feature_names=transf_X_col_names, out_file=f)

    # save the learned model and what is needed to adapt the data to it
    learned_stuff_fpath = os.path.join(output_dir, 'model_{}_.pkl'.format(model_num))
    learned_stuff = {'model': model, 'transformers': transformers}
    joblib.dump(learned_stuff, learned_stuff_fpath)

    return model, transformers, transf_X_col_names


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

            # TODO: refactor this, most functions could simply use the info column instead of its index
            atks_cnt_col = info_col_names.index('#atkd_a')
            plain_ds_X, ds_y, ds_info = load_dataset(dataset_fpath, X_col_names, y_col_name, info_col_names)

            if 'filter' in plot_conf:
                filter_conf = plot_conf['filter']
                plain_ds_X, ds_y, ds_info =\
                    filter_dataset(plain_ds_X, ds_y, ds_info, X_col_names, info_col_names, filter_conf)

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
            atk_sizes, costs, error_stdevs = calc_cost_group_by(ds_X, ds_y, ds_info, atks_cnt_col, predictor)
            plot_cost_by_atk_size(atk_sizes, costs, error_stdevs)

        if plot_name == 'deaths_and_preds_by_atk_size':
            atk_sizes, avg_deaths, avg_preds = \
                avg_labels_and_preds_group_by(ds_X, ds_y, ds_info, atks_cnt_col, predictor)
            plot_deaths_and_preds_by_atk_size(atk_sizes, avg_deaths, avg_preds)

        if 'overlays' in plot_conf:
            overlays = plot_conf['overlays']

            if plot_name in ['cost_by_atk_size_many', 'deaths_and_preds_by_atk_size_many']:
                ax_x_label = plot_conf['ax_x_label']
                ax_y_label = plot_conf['ax_y_label']
                fig, ax = setup_2d_axes(ax_x_label, ax_y_label)
                ax.grid(linestyle='-', linewidth=0.5)

            for overlay in overlays:
                data_label = overlay['label']
                dataset_num = overlay['dataset_num']
                dataset = config['datasets'][dataset_num]
                X_col_names = dataset['X_col_names']
                y_col_name = dataset['y_col_name']
                info_col_names = dataset['info_col_names']
                dataset_fpath = dataset['fpath']

                # TODO: refactor this, most functions could simply use the info column instead of its index
                atks_cnt_col = info_col_names.index('#atkd_a')
                plain_ds_X, ds_y, ds_info = load_dataset(dataset_fpath, X_col_names, y_col_name, info_col_names)

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
                    atk_sizes, costs, error_stdevs = calc_cost_group_by(ds_X, ds_y, ds_info, atks_cnt_col, predictor)
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

                    if 'model_num' in overlay:
                        atk_sizes, avg_deaths, avg_preds = \
                            avg_labels_and_preds_group_by(ds_X, ds_y, ds_info, atks_cnt_col, predictor)
                        if x_multiplier != 1 or y_multiplier != 1:
                            plt.plot(x_multiplier * atk_sizes, y_multiplier * avg_preds, style, label=data_label)
                        else:
                            plt.plot(atk_sizes, avg_preds, style, label=data_label)
                    else:
                        atk_sizes, avg_deaths, avg_preds = \
                            avg_labels_and_preds_group_by(plain_ds_X, ds_y, ds_info, atks_cnt_col, predictor)
                        if x_multiplier != 1 or y_multiplier != 1:
                            plt.plot(x_multiplier * atk_sizes, y_multiplier * avg_deaths, style, label=data_label)
                        else:
                            plt.plot(atk_sizes, avg_deaths, style, label=data_label)
                        # if avg_deaths.shape[0] == 1:
                        #     plt.axhline(avg_deaths[0], color='red', linestyle='--', label=data_label)
                        # else:
                        #     plt.plot(atk_sizes, avg_deaths, style, label=data_label)

            ax.legend()
            plt.tight_layout()
            plt.show()


def run():
    conf_fpath = './dataset2.json'
    with open(conf_fpath) as conf_file:
        config = json.load(conf_file)

    models = []
    model_trainings = config['model_trainings']
    for model_num in range(0, len(model_trainings)):
        model, transformers, transf_X_col_names = train_model_on_dataset(config, model_num)
        models.append({'model': model, 'transformers': transformers})

    make_plots(config, models)


def run_old():

    # train_set_fpath = '/home/agostino/Documents/Sims/netw_a_0-100/0-100_union/equispaced_train_union.tsv'
    # test_set_fpath = '/home/agostino/Documents/Sims/netw_a_0-100/0-100_union/test_union.tsv'
    # train_set_fpath = '/home/agostino/Documents/Sims/single_net_20170725/train_1000_n_20_s.tsv'
    # test_set_fpath = '/home/agostino/Documents/Sims/single_net_20170725/test_1000_n_20_s.tsv'
    # train_set_fpath = '/home/agostino/Documents/Simulations/test_mp_12/train_1000_n_20_s.tsv'
    # test_set_fpath = '/home/agostino/Documents/Simulations/test_mp_12/test_1000_n_20_s.tsv'
    # train_set_fpath = '/home/agostino/Documents/Simulations/test_mp_11/train_2000_n_20_s_atkd_a.tsv'
    # test_set_fpath = '/home/agostino/Documents/Simulations/test_mp_11/test_2000_n_20_s_atkd_a.tsv'
    # train_set_fpath = '/home/agostino/Documents/Simulations/test_mp_11/train_2000_n_20_s_atkd_b.tsv'
    # test_set_fpath = '/home/agostino/Documents/Simulations/test_mp_11/test_2000_n_20_s_atkd_b.tsv'
    train_set_fpath = '/home/agostino/Documents/Simulations/test_mp_mn/train_mn.tsv'
    test_set_fpath = '/home/agostino/Documents/Simulations/test_mp_mn/test_mn.tsv'

    output_dir = '/home/agostino/Documents/Simulations/test_mp_mn'
    # output_dir = '/home/agostino/Documents/Simulations/test_mp_12/'

    # X_col_names = ['p_atkd', 'p_atkd_a', 'p_atkd_b',
    #                   'indeg_c_ab_q_1', 'indeg_c_ab_q_2', 'indeg_c_ab_q_3', 'indeg_c_ab_q_4', 'indeg_c_ab_q_5',
    #                   'indeg_c_i_q_1', 'indeg_c_i_q_2', 'indeg_c_i_q_3', 'indeg_c_i_q_4', 'indeg_c_i_q_5',
    #                   'rel_betw_c_q_1', 'rel_betw_c_q_2', 'rel_betw_c_q_3', 'rel_betw_c_q_4', 'rel_betw_c_q_5',
    #                   'p_atkd_gen', 'p_atkd_ts', 'p_atkd_ds', 'p_atkd_rel', 'p_atkd_cc']
    # y_col_name = 'p_dead'
    # info_col_names = ['instance', 'seed', '#atkd']
    # X_col_names = ['p_atkd_a', 'p_atkd_ds', 'p_atkd_gen', 'p_atkd_ts',
    #                   'p_q_1_atkd_betw_c_a', 'p_q_1_atkd_betw_c_ab', 'p_q_1_atkd_betw_c_i',
    #                   'p_q_1_atkd_deg_c_a', 'p_q_1_atkd_indeg_c_i',
    #                   'p_q_1_atkd_katz_c_ab', 'p_q_1_atkd_katz_c_i', 'p_q_1_atkd_ts_betw_c',
    #                   'p_q_2_atkd_betw_c_a', 'p_q_2_atkd_betw_c_ab',
    #                   'p_q_2_atkd_deg_c_a', 'p_q_2_atkd_indeg_c_i',
    #                   'p_q_2_atkd_katz_c_ab', 'p_q_2_atkd_katz_c_i', 'p_q_2_atkd_ts_betw_c',
    #                   'p_q_3_atkd_betw_c_a', 'p_q_3_atkd_betw_c_ab', 'p_q_3_atkd_betw_c_i',
    #                   'p_q_3_atkd_deg_c_a', 'p_q_3_atkd_indeg_c_i',
    #                   'p_q_3_atkd_katz_c_ab', 'p_q_3_atkd_katz_c_i', 'p_q_3_atkd_ts_betw_c',
    #                   'p_q_4_atkd_betw_c_a', 'p_q_4_atkd_betw_c_ab',
    #                   'p_q_4_atkd_deg_c_a', 'p_q_4_atkd_indeg_c_i',
    #                   'p_q_4_atkd_katz_c_ab', 'p_q_4_atkd_katz_c_i', 'p_q_4_atkd_ts_betw_c',
    #                   'p_q_5_atkd_betw_c_a', 'p_q_5_atkd_betw_c_ab', 'p_q_5_atkd_betw_c_i',
    #                   'p_q_5_atkd_deg_c_a', 'p_q_5_atkd_indeg_c_i',
    #                   'p_q_5_atkd_katz_c_ab', 'p_q_5_atkd_katz_c_i', 'p_q_5_atkd_ts_betw_c',
    #                   'p_tot_atkd_betw_c_a', 'p_tot_atkd_betw_c_ab', 'p_tot_atkd_betw_c_i',
    #                   'p_tot_atkd_deg_c_a', 'p_tot_atkd_indeg_c_i',
    #                   'p_tot_atkd_katz_c_ab', 'p_tot_atkd_katz_c_i', 'p_tot_atkd_ts_betw_c'
    #                   ]
    # very good set, p_tot_atkd_betw_c_ab can be cut
    # X_col_names = ['p_atkd_ds',
    #                   'p_q_5_atkd_betw_c_ab', 'p_q_5_atkd_betw_c_i',
    #                   'p_q_5_atkd_indeg_c_i', 'p_q_5_atkd_ts_betw_c',
    #                   'p_tot_atkd_betw_c_ab', 'p_tot_atkd_betw_c_i',
    #                   'p_tot_atkd_indeg_c_i', 'p_tot_atkd_ts_betw_c'
    #                   ]
    # quite good set
    # X_col_names = ['p_atkd_a', 'p_atkd_ds', 'p_atkd_ts',
    #                   'p_q_4_atkd_betw_c_ab', 'p_q_4_atkd_betw_c_i',
    #                   'p_q_4_atkd_indeg_c_i', 'p_q_4_atkd_ts_betw_c',
    #                   'p_q_5_atkd_betw_c_ab', 'p_q_5_atkd_betw_c_i',
    #                   'p_q_5_atkd_indeg_c_i', 'p_q_5_atkd_ts_betw_c',
    #                   'p_tot_atkd_betw_c_ab', 'p_tot_atkd_betw_c_i',
    #                   'p_tot_atkd_indeg_c_i', 'p_tot_atkd_ts_betw_c'
    #                   ]
    # X_col_names = ['p_tot_atkd_betw_c_i', 'p_atkd_cc']
    # X_col_names = ['p_atkd_a', 'p_tot_atkd_betw_c_i']
    X_col_names = ['p_atkd_a', 'p_tot_atkd_betw_c_i', 'p_tot_atkd_ts_betw_c']
    # X_col_names = ['p_atkd_b', 'p_tot_atkd_betw_c_i', 'p_tot_atkd_rel_betw_c']

    # also good for trees
    # X_col_names = ['p_atkd_a', 'p_tot_atkd_betw_c_i', 'p_tot_atkd_ts_betw_c', 'p_tot_atkd_betw_c_ab']

    # X_col_names = ['p_tot_atkd_betw_c_i', 'p_atkd_a', 'p_q_4_atkd_betw_c_i', 'p_q_5_atkd_betw_c_i']
    # X_col_names = ['p_atkd_ds', 'p_atkd_gen', 'p_atkd_ts',
    #                   'p_q_1_atkd_betw_c_a', 'p_q_1_atkd_betw_c_ab', 'p_q_1_atkd_betw_c_i',
    #                   'p_q_1_atkd_clos_c_a', 'p_q_1_atkd_clos_c_ab', 'p_q_1_atkd_clos_c_i',
    #                   'p_q_1_atkd_deg_c_a', 'p_q_1_atkd_indeg_c_ab', 'p_q_1_atkd_indeg_c_i',
    #                   'p_q_1_atkd_katz_c_ab', 'p_q_1_atkd_katz_c_i', 'p_q_1_atkd_ts_betw_c',
    #                   'p_q_2_atkd_betw_c_a', 'p_q_2_atkd_betw_c_ab',
    #                   'p_q_2_atkd_clos_c_a', 'p_q_2_atkd_clos_c_ab', 'p_q_2_atkd_clos_c_i',
    #                   'p_q_2_atkd_deg_c_a', 'p_q_2_atkd_indeg_c_ab', 'p_q_2_atkd_indeg_c_i',
    #                   'p_q_2_atkd_katz_c_ab', 'p_q_2_atkd_katz_c_i', 'p_q_2_atkd_ts_betw_c',
    #                   'p_q_3_atkd_betw_c_a', 'p_q_3_atkd_betw_c_ab', 'p_q_3_atkd_betw_c_i',
    #                   'p_q_3_atkd_clos_c_a', 'p_q_3_atkd_clos_c_ab', 'p_q_3_atkd_clos_c_i',
    #                   'p_q_3_atkd_deg_c_a', 'p_q_3_atkd_indeg_c_ab', 'p_q_3_atkd_indeg_c_i',
    #                   'p_q_3_atkd_katz_c_ab', 'p_q_3_atkd_katz_c_i', 'p_q_3_atkd_ts_betw_c',
    #                   'p_q_4_atkd_betw_c_a', 'p_q_4_atkd_betw_c_ab',
    #                   'p_q_4_atkd_clos_c_a', 'p_q_4_atkd_clos_c_ab', 'p_q_4_atkd_clos_c_i',
    #                   'p_q_4_atkd_deg_c_a', 'p_q_4_atkd_indeg_c_ab', 'p_q_4_atkd_indeg_c_i',
    #                   'p_q_4_atkd_katz_c_ab', 'p_q_4_atkd_katz_c_i', 'p_q_4_atkd_ts_betw_c',
    #                   'p_q_5_atkd_betw_c_a', 'p_q_5_atkd_betw_c_ab', 'p_q_5_atkd_betw_c_i',
    #                   'p_q_5_atkd_clos_c_a', 'p_q_5_atkd_clos_c_ab', 'p_q_5_atkd_clos_c_i',
    #                   'p_q_5_atkd_deg_c_a', 'p_q_5_atkd_indeg_c_ab', 'p_q_5_atkd_indeg_c_i',
    #                   'p_q_5_atkd_katz_c_ab', 'p_q_5_atkd_katz_c_i', 'p_q_5_atkd_ts_betw_c',
    #                   'p_tot_atkd_betw_c_a', 'p_tot_atkd_betw_c_ab', 'p_tot_atkd_betw_c_i',
    #                   'p_tot_atkd_clos_c_a', 'p_tot_atkd_clos_c_ab', 'p_tot_atkd_clos_c_i',
    #                   'p_tot_atkd_deg_c_a', 'p_tot_atkd_indeg_c_ab', 'p_tot_atkd_indeg_c_i',
    #                   'p_tot_atkd_katz_c_ab', 'p_tot_atkd_katz_c_i', 'p_tot_atkd_ts_betw_c'
    #                   ]
    # X_col_names = ['p_atkd_cc', 'p_atkd_ds', 'p_atkd_gen', 'p_atkd_rel', 'p_atkd_ts',
    #                   'p_q_1_atkd_indeg_c_i', 'p_q_2_atkd_indeg_c_i', 'p_q_3_atkd_indeg_c_i', 'p_q_4_atkd_indeg_c_i',
    #                   'p_q_5_atkd_indeg_c_i', 'p_tot_atkd_indeg_c_i',
    #                   'p_q_1_atkd_ts_betw_c', 'p_q_2_atkd_ts_betw_c', 'p_q_3_atkd_ts_betw_c', 'p_q_4_atkd_ts_betw_c',
    #                   'p_q_5_atkd_ts_betw_c', 'p_tot_atkd_ts_betw_c',
    #                   'p_q_1_atkd_katz_c_ab', 'p_q_2_atkd_katz_c_ab', 'p_q_3_atkd_katz_c_ab', 'p_q_4_atkd_katz_c_ab',
    #                   'p_q_5_atkd_katz_c_ab', 'p_tot_atkd_katz_c_ab'
    #                   ]
    y_col_name = 'p_dead'
    # y_col_name = 'dead_lvl'
    info_col_names = ['instance', 'seed', '#atkd_a']
    # info_col_names = ['instance', 'seed', '#atkd_b']

    atks_cnt_col = info_col_names.index('#atkd_a')
    # atks_cnt_col = info_col_names.index('#atkd_b')

    model_name = 'DecisionTreeRegressor'
    # model_name = 'elasticnetcv'
    # model_name = 'ridgecv'
    model_kind = 'regression'

    # find columns
    train_X, train_y, train_info = load_dataset(train_set_fpath, X_col_names, y_col_name, info_col_names)

    # model, transformers, transf_train_X, transf_X_col_names = \
    #     train_regr_model(train_X, train_y, X_col_names, True, True, True, model_name, 'rfecv')
    model, transformers, transf_train_X, transf_X_col_names = \
        train_regr_model(train_X, train_y, X_col_names, True, False, True, model_name, None)

    if 'tree' in model_name.lower():
        # save a representation of the learned decision tree, to plot it, install graphviz
        # it can be turned into an image by running a command like the following
        # dot -T png decision_tree.dot -o decision_tree.png
        with open(os.path.join(output_dir, 'decision_tree.dot'), 'w') as f:
            tree.export_graphviz(model, feature_names=transf_X_col_names, out_file=f)

    # save the learned model and what is needed to adapt the data to it
    learned_stuff_fpath = os.path.join(output_dir, 'model.pkl')
    learned_stuff = {'model': model, 'transformers': transformers}
    joblib.dump(learned_stuff, learned_stuff_fpath)

    predictor = lambda x: model.predict(x)  # function we use to predict the result

    datasets = [
        {
        #     'dataset_fpath': '/home/agostino/Documents/Simulations/test_mp_13/test_500_n_10_s.tsv',
        #     'relevant_atk_sizes': [0, 3, 5, 10, 15, 20, 25, 35, 50],
        #     'node_cnt_A': 500, 'name': '500 power nodes, 10 subnets'
        # }, {
        #     'dataset_fpath': '/home/agostino/Documents/Simulations/test_mp_13/test_500_n_20_s.tsv',
        #     'relevant_atk_sizes': [0, 3, 5, 10, 15, 20, 25, 35, 50],
        #     'node_cnt_A': 500, 'name': '500 power nodes, 20 subnets'
        # }, {
        #     'dataset_fpath': test_set_fpath,
        #     'relevant_atk_sizes': [0, 5, 10, 20, 30, 40, 50, 70, 100],
        #     'node_cnt_A': 1000, 'name': '1000 power nodes, 20 subnets'
        # }, {
        #     'dataset_fpath': '/home/agostino/Documents/Simulations/test_mp_14/test_2000_n_20_s.tsv',
        #     'relevant_atk_sizes': [0, 10, 20, 40, 60, 80, 100, 140, 200],
        #     'node_cnt_A': 2000, 'name': '2000 power nodes, 20 subnets'
        # }, {
        #     'dataset_fpath': '/home/agostino/Documents/Simulations/test_mp_14/test_2000_n_40_s.tsv',
        #     'relevant_atk_sizes': [0, 10, 20, 40, 60, 80, 100, 140, 200],
        #     'node_cnt_A': 2000, 'name': '2000 power nodes, 40 subnets'
        # }
        #
            'dataset_fpath': test_set_fpath,
            'relevant_atk_sizes': [0, 6, 11, 22, 33, 44, 55, 77, 88, 99, 109, 120, 131, 142, 153, 164, 175, 186, 197,
                                   208, 218, 229, 240, 251, 262, 273, 284, 295, 306, 317, 327],
            'node_cnt_A': 1091, 'name': 'MN power grid'
        }
    ]

    for dataset in datasets:
        plain_ds_X, ds_y, ds_info = load_dataset(dataset['dataset_fpath'], X_col_names, y_col_name, info_col_names)
        ds_X = plain_ds_X.copy()
        for transformer in transformers:
            logger.debug('Applying {}.transform'.format(type(transformer).__name__))
            ds_X = transformer.transform(ds_X)

        predictions = predictor(ds_X)

        # visualize how two features affect the results
        # TODO: find the columns of these features by name, instead of hardcoding their position!
        ax_x_vec, ax_x_label = plain_ds_X[:, 0], 'initial fraction of failed nodes'
        ax_y_vec, ax_y_label = plain_ds_X[:, 1], 'loss of centrality'
        ax_z_vec, ax_z_label = ds_y, 'actual resulting fraction of dead nodes'
        # plot_3d_no_interpolate(ax_x_vec, ax_y_vec, ax_z_vec, ax_x_label, ax_y_label, ax_z_label, True, True, False)

        # show the original features on the x and y axis; show the predictions on the z axis
        ax_z_vec, ax_z_label = predictions, 'predicted fraction of dead nodes'
        # plot_3d_no_interpolate(ax_x_vec, ax_y_vec, ax_z_vec, ax_x_label, ax_y_label, ax_z_label, True, False, True)

        # plot_predictions_for_2d_dataset(plain_ds_X, ds_y, transformers, predictor)

        if model_kind == 'regression':
            check_prediction_bounds(plain_ds_X, ds_info, X_col_names, info_col_names, predictions, 0.0, True, 1.05, True)

            atk_sizes, costs, error_stdevs = calc_cost_group_by(ds_X, ds_y, ds_info, atks_cnt_col, predictor)
            plot_cost_by_atk_size(atk_sizes, costs, error_stdevs)

            atk_sizes, avg_deaths, avg_preds = \
                avg_labels_and_preds_group_by(ds_X, ds_y, ds_info, atks_cnt_col, predictor)
            plot_deaths_and_preds_by_atk_size(atk_sizes, avg_deaths, avg_preds)

        else:
            atk_sizes, accuracies, f1_scores =\
                calc_scores_group_by(ds_X, ds_y, ds_info, atks_cnt_col, predictor)
            atk_sizes, actual_cnt, pred_cnt =\
                count_label_in_labels_and_preds_group_by(ds_X, ds_y, ds_info, atks_cnt_col, predictor, 'total')

        # only keep the data we want
        relevant_idx = []
        relevant_atk_sizes = dataset['relevant_atk_sizes']
        for atk_size_idx, atk_size in enumerate(atk_sizes):
            if atk_size in relevant_atk_sizes:
                relevant_idx.append(atk_size_idx)

        atk_sizes = [atk_sizes[i] for i in relevant_idx]

        if model_kind == 'regression':
            costs = costs[relevant_idx]
            error_stdevs = error_stdevs[relevant_idx]
            avg_deaths = avg_deaths[relevant_idx]
            avg_preds = avg_preds[relevant_idx]

            dataset['results'] = {'atk_sizes': atk_sizes, 'costs': costs, 'error_stdevs': error_stdevs,
                                  'avg_deaths': avg_deaths, 'avg_preds': avg_preds}
        else:
            accuracies = accuracies[relevant_idx]
            f1_scores = f1_scores[relevant_idx]

            dataset['results'] = {'atk_sizes': atk_sizes, 'accuracies': accuracies, 'f1_scores': f1_scores,
                                  'actual #total deaths': actual_cnt, 'predicted #total deaths': pred_cnt}

        # plot_all_scenario_performances(ds_X, ds_y, ds_info, info_col_names, predictor, 3, 2)

    # TODO: calculate these, they are the fractions of attacked nodes
    # atkd_ps = [0.5, 1., 2., 5., 10.]
    # atkd_ps = [0, 0.5, 1, 2, 3, 4, 5, 7, 10]
    atkd_ps = [0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16,
               0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3]
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    markers = ['o', '^', 's', '*', 'x', '+', 'd']
    styles = ['b-o', 'g-^', 'r-s', 'c-*', 'm-x', 'y-+']

    if model_kind == 'regression':
        lines_to_plot = []
        for i, dataset in enumerate(datasets):
            line = {'x': atkd_ps, 'y': dataset['results']['avg_deaths'], 'style': styles[i], 'label': dataset['name']}
            lines_to_plot.append(line)
        ax = setup_2d_axes('% of attacked power nodes', 'Actual avg dead fraction (pow+tel)',
                           xticks=atkd_ps, ylim=(0, 1.1))
        plot_2d_lines(lines_to_plot, ax)

        lines_to_plot = []
        for i, dataset in enumerate(datasets):
            line = {'x': atkd_ps, 'y': dataset['results']['avg_preds'], 'style': styles[i], 'label': dataset['name']}
            lines_to_plot.append(line)
        ax = setup_2d_axes('% of attacked power nodes', 'Predicted avg dead fraction (pow+tel)',
                           xticks=atkd_ps, ylim=(0, 1.1))
        plot_2d_lines(lines_to_plot, ax)

    else:
        # TODO: now bar chart with groups of 4 bars, describing for each atk size:
        # number of scenarios with an actual "total" death_lvl label
        # number of scenarios predicted a predicted "total" death_lvl label
        # accuracy and f1 score
        # TODO: later, make the number of scenarios a fraction

        for i, dataset in enumerate(datasets):
            lines_to_plot = []
            plot_actual_and_pred_cnts_by_atk_size(dataset['results']['atk_sizes'],
                                                  dataset['results']['actual #total deaths'],
                                                  dataset['results']['predicted #total deaths'])
            line = {'x': atkd_ps, 'y': dataset['results']['f1_scores'], 'style': 'b-o', 'label': 'f1 score'}
            lines_to_plot.append(line)
            line = {'x': atkd_ps, 'y': dataset['results']['accuracies'], 'style': 'g-^', 'label': 'accuracy'}
            lines_to_plot.append(line)
            ax = setup_2d_axes('Prediction model scores', '% of attacked power nodes', xticks=atkd_ps, ylim=(0,))
            plot_2d_lines(lines_to_plot, ax)


    # use all_results to plot the graph
    # need atk_sizes, avg_deaths, avg_preds from plot_deaths_and_preds_by_atk_size
    # and somehow error_stdevs from plot_cost_by_atk_size

def test_plot():
    # [(label, [(atk_size, label_cnt)])]
    labels = ['a1', 'p1', 'a2', 'p2']
    height_lists = [[10, 20], [20,30], [30, 40], [40, 50]]
    plot_label_cnts_by_atk_size(labels, [1, 2], height_lists)

# test_plot()

setup_logging('logging_base_conf.json')
run()
