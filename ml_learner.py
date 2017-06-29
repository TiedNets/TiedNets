import csv
import json
import random
import logging.config
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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


# find the parameters used to "standardize" the features of our examples
def find_mu_and_sigma(X, col_names):
    global logger
    if X.shape[1] != len(col_names):
        raise ValueError('The number of columns in X and the number'
                         'of column names do not match, {} != {}'.format(X.shape[1], len(col_names)))
    mu = np.mean(X, 0)  # mean of each column
    sigma = np.std(X, 0, ddof=1)
    logger.debug('mu shape = {} mu = {}\nsigma shape = {} sigma = {}'.format(mu.shape, mu, sigma.shape, sigma))

    for i in range(sigma.size):
        if sigma[i] == 0:
            logger.info('Column {} of X is full of identical values {}'.format(col_names[i], X[0, i]))
            sigma[i] = 1

    return mu, sigma


def find_mu_and_sigma_bis(X):
    global logger
    mu = np.mean(X, 0)  # mean of each column
    sigma = np.std(X, 0, ddof=1)
    logger.debug('mu shape = {} mu = {}\nsigma shape = {} sigma = {}'.format(mu.shape, mu, sigma.shape, sigma))

    for i in range(sigma.size):
        if sigma[i] == 0:
            logger.info('Column {} of X is full of identical values {}'.format(i, X[0, i]))
            sigma[i] = 1

    return mu, sigma


# Applies the pre-calculated standardization parameters to X
# the mean value of each feature becomes 0 and the standard deviation becomes 1
# The proper term for this operation is "standardization", but in ML we have this frequent misnomer
# do not confuse standardization (works by column) with normalization (works by row)
def apply_standardization(X, mu, sigma):
    global logger
    logger.debug('X.dtype = {}, sigma.dtype = {}'.format(X.dtype, sigma.dtype))
    if 0.0 in sigma:
        raise RuntimeError('The sigma vector should not contain zeros')

    return (X - mu) / sigma


# the gradient descent update function is derived from this cost function
def calc_cost(X, y, theta):
    m = y.size  # number of examples given
    predictions = X.dot(theta)
    errors = (predictions - y)
    cost = (1.0 / (2 * m)) * errors.T.dot(errors)  # half the average of squared errors

    return cost


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


# the gradient descent algorithm changes theta by steps, improving it little by little
# alpha is the "learning rate", it's in the range (0, 1], and it limits the size of the correction happening each step
# it learns the multivariate linear regression that best fits our datapoints
# since multivariate linear regressions happens in an n dimensional space (n = #variables), the polynomial solution we
# will find, the "hypothesis" used to fit the data, will not represent a simple 2D line
# what we learn is not guaranteed to be the optimal (best possible) linear regression, but on larger datasets it's
# much faster to find a good enough solution than to calculate the analytical solution
def gradient_descent_ext(train_X, train_y, theta, alpha, num_iters):
    global logger
    m = train_y.size
    theta_history = np.zeros((num_iters, train_X.shape[1]))

    # n is the number of features (the columns of X)
    # X has size (m x n), theta has size (n x 1), so their product is (m x 1)
    logger.debug('X.shape = {}, y.shape = {}, theta.shape = {}'.format(train_X.shape, train_y.shape, theta.shape))

    for i in range(num_iters):
        h_predicts = train_X.dot(theta)
        errors = h_predicts - train_y  # note that errors are not squared here
        # aside from alpha, this function is derived from the cost function
        theta_change = alpha * train_X.T.dot(errors) / m
        theta -= theta_change
        theta_history[i] = theta

    return theta_history


def calc_cost_histories(train_X, train_y, test_X, test_y, theta_history, cost_function):
    num_iters = theta_history.shape[0]
    train_cost_history = np.zeros(num_iters)
    test_cost_history = np.zeros(num_iters)

    for i in range(num_iters):
        theta = theta_history[i]
        train_cost_history[i] = cost_function(train_X, train_y, theta)
        test_cost_history[i] = cost_function(test_X, test_y, theta)

    return train_cost_history, test_cost_history


def setup_2d_axes(xlabel, ylabel, xlim=None, ylim=None, xticks=None, yticks=None):
    ax = plt.axes()
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

    return ax


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

    return ax


def plot_cost_history(train_cost_history, test_cost_history, xlabel, ylabel):
    setup_2d_axes(xlabel, ylabel)

    num_iters = train_cost_history.size
    plt.plot(range(num_iters), train_cost_history, label='training set')
    plt.plot(range(num_iters), test_cost_history, label='test set')

    plt.tight_layout()  # make sure everything is showing
    plt.show()


def plot_2d_line(x, y, xlabel, ylabel, line_label, std_devs=None):
    setup_2d_axes(xlabel, ylabel)
    plt.plot(x, y, 'b-o', label=line_label)
    if std_devs is not None:
        plt.errorbar(x, y, std_devs)
    plt.tight_layout()  # make sure everything is showing
    plt.show()


# lines is a list of dictionaries [{x, y, line_style, line_label}, {...}, ...]
def plot_2d_lines(lines, ax):
    for line in lines:
        plt.plot(line['x'], line['y'], line['style'], label=line['label'])

    ax.grid(linestyle='-', linewidth=0.5)
    plt.tight_layout()  # make sure everything is showing
    plt.show()


def plot_scenario_performances(x_values, results, predictions, xlabel, ylabel):
    setup_2d_axes(xlabel, ylabel)

    plt.plot(x_values, results, 'g-o', label='results')
    plt.plot(x_values, predictions, 'r-o', label='predictions')

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

    ax = setup_3d_axes(ax_x_label, ax_y_label, ax_z_label,
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

    ax = setup_3d_axes(ax_x_label, ax_y_label, ax_z_label,
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
        raise ValueError('No simulation in the dataset passed by parameter "data_X" matches the conditions specified by'
                         'the parameter "info_filter"\n{}'.format(info_filter))

    # we are under the assumption that each scenario only has a simulation for each number of attacks
    atk_counts = subset_info[:, atks_cnt_col]
    if atk_counts.shape != (np.unique(atk_counts)).shape:
        raise RuntimeError('This function assumes that, for a given scenario, only one simulation for each quantity of'
                           'attacks. Check the value of the parameter "atks_cnt_col", then check your dataset.')

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


def calc_cost_by_atk_size(data_X, data_y, data_info, atks_cnt_col, predictor):
    # pick unique values on the column with the number of attacks
    atk_sizes = np.sort(np.unique(data_info[:, atks_cnt_col]))
    var_costs = np.zeros(atk_sizes.size)
    error_std_devs = np.zeros(atk_sizes.size)
    # create the mask (row y/n) using the info matrix, but apply it to the data matrix
    # this way we can filter data rows using information not provided to the learning alg
    for i, atk_size in enumerate(atk_sizes):
        relevant_test_idx = data_info[:, atks_cnt_col] == atk_size
        relevant_data_X = data_X[relevant_test_idx, :]
        relevant_data_y = data_y[relevant_test_idx]
        var_costs[i], error_std_devs[i] = calc_my_cost(relevant_data_X, relevant_data_y, predictor)
    return atk_sizes, var_costs, error_std_devs


# possible improvement, refactor to accept a list of score functions and return a list of scores for each atk size,
# or just accept a list of functions to apply and return a list of results
def calc_scores_by_atk_size(data_X, data_y, data_info, atks_cnt_col, predictor):
    # pick unique values on the column with the number of attacks
    atk_sizes = np.sort(np.unique(data_info[:, atks_cnt_col]))
    accuracies = np.zeros(atk_sizes.size)
    f1_scores = np.zeros(atk_sizes.size)
    # create the mask (row y/n) using the info matrix, but apply it to the data matrix
    # this way we can filter data rows using information not provided to the learning alg
    for i, atk_size in enumerate(atk_sizes):
        relevant_test_idx = data_info[:, atks_cnt_col] == atk_size
        relevant_data_X = data_X[relevant_test_idx, :]
        relevant_data_y = data_y[relevant_test_idx]
        predictions = predictor(relevant_data_X)
        accuracies[i] = accuracy_score(relevant_data_y, predictions)
        # change if this is not a binary classification
        f1_scores[i] = f1_score(relevant_data_y, predictions, pos_label='low', average='binary')
    return atk_sizes, accuracies, f1_scores


# returns three arrays:
# 1) atk_sizes, the different numbers of attacks operated on the network, from lowest to highest
# 2) avg_deaths, the average fraction of dead nodes after the attacks
# 3) avg_preds, the average of the predicted fractions of dead nodes after the attacks
# The ith cells of avg_deaths and avg_preds are related to the number of attacks in atk_sizes[i]
def avg_deaths_and_preds_by_atk_size(data_X, data_y, data_info, atks_cnt_col, predictor):
    # pick unique values on the column with the number of attacks
    atk_sizes = np.sort(np.unique(data_info[:, atks_cnt_col]))
    avg_preds = np.zeros(atk_sizes.size)
    avg_deaths = np.zeros(atk_sizes.size)
    # create the mask (row y/n) using the info matrix, but apply it to the data matrix
    # this way we can filter data rows using information not provided to the learning alg
    for i, atk_size in enumerate(atk_sizes):
        relevant_test_idx = data_info[:, atks_cnt_col] == atk_size
        relevant_data_X = data_X[relevant_test_idx, :]
        relevant_data_y = data_y[relevant_test_idx]
        avg_preds[i] = np.mean(predictor(relevant_data_X))
        avg_deaths[i] = np.mean(relevant_data_y)
    return atk_sizes, avg_deaths, avg_preds


# counts the total occurrences of a label for each attack size
def count_actual_and_pred_labellings_by_atk_size(data_X, data_y, data_info, atks_cnt_col, predictor, label):
    # pick unique values on the column with the number of attacks
    atk_sizes = np.sort(np.unique(data_info[:, atks_cnt_col]))
    actual_cnt = np.zeros(atk_sizes.size)
    pred_cnt = np.zeros(atk_sizes.size)
    # create the mask (row y/n) using the info matrix, but apply it to the data matrix
    # this way we can filter data rows using information not provided to the learning alg
    for i, atk_size in enumerate(atk_sizes):
        relevant_test_idx = data_info[:, atks_cnt_col] == atk_size
        relevant_data_X = data_X[relevant_test_idx, :]
        relevant_data_y = data_y[relevant_test_idx]
        predictions = predictor(relevant_data_X)
        actual_cnt[i] = np.count_nonzero(relevant_data_y == label)
        pred_cnt[i] = np.count_nonzero(predictions == label)
    return atk_sizes, actual_cnt, pred_cnt


# TODO: refactor to take a single, loaded dataset and return the solution
# uses normal equations
def solve_and_test(train_set_fpath, test_set_fpath, X_col_names, y_col_name, info_col_names):
    # from each file we load load 3 sets of data, the examples (X), the labels (y) and another with related info
    # for each array we remember the correlation column name -> column number
    with open(train_set_fpath, 'r') as header_file:
        csvreader = csv.reader(header_file, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        header = csvreader.next()

    result_col = header.index(y_col_name)
    train_X = load_named_cols(train_set_fpath, X_col_names, header)
    train_y = np.loadtxt(train_set_fpath, delimiter='\t', skiprows=1, usecols=[result_col])
    train_info = load_named_cols(train_set_fpath, info_col_names, header)
    m = train_y.size  # number of training examples

    # print('train_X (first 2 rows)\n{}'.format(train_X[range(2), :]))  # debug
    # print('train_y (first 2 rows)\n{}'.format(train_y[range(2)]))  # debug
    print('train_info (first 2 rows)\n{}'.format(train_info[range(2), :]))  # debug

    test_X = load_named_cols(test_set_fpath, X_col_names, header)
    test_y = np.loadtxt(test_set_fpath, delimiter='\t', skiprows=1, usecols=[result_col])
    test_info = load_named_cols(test_set_fpath, info_col_names, header)

    # print('test_X (first 2 rows)\n{}'.format(test_X[range(2), :]))  # debug
    # print('test_y (first 2 rows)\n{}'.format(test_y[range(2)]))  # debug

    train_X = np.c_[np.ones(m, dtype=train_X.dtype), train_X]  # add intercept term (first column of ones)
    # print('train_X with intercept (first 2 rows)\n{}'.format(train_X[range(2), :]))  # debug

    X_col_names.insert(0, '')  # prepend empty label to keep column names aligned

    test_X = np.c_[np.ones(test_X.shape[0], dtype=test_X.dtype), test_X]  # add intercept term
    # print('test_X (first 2 rows)\n{}'.format(test_X[range(2), :]))  # debug

    theta = normal_equation(train_X, train_y)
    predictor = lambda x: x.dot(theta)

    # draw a graph showing the mean error for each #attacks
    atks_cnt_col = info_col_names.index('#atkd_a')  # TODO: this should not be hardcoded
    atk_sizes, test_costs = calc_cost_by_atk_size(test_X, test_y, test_info, atks_cnt_col, predictor)
    logger.debug('atk_sizes = {}'.format(atk_sizes))
    plot_2d_line(atk_sizes, test_costs, '# of attacked power nodes', 'Avg abs prediction error', 'test set')

    full_X = np.append(train_X, test_X, axis=0)
    full_y = np.append(train_y, test_y, axis=0)
    full_info = np.append(train_info, test_info, axis=0)

    for cur_seed in range(20):
        info_filter = {info_col_names.index('instance'): 0, info_col_names.index('seed'): cur_seed}
        indep_var_vals, results, predictions = \
            find_scenario_results_and_predictions(full_X, full_y, full_info, info_filter, atks_cnt_col, predictor)
        plot_scenario_performances(indep_var_vals, results, predictions, '# attacked nodes', '% dead nodes')


# iterately apply SelectFromModel.transform changing the threshold until we get the desired number of features
# If you have a test set, just use the returned fitted_sfm to transform it
def iterate_sfm_transform(fitted_sfm, train_X, max_feature_cnt, max_rounds, base_thresh, thresh_incr):
    temp_train_X = fitted_sfm.transform(train_X)
    sel_feature_cnt = temp_train_X.shape[1]

    if sel_feature_cnt > max_feature_cnt:
        rounds = 0
        fitted_sfm.threshold = base_thresh
        temp_train_X = fitted_sfm.transform(train_X)
        sel_feature_cnt = temp_train_X.shape[1]
        while sel_feature_cnt > max_feature_cnt and rounds < max_rounds:
            fitted_sfm.threshold += thresh_incr
            temp_train_X = fitted_sfm.transform(train_X)
            sel_feature_cnt = temp_train_X.shape[1]
            rounds += 1
    train_X = temp_train_X

    return train_X, fitted_sfm


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
        alphas = (0.1, 0.5, 1.0, 5.0, 10.0, 25.0, 50.0)
        clf = linear_model.RidgeCV(alphas)
    elif model_name == 'lassocv':
        clf = linear_model.LassoCV(max_iter=100000)
    elif model_name == 'elasticnetcv':
        clf = linear_model.ElasticNetCV(max_iter=40000)
    elif model_name == 'decisiontreeregressor':
        # other useful parameters are max_depth and min_samples_leaf
        clf = tree.DecisionTreeRegressor(criterion='mse', min_samples_split=0.1)
    else:
        raise ValueError('Unsupported value for parameter model_name')

    # model selection by feature selection
    if feat_sel_name in [None, '']:
        clf.fit(train_X, train_y)
    else:
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

        sel_feature_mask = selector.get_support()
        sel_features_names = [item for item_num, item in enumerate(X_col_names) if sel_feature_mask[item_num]]
        logger.debug('sel_features_names = {}'.format(sel_features_names))
        logger.debug('after transform train_X.shape[1] = {}'.format(train_X.shape[1]))

        # TODO: make sure this is correct, maybe prediction can only be done with selector and not with clf
        transformers.append(selector)

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


def plot_predictions_for_2d_dataset(X, y, transformers, predictor):
    if X.shape[1] != 2:
        raise ValueError('This function only works for datasets with exactly 2 features.')

    # create a dataset with uniformly spaced points
    x_grid, y_grid = make_uniform_grid_xy(X[:, 0], X[:, 1], 100, 100)

    # this is why we need X to only have 2 features, we need to recreate it
    uniform_X = np.c_[x_grid.ravel(), y_grid.ravel()]

    for transformer in transformers:
        uniform_X = transformer.transform(uniform_X)

    predictions = predictor(uniform_X)
    z_grid = np.reshape(predictions, x_grid.shape)

    ax_x_label = 'initial fraction of failed nodes'
    ax_y_label = 'loss of centrality'
    ax_z_label = 'predicted fraction of dead nodes'

    ax_x_vec, ax_y_vec, ax_z_vec = X[:, 0], X[:, 1], y
    plot_3d_lots(ax_x_label, ax_y_label, ax_z_label, ax_x_vec, ax_y_vec, ax_z_vec, x_grid, y_grid, z_grid)


def plot_all_scenario_performances(X, y, info, info_col_names, predictor, rnd_inst_cnt, rnd_seed_cnt):
    global logger

    atks_cnt_col = info_col_names.index('#atkd_a')
    instances = np.sort(np.unique(info[:, info_col_names.index('instance')]))
    sim_seeds = np.sort(np.unique(info[:, info_col_names.index('seed')]))
    logger.debug('All instances = {}\nAll sim seeds {}'.format(instances, sim_seeds))

    # TODO: pass the seed/shuffler from outside
    np.random.shuffle(instances)
    np.random.shuffle(sim_seeds)

    for cur_inst in instances[0:rnd_inst_cnt]:
        logger.info('instance = {}'.format(cur_inst))
        for cur_seed in sim_seeds[0:rnd_seed_cnt]:
            logger.info('seed = {}'.format(cur_seed))
            info_filter = {info_col_names.index('instance'): cur_inst, info_col_names.index('seed'): cur_seed}
            indep_var_vals, results, predictions = \
                find_scenario_results_and_predictions(X, y, info, info_filter, atks_cnt_col, predictor)
            plot_scenario_performances(indep_var_vals, results, predictions, '# of attacked power nodes', '% dead nodes')


# draw a graph showing the mean error for each #attacks, and the standard deviation of this error
def plot_cost_by_atk_size(atk_sizes, costs, error_stdevs):
    line_1 = {'x': atk_sizes, 'y': costs, 'style': 'b-o', 'label': 'Avg abs prediction error'}
    line_2 = {'x': atk_sizes, 'y': error_stdevs, 'style': 'r-o', 'label': 'Standard deviation'}
    plot_2d_lines([line_1, line_2], setup_2d_axes('# of attacked power nodes', 'Measured fraction', ylim=(0, 0.5)))


def plot_deaths_and_preds_by_atk_size(atk_sizes, avg_deaths, avg_preds):
    line_1 = {'x': atk_sizes, 'y': avg_deaths, 'style': 'g-o', 'label': 'Actual'}
    line_2 = {'x': atk_sizes, 'y': avg_preds, 'style': 'b-o', 'label': 'Predicted'}
    plot_2d_lines([line_1, line_2], setup_2d_axes('# of attacked power nodes', 'Average fraction of dead nodes'))


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


def run():
    # setup logging
    global logger
    log_conf_path = 'logging_base_conf.json'
    with open(log_conf_path, 'rt') as f:
        config = json.load(f)
    logging.config.dictConfig(config)
    logger = logging.getLogger(__name__)

    train_set_fpath = '/home/agostino/Documents/Sims/netw_a_0-100/0-100_union/equispaced_train_union.tsv'
    test_set_fpath = '/home/agostino/Documents/Sims/netw_a_0-100/0-100_union/test_union.tsv'

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
    X_col_names = ['p_atkd_a', 'p_tot_atkd_betw_c_i']

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

    model_name = 'DecisionTreeRegressor'
    model_kind = 'regression'

    # find columns
    train_X, train_y, train_info = load_dataset(train_set_fpath, X_col_names, y_col_name, info_col_names)

    # solve_and_test(train_set_fpath, test_set_fpath, X_col_names, y_col_name, info_col_names)
    model, transformers, transf_train_X, transf_X_col_names =\
        train_regr_model(train_X, train_y, X_col_names, False, False, False, model_name, None)

    if 'tree' in model_name.lower():
        with open("/home/agostino/whatever.dot", 'w') as f:
            tree.export_graphviz(model, feature_names=X_col_names, out_file=f)

    # save the model
    learned_stuff_fpath = '/home/agostino/Documents/Sims/results_compare/model_1000.pkl'
    learned_stuff = {'model': model, 'transformers': transformers}
    joblib.dump(learned_stuff, learned_stuff_fpath)

    atks_cnt_col = info_col_names.index('#atkd_a')
    predictor = lambda x: model.predict(x)  # function we use to predict the result

    datasets = [
        {
            'dataset_fpath': '/home/agostino/Documents/Simulations/test_mp_10/500_nodes_10_subnets_results/test_500_n_10_s.tsv',
            'relevant_atk_sizes': [3, 5, 10, 25, 50],
            'node_cnt_A': 500, 'name': '500 power nodes, 10 subnets'
        }, {
            'dataset_fpath': '/home/agostino/Documents/Simulations/test_mp_10/500_nodes_20_subnets_results/test_500_n_20_s.tsv',
            'relevant_atk_sizes': [3, 5, 10, 25, 50],
            'node_cnt_A': 500, 'name': '500 power nodes, 20 subnets'
        }, {
            'dataset_fpath': test_set_fpath,
            'relevant_atk_sizes': [5, 10, 20, 50, 100],
            'node_cnt_A': 1000, 'name': '1000 power nodes, 20 subnets'
        }, {
            'dataset_fpath': '/home/agostino/Documents/Simulations/test_mp_10/2000_nodes_20_subnets_results/test_2000_n_20_s.tsv',
            'relevant_atk_sizes': [10, 20, 40, 100, 200],
            'node_cnt_A': 2000, 'name': '2000 power nodes, 20 subnets'
        }, {
            'dataset_fpath': '/home/agostino/Documents/Simulations/test_mp_10/2000_nodes_40_subnets_results/test_2000_n_40_s.tsv',
            'relevant_atk_sizes': [10, 20, 40, 100, 200],
            'node_cnt_A': 2000, 'name': '2000 power nodes, 40 subnets'
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
        ax_x_vec, ax_x_label = plain_ds_X[:, 0], 'initial fraction of failed nodes'
        ax_y_vec, ax_y_label = plain_ds_X[:, 1], 'loss of centrality'
        ax_z_vec, ax_z_label = ds_y, 'actual resulting fraction of dead nodes'
        plot_3d_no_interpolate(ax_x_vec, ax_y_vec, ax_z_vec, ax_x_label, ax_y_label, ax_z_label, True, True, False)

        # show the original features on the x and y axis; show the predictions on the z axis
        ax_z_vec, ax_z_label = predictions, 'predicted fraction of dead nodes'
        plot_3d_no_interpolate(ax_x_vec, ax_y_vec, ax_z_vec, ax_x_label, ax_y_label, ax_z_label, True, False, True)

        plot_predictions_for_2d_dataset(plain_ds_X, ds_y, transformers, predictor)

        if model_kind == 'regression':
            check_prediction_bounds(plain_ds_X, ds_info, X_col_names, info_col_names, predictions, 0.0, True, 1.05, True)

            atk_sizes, costs, error_stdevs = calc_cost_by_atk_size(ds_X, ds_y, ds_info, atks_cnt_col, predictor)
            plot_cost_by_atk_size(atk_sizes, costs, error_stdevs)

            atk_sizes, avg_deaths, avg_preds = \
                avg_deaths_and_preds_by_atk_size(ds_X, ds_y, ds_info, atks_cnt_col, predictor)
            plot_deaths_and_preds_by_atk_size(atk_sizes, avg_deaths, avg_preds)

        else:
            atk_sizes, accuracies, f1_scores =\
                calc_scores_by_atk_size(ds_X, ds_y, ds_info, atks_cnt_col, predictor)
            atk_sizes, actual_cnt, pred_cnt =\
                count_actual_and_pred_labellings_by_atk_size(ds_X, ds_y, ds_info, atks_cnt_col, predictor, 'total')

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

    atkd_ps = [0.5, 1., 2., 5., 10.]  # TODO: calculate this
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

run()
