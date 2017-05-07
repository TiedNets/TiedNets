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


def calc_cost_scikit(X, y, predictor):
    m = y.size  # number of examples given
    predictions = predictor(X)
    errors = (predictions - y)
    cost = (1.0 / (2 * m)) * errors.T.dot(errors)

    return cost


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


def plot_cost_history(train_cost_history, test_cost_history, xlabel, ylabel):
    ax = plt.axes()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    num_iters = train_cost_history.size
    plt.plot(range(num_iters), train_cost_history, label='training set')
    plt.plot(range(num_iters), test_cost_history, label='test set')

    plt.legend(fontsize=10)
    plt.tight_layout()  # make sure everything is showing
    plt.show()


def plot_2D_line(x, y, xlabel, ylabel, line_label, std_devs=None):
    ax = plt.axes()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.plot(x, y, 'b-o', label=line_label)
    if std_devs is not None:
        plt.errorbar(x, y, std_devs)
    plt.legend(fontsize=10)
    plt.tight_layout()  # make sure everything is showing
    plt.show()


# lines is a list of dictionaries {x, y, line_label}
def plot_2D_lines(lines, xlabel, ylabel, xticks, xlim=None, ylim=None):
    ax = plt.axes()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(xticks)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    for line in lines:
        plt.plot(line['x'], line['y'], line['style'], label=line['label'])
    plt.legend(fontsize=10)
    plt.tight_layout()  # make sure everything is showing
    plt.show()


def plot_scenario_performances(x_values, results, predictions, xlabel, ylabel):
    ax = plt.axes()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    plt.plot(x_values, results, 'g-o', label='results')
    plt.plot(x_values, predictions, 'r-o', label='predictions')

    plt.legend(fontsize=10)
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


# Takes 3 vectors and interpolates them to return a uniformly spaced grid, made by three 2d ndarrays
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


# X must have 2 features, so we can plot a 3D function after calculating the predictions
def plot_learned_fun_on_set(X, X_col_names, col_name_1, col_name_2, predictor):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    zs = predictor(X)
    col_1 = X[:, X_col_names.index(col_name_1)]
    col_2 = X[:, X_col_names.index(col_name_2)]

    surf = ax.plot_trisurf(col_1, col_2, zs, cmap=cm.jet, linewidth=0)
    fig.colorbar(surf)
    ax.scatter(col_1, col_2, zs)

    plt.show()


def plot_3d_scatter(ax_x_vec, ax_y_vec, ax_z_vec, ax_x_label, ax_y_label, ax_z_label, project=False):
    if len(ax_x_vec.shape) != 1 or len(ax_y_vec.shape) != 1:
        raise ValueError('The parameters "xs" and "ys" must be 1D ndarrays, '
                         'xs.shape={}, ys.shape={}'.format(ax_x_vec.shape, ax_y_vec.shape))

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel(ax_x_label)
    ax.set_ylabel(ax_y_label)
    ax.set_zlabel(ax_z_label)

    min_x, max_x = np.min(ax_x_vec), np.max(ax_x_vec)
    min_y, max_y = np.min(ax_y_vec), np.max(ax_y_vec)
    min_z, max_z = np.min(ax_z_vec), np.max(ax_z_vec)

    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)
    ax.set_zlim(min_z, max_z)

    if project is True:
        ax.plot(ax_y_vec, ax_z_vec, 'g+', zdir='x', zs=min_x)
        ax.plot(ax_x_vec, ax_z_vec, 'r+', zdir='y', zs=max_y)
        ax.plot(ax_x_vec, ax_y_vec, 'k+', zdir='z', zs=min_z)

    ax.scatter(ax_x_vec, ax_y_vec, ax_z_vec)

    plt.show()


def plot_3d_no_interpolate(ax_x_vec, ax_y_vec, ax_z_vec, ax_x_label, ax_y_label, ax_z_label, scatter=False):
    if len(ax_x_vec.shape) != 1 or len(ax_y_vec.shape) != 1:
        raise ValueError('The parameters "xs" and "ys" must be 1D ndarrays, '
                         'xs.shape={}, ys.shape={}'.format(ax_x_vec.shape, ax_y_vec.shape))

    min_x, max_x = np.min(ax_x_vec), np.max(ax_x_vec)
    min_y, max_y = np.min(ax_y_vec), np.max(ax_y_vec)
    min_z, max_z = np.min(ax_z_vec), np.max(ax_z_vec)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel(ax_x_label)
    ax.set_ylabel(ax_y_label)
    ax.set_zlabel(ax_z_label)
    surf = ax.plot_trisurf(ax_x_vec, ax_y_vec, ax_z_vec, cmap=cm.jet, linewidth=0, alpha=0.8)
    fig.colorbar(surf)

    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)
    ax.set_zlim(min_z, max_z)

    if scatter is True:
        ax.scatter(ax_x_vec, ax_y_vec, ax_z_vec)

    plt.show()


def plot_interpolated_3d(x_grid, y_grid, z_grid, ax_x_label, ax_y_label, ax_z_label):
    offset_x = np.min(x_grid)
    offset_y = np.max(y_grid)
    offset_z = np.min(z_grid)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel(ax_x_label)
    ax.set_ylabel(ax_y_label)
    ax.set_zlabel(ax_z_label)
    surf = ax.plot_surface(x_grid, y_grid, z_grid, cmap=cm.jet, linewidth=0)
    fig.colorbar(surf)
    ax.contourf(x_grid, y_grid, z_grid, zdir='x', offset=offset_x, cmap=cm.jet)
    ax.contourf(x_grid, y_grid, z_grid, zdir='y', offset=offset_y, cmap=cm.jet)
    ax.contourf(x_grid, y_grid, z_grid, zdir='z', offset=offset_z, cmap=cm.jet)

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
        # var_costs[i] = calc_cost_scikit(relevant_data_X, relevant_data_y, predictor)
    return atk_sizes, var_costs, error_std_devs


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
    plot_2D_line(atk_sizes, test_costs, '# of attacked power nodes', 'Avg abs prediction error', 'test set')

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


def train_model(train_X, train_y, train_info, X_col_names, y_col_name, info_col_names):
    global logger
    # make a local copy of the objects we received as parameters and might change
    train_X = train_X.copy()  # make a local copy of the array
    X_col_names = list(X_col_names)  # make a local copy of the list

    # baseline feature selection, round 1
    base_feature_cnt = train_X.shape[1]
    vt_sel = VarianceThreshold()
    train_X = vt_sel.fit_transform(train_X)
    vt_feature_mask = vt_sel.get_support()
    X_col_names = [item for item_num, item in enumerate(X_col_names) if vt_feature_mask[item_num]]
    sel_feature_cnt = train_X.shape[1]
    logger.debug('VarianceThreshold removed {} features'.format(base_feature_cnt - sel_feature_cnt))

    # polynomial features/interactions, allow to learn a more complex prediction function
    poly = preprocessing.PolynomialFeatures(4, interaction_only=False)
    train_X = poly.fit_transform(train_X)
    X_col_names = poly.get_feature_names(X_col_names)
    logger.debug('Polynomial X_col_names = {}'.format(X_col_names))
    logger.debug('train_X with polynomial features (first 2 rows)\n{}'.format(train_X[range(2), :]))

    # apply a standardization step
    scaler = preprocessing.StandardScaler()
    train_X = scaler.fit_transform(train_X)

    # plain linear regression can work as polynomial regression too
    clf = linear_model.LinearRegression()

    # # other polynomial regression models with built-in regularization and cross validation
    # # regularization is used to lower the weight given to less important and useless features
    # # cross validation is used to tune the hyperparameters, like alpha
    # alphas = (0.1, 0.5, 1.0, 5.0, 10.0, 25.0, 50.0)
    # clf = linear_model.RidgeCV(alphas)
    # clf = linear_model.LassoCV(max_iter=100000)
    # clf = linear_model.ElasticNetCV(max_iter=40000)
    # clf = linear_model.ElasticNetCV(normalize=True, max_iter=100000)

    # # model selection by feature selection, two different options
    # selector = RFE(clf, step=1, n_features_to_select=20)
    # selector = RFECV(clf, step=1, cv=5, scoring='neg_mean_absolute_error')

    # # model selection by feature selection, third option
    # selector = SelectFromModel(clf)
    # selector.fit(train_X, train_y)
    # train_X, selector = iterate_sfm_transform(selector, train_X, 20, 100, 0.01, 0.01)

    # sel_feature_mask = selector.get_support()
    # sel_features_names = [item for item_num, item in enumerate(X_col_names) if sel_feature_mask[item_num]]
    # logger.debug('sel_features_names = {}'.format(sel_features_names))
    # logger.debug('after transform train_X.shape[1] = {}'.format(train_X.shape[1]))

    clf.fit(train_X, train_y)
    logger.debug('learning done')

    # objects for preprocessing and feature selection, in the order they were used
    transformers = [vt_sel, poly, scaler]

    return clf, transformers, train_X, X_col_names


def check_prediction_bounds(plain_X, info, X_col_names, info_col_names, predictions, lb, include_lb, ub, include_ub):
    if plain_X.shape[1] != len(X_col_names):
        raise ValueError('plain_X and X_col_names should have the same number of columns')
    if info.shape[1] != len(info_col_names):
        raise ValueError('info and info_col_names should have the same number of columns')

    if include_lb is True:
        below_mask = np.less(predictions, lb)  # only strictly below the bound
    else:
        below_mask = np.less_equal(predictions, lb)
    if True in below_mask:
        logger.info('Prediction below lower bound for the following {} examples'.format(np.count_nonzero(below_mask)))
        logger.info('{}\n{}'.format(X_col_names, plain_X[below_mask]))
        logger.info('{}\n{}'.format(info_col_names, info[below_mask]))

    if include_ub is True:
        over_mask = np.greater(predictions, ub)
    else:
        over_mask = np.greater_equal(predictions, ub)
    if True in over_mask:
        logger.info('Prediction over upper bound for the following {} examples'.format(np.count_nonzero(over_mask)))
        logger.info('{}\n{}'.format(X_col_names, plain_X[over_mask]))
        logger.info('{}\n{}'.format(info_col_names, info[over_mask]))


# TODO: make this a data visualization that only accepts 1 dataset (not both train and test)
# this needs everything, untransformed data and transformed data
def predict_and_plot(full_X, full_y, full_info, transformers, predictor,
                     plain_full_X, plain_full_y, plain_X_col_names, plain_full_info, info_col_names):
    global logger
    full_X = full_X.copy()

    for transformer in transformers:
        full_X = transformer.transform(full_X)

    # we want the first feature on the x axis, the second feature on the y axis, and the results on the z axis
    ax_x_vec, ax_x_label = plain_full_X[:, 0], 'initial fraction of failed nodes'
    ax_y_vec, ax_y_label = plain_full_X[:, 1], 'loss of centrality'
    ax_z_vec, ax_z_label = plain_full_y, 'resulting fraction of dead nodes'
    plot_3d_scatter(ax_x_vec, ax_y_vec, ax_z_vec, ax_x_label, ax_y_label, ax_z_label, True)

    full_prediction = predictor(full_X)

    # show the original features on the x and y axis, but show the predictions on the transformed data
    ax_z_vec, ax_z_label = full_prediction, 'predicted fraction of dead nodes'
    plot_3d_no_interpolate(ax_x_vec, ax_y_vec, ax_z_vec, ax_x_label, ax_y_label, ax_z_label, True)

    # create a dataset with uniformly spaced points
    x_grid, y_grid = make_uniform_grid_xy(plain_full_X[:, 0], plain_full_X[:, 1], 100, 100)
    # make sure the columns are in the right order when you build the model
    uniform_X = np.c_[x_grid.ravel(), y_grid.ravel()]
    uniform_X = apply_transforms(transformers, uniform_X)
    uniform_X_predictions = predictor(uniform_X)
    z_grid = np.reshape(uniform_X_predictions, x_grid.shape)
    ax_z_label = 'predicted fraction of dead nodes'
    plot_interpolated_3d(x_grid, y_grid, z_grid, ax_x_label, ax_y_label, ax_z_label)


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


def plot_cost_by_atk_size(X, y, info, atks_cnt_col, predictor):
    global logger

    # draw a graph showing the mean error for each #attacks, and the standard deviation of this error
    atk_sizes, costs, error_stdevs = calc_cost_by_atk_size(X, y, info, atks_cnt_col, predictor)
    logger.debug('atk_sizes = {}'.format(atk_sizes))
    line_1 = {'x': atk_sizes, 'y': costs, 'style': 'b-o', 'label': 'Avg abs prediction error'}
    line_2 = {'x': atk_sizes, 'y': error_stdevs, 'style': 'r-o', 'label': 'Standard deviation'}
    # plot_2D_lines([line_1, line_2], '# of attacked power nodes', 'Measured fraction', xlim=None, ylim=[0, 0.5])
    #
    # plot_2D_line(atk_sizes, costs, '# of attacked power nodes', 'Measured fraction', 'Avg abs prediction error',
    #              error_stdevs)

    return atk_sizes, costs, error_stdevs


def plot_deaths_and_preds_by_atk_size(X, y, info, atks_cnt_col, predictor):
    atk_sizes, avg_deaths, avg_preds = \
        avg_deaths_and_preds_by_atk_size(X, y, info, atks_cnt_col, predictor)
    line_1 = {'x': atk_sizes, 'y': avg_deaths, 'style': 'g-o', 'label': 'Actual'}
    line_2 = {'x': atk_sizes, 'y': avg_preds, 'style': 'b-o', 'label': 'Predicted'}
    # plot_2D_lines([line_1, line_2], '# of attacked power nodes', 'Average fraction of dead nodes')

    return atk_sizes, avg_deaths, avg_preds


def run():
    # setup logging
    global logger
    log_conf_path = 'logging_base_conf.json'
    with open(log_conf_path, 'rt') as f:
        config = json.load(f)
    logging.config.dictConfig(config)
    logger = logging.getLogger(__name__)

    # train_set_fpath = 'Data/train_0-7_union.tsv'
    # test_set_fpath = 'Data/test_0-7_union.tsv'
    # train_set_fpath = '/home/agostino/Documents/Sims/20 nets 20170204/15 to 100 a/train_0-4_union_a.tsv'
    # test_set_fpath = '/home/agostino/Documents/Sims/20 nets 20170204/15 to 100 a/test_0-4_union_a.tsv'
    # train_set_fpath = '/home/agostino/Documents/Sims/netw_a_0-100/0-100_union/train_union.tsv'
    # test_set_fpath = '/home/agostino/Documents/Sims/netw_a_0-100/0-100_union/test_union.tsv'
    # train_set_fpath = '/home/agostino/Documents/Simulations/test_mp_10/500_nodes_10_subnets_results/train_500_n_10_s.tsv'
    # test_set_fpath = '/home/agostino/Documents/Simulations/test_mp_10/500_nodes_10_subnets_results/test_500_n_10_s.tsv'
    train_set_fpath = '/home/agostino/Documents/Simulations/test_mp_10/500_nodes_20_subnets_results/train_500_n_20_s.tsv'
    test_set_fpath = '/home/agostino/Documents/Simulations/test_mp_10/500_nodes_20_subnets_results/test_500_n_20_s.tsv'

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
    # X_col_names = ['p_atkd_ds', 'p_atkd_ts',
    #                   'p_q_4_atkd_betw_c_ab', 'p_q_4_atkd_betw_c_i',
    #                   'p_q_4_atkd_indeg_c_i', 'p_q_4_atkd_ts_betw_c',
    #                   'p_q_5_atkd_betw_c_ab', 'p_q_5_atkd_betw_c_i',
    #                   'p_q_5_atkd_indeg_c_i', 'p_q_5_atkd_ts_betw_c',
    #                   'p_tot_atkd_betw_c_ab', 'p_tot_atkd_betw_c_i',
    #                   'p_tot_atkd_indeg_c_i', 'p_tot_atkd_ts_betw_c'
    #
    # X_col_names = ['p_tot_atkd_betw_c_i', 'p_atkd_cc']
    X_col_names = ['p_atkd_a', 'p_tot_atkd_betw_c_i']
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
    info_col_names = ['instance', 'seed', '#atkd_a']

    train_X, train_y, train_info = load_dataset(train_set_fpath, X_col_names, y_col_name, info_col_names)

    # solve_and_test(train_set_fpath, test_set_fpath, X_col_names, y_col_name, info_col_names)
    model, transformers, transf_train_X, transf_X_col_names =\
        train_model(train_X, train_y, train_info, X_col_names, y_col_name, info_col_names)

    # save the model
    learned_stuff_fpath = '/home/agostino/Documents/Sims/results_compare/model_1000.pkl'
    learned_stuff = {'model': model, 'transformers': transformers}
    joblib.dump(learned_stuff, learned_stuff_fpath)

    atks_cnt_col = info_col_names.index('#atkd_a')
    predictor = lambda x: model.predict(x)  # function we use to predict the result

    datasets = [
        {
            'dataset_fpath': '/home/agostino/Documents/Simulations/test_mp_10/500_nodes_10_subnets_results/500_n_10_s.tsv',
            'relevant_atk_sizes': [3, 5, 10, 25, 50],
            'node_cnt_A': 500, 'name': '500 power nodes, 10 power subnets'
        }, {
            'dataset_fpath': '/home/agostino/Documents/Simulations/test_mp_10/500_nodes_20_subnets_results/500_n_20_s.tsv',
            'relevant_atk_sizes': [3, 5, 10, 25, 50],
            'node_cnt_A': 500, 'name': '500 power nodes, 20 power subnets'
        }, {
            'dataset_fpath': '/home/agostino/Documents/Sims/netw_a_0-100/0-100_union/train_union.tsv',
            'relevant_atk_sizes': [5, 10, 20, 50, 100],
            'node_cnt_A': 1000, 'name': '1000 power nodes, 20 power subnets'
        }, {
            'dataset_fpath': '/home/agostino/Documents/Simulations/test_mp_10/2000_nodes_20_subnets_results/2000_n_20_s.tsv',
            'relevant_atk_sizes': [10, 20, 40, 100, 200],
            'node_cnt_A': 2000, 'name': '2000 power nodes, 20 power subnets'
        }, {
            'dataset_fpath': '/home/agostino/Documents/Simulations/test_mp_10/2000_nodes_40_subnets_results/2000_n_40_s.tsv',
            'relevant_atk_sizes': [10, 20, 40, 100, 200],
            'node_cnt_A': 2000, 'name': '2000 power nodes, 40 power subnets'
        }
    ]

    for dataset in datasets:
        plain_ds_X, ds_y, ds_info = load_dataset(dataset['dataset_fpath'], X_col_names, y_col_name, info_col_names)
        ds_X = plain_ds_X.copy()
        for transformer in transformers:
            logger.debug('Applying {}.transform'.format(type(transformer).__name__))
            ds_X = transformer.transform(ds_X)

        predictions = predictor(ds_X)
        check_prediction_bounds(plain_ds_X, ds_info, X_col_names, info_col_names, predictions, 0.0, True, 1.05, True)

        atk_sizes, costs, error_stdevs = plot_cost_by_atk_size(ds_X, ds_y, ds_info, atks_cnt_col, predictor)
        atk_sizes, avg_deaths, avg_preds = plot_deaths_and_preds_by_atk_size(ds_X, ds_y, ds_info, atks_cnt_col, predictor)

        # only keep the data we want
        relevant_idx = []
        relevant_atk_sizes = dataset['relevant_atk_sizes']
        for i, atk_size in enumerate(atk_sizes):
            if atk_size in relevant_atk_sizes:
                relevant_idx.append(i)

        atk_sizes = [atk_sizes[i] for i in relevant_idx]
        costs = [costs[i] for i in relevant_idx]
        error_stdevs = [error_stdevs[i] for i in relevant_idx]
        avg_deaths = [avg_deaths[i] for i in relevant_idx]
        avg_preds = [avg_preds[i] for i in relevant_idx]

        dataset['results'] = {'atk_sizes': atk_sizes, 'costs': costs, 'error_stdevs': error_stdevs,
                              'avg_deaths': avg_deaths, 'avg_preds': avg_preds}

        # plot_all_scenario_performances(ds_X, ds_y, ds_info, info_col_names, predictor, 3, 2)
        # TODO: call the 3d visualization function here

    atkd_ps = [0.5, 1., 2., 5., 10.]  # TODO: calculate this
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    markers = ['o', '^', 's', '*', 'x', '+', 'd']
    styles = ['b-o', 'g-^', 'r-s', 'c-*', 'm-x', 'y-+']
    lines_to_plot = []
    for i, dataset in enumerate(datasets):
        line = {'x': atkd_ps, 'y': dataset['results']['avg_deaths'], 'style': styles[i], 'label': dataset['name']}
        lines_to_plot.append(line)
    plot_2D_lines(lines_to_plot, '% of attacked power nodes', 'Actual avg dead fraction (pow+tel)', atkd_ps)

    lines_to_plot = []
    for i, dataset in enumerate(datasets):
        line = {'x': atkd_ps, 'y': dataset['results']['avg_preds'], 'style': styles[i], 'label': dataset['name']}
        lines_to_plot.append(line)
    plot_2D_lines(lines_to_plot, '% of attacked power nodes', 'Predicted avg dead fraction (pow+tel)', atkd_ps)

    # use all_results to plot the graph
    # need atk_sizes, avg_deaths, avg_preds from plot_deaths_and_preds_by_atk_size
    # and somehow error_stdevs from plot_cost_by_atk_size

run()
