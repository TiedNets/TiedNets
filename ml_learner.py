import csv
import json
import logging.config
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import linear_model

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


# Applies the pre-calculated normalization parameters to X
# the mean value of each feature becomes 0 and the standard deviation becomes 1
# The proper term for this operation is "standardization", but in ML we have this frequent misnomer
# if this was an actual normalization, then the value of each feature of each example would be in [0,1]
def apply_normalization(X, mu, sigma):
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


# cost function is used in some plots to better visualize the prediction error
def calc_my_cost(X, y, predictor):
    predictions = predictor(X)
    errors = (predictions - y)
    cost = np.mean(np.abs(errors))  # average of absolute error

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


def plot_2D_line(x_vector, y_vector, xlabel, ylabel, line_label):
    ax = plt.axes()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.plot(x_vector, y_vector, 'b-o', label=line_label)
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
        predictions[i] = predictor(subset_X[data_row_num])  # predicted result of the simulation

    return sorted_atk_counts, results, predictions


# read a text file (e.g. tab separated values) and only load columns with the given names
def load_named_cols(input_fpath, col_names, file_header):
    col_nums = [file_header.index(col_name) for col_name in col_names]
    data_array = np.loadtxt(input_fpath, delimiter='\t', skiprows=1, usecols=col_nums)
    return data_array


def calc_cost_by_atk_size(data_X, data_y, data_info, atks_cnt_col, predictor):
    # pick unique values on the column with the number of attacks
    atk_sizes = np.sort(np.unique(data_info[:, atks_cnt_col]))
    var_costs = np.zeros(atk_sizes.size)
    # create the mask (row y/n) using the info matrix, but apply it to the data matrix
    # this way we can filter data rows using information not provided to the learning alg
    for i, atk_size in enumerate(atk_sizes):
        relevant_test_idx = data_info[:, atks_cnt_col] == atk_size
        relevant_data_X = data_X[relevant_test_idx, :]
        relevant_data_y = data_y[relevant_test_idx]
        var_costs[i] = calc_my_cost(relevant_data_X, relevant_data_y, predictor)
    return atk_sizes, var_costs


def train_and_test(train_set_fpath, test_set_fpath, X_col_names, y_col_name, info_col_names, alpha, num_iters):
    global logger
    X_col_names = list(X_col_names)  # make a local copy of the list
    info_col_names = list(info_col_names)

    # from each file we load load 3 sets of data, the examples (X), the labels (y) and another with related info
    # for each array we remember the correlation column name -> column number
    with open(train_set_fpath, 'r') as data_file:
        csvreader = csv.reader(data_file, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        header = csvreader.next()

    result_col = header.index(y_col_name)
    train_X = load_named_cols(train_set_fpath, X_col_names, header)
    train_y = np.loadtxt(train_set_fpath, delimiter='\t', skiprows=1, usecols=[result_col])
    train_info = load_named_cols(train_set_fpath, info_col_names, header)

    logger.debug('train_X (first 2 rows)\n{}'.format(train_X[range(2), :]))
    logger.debug('train_y (first 2 rows)\n{}'.format(train_y[range(2)]))
    logger.debug('train_info (first 2 rows)\n{}'.format(train_info[range(2), :]))

    test_X = load_named_cols(test_set_fpath, X_col_names, header)
    test_y = np.loadtxt(test_set_fpath, delimiter='\t', skiprows=1, usecols=[result_col])
    test_info = load_named_cols(test_set_fpath, info_col_names, header)

    logger.debug('test_X (first 2 rows)\n{}'.format(test_X[range(2), :]))
    logger.debug('test_y (first 2 rows)\n{}'.format(test_y[range(2)]))

    # feature standardization
    mu, sigma = find_mu_and_sigma(train_X, X_col_names)

    train_X = apply_normalization(train_X, mu, sigma)
    train_X = np.c_[np.ones(train_X.shape[0], dtype=train_X.dtype), train_X]  # add intercept (prefix a column of ones)
    logger.debug('train_X normalized with intercept (first 2 rows)\n{}'.format(train_X[range(2), :]))

    X_col_names.insert(0, '')  # prepend empty label to keep column names aligned

    test_X = apply_normalization(test_X, mu, sigma)
    test_X = np.c_[np.ones(test_X.shape[0], dtype=test_X.dtype), test_X]  # add intercept term
    logger.debug('test_X normalized with intercept (first 2 rows)\n{}'.format(test_X[range(2), :]))

    theta = np.zeros(train_X.shape[1], dtype=train_X.dtype)  # initialize theta to all zeros
    theta_history = gradient_descent_ext(train_X, train_y, theta, alpha, num_iters)
    predictor = lambda x: x.dot(theta)

    train_cost_history, test_cost_history = \
        calc_cost_histories(train_X, train_y, test_X, test_y, theta_history, calc_cost)
    plot_cost_history(train_cost_history, test_cost_history, 'Number of iterations', 'Value of cost function')

    # draw a graph showing the mean error for each #attacks
    atks_cnt_col = info_col_names.index('#atkd')
    atk_sizes, test_costs = calc_cost_by_atk_size(test_X, test_y, test_info, atks_cnt_col, predictor)
    logger.debug('atk_sizes = {}'.format(atk_sizes))
    plot_2D_line(atk_sizes, test_costs, '% of attacked nodes', 'Avg abs prediction error', 'test set')

    full_X = np.append(train_X, test_X, axis=0)
    full_y = np.append(train_y, test_y, axis=0)
    full_info = np.append(train_info, test_info, axis=0)

    for cur_seed in range(20):
        info_filter = {info_col_names.index('instance'): 0, info_col_names.index('seed'): cur_seed}
        indep_var_vals, results, predictions = \
            find_scenario_results_and_predictions(full_X, full_y, full_info, info_filter, atks_cnt_col, predictor)
        plot_scenario_performances(indep_var_vals, results, predictions, '# attacked nodes', '% dead nodes')


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
    atks_cnt_col = info_col_names.index('#atkd')
    atk_sizes, test_costs = calc_cost_by_atk_size(test_X, test_y, test_info, atks_cnt_col, predictor)
    logger.debug('atk_sizes = {}'.format(atk_sizes))
    plot_2D_line(atk_sizes, test_costs, '% of attacked nodes', 'Avg abs prediction error', 'test set')

    full_X = np.append(train_X, test_X, axis=0)
    full_y = np.append(train_y, test_y, axis=0)
    full_info = np.append(train_info, test_info, axis=0)

    for cur_seed in range(20):
        info_filter = {info_col_names.index('instance'): 0, info_col_names.index('seed'): cur_seed}
        indep_var_vals, results, predictions = \
            find_scenario_results_and_predictions(full_X, full_y, full_info, info_filter, atks_cnt_col, predictor)
        plot_scenario_performances(indep_var_vals, results, predictions, '# attacked nodes', '% dead nodes')


def train_and_test_sk(train_set_fpath, test_set_fpath, X_col_names, y_col_name, info_col_names):
    global logger
    X_col_names = list(X_col_names)  # make a local copy of the list
    info_col_names = list(info_col_names)

    # from each file we load load 3 sets of data, the examples (X), the labels (y) and another with related info
    # for each array we remember the correlation column name -> column number
    with open(train_set_fpath, 'r') as data_file:
        csvreader = csv.reader(data_file, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        header = csvreader.next()

    result_col = header.index(y_col_name)
    train_X = load_named_cols(train_set_fpath, X_col_names, header)
    train_y = np.loadtxt(train_set_fpath, delimiter='\t', skiprows=1, usecols=[result_col])
    train_info = load_named_cols(train_set_fpath, info_col_names, header)

    logger.debug('train_X (first 2 rows)\n{}'.format(train_X[range(2), :]))
    logger.debug('train_y (first 2 rows)\n{}'.format(train_y[range(2)]))
    logger.debug('train_info (first 2 rows)\n{}'.format(train_info[range(2), :]))

    test_X = load_named_cols(test_set_fpath, X_col_names, header)
    test_y = np.loadtxt(test_set_fpath, delimiter='\t', skiprows=1, usecols=[result_col])
    test_info = load_named_cols(test_set_fpath, info_col_names, header)

    logger.debug('test_X (first 2 rows)\n{}'.format(test_X[range(2), :]))
    logger.debug('test_y (first 2 rows)\n{}'.format(test_y[range(2)]))

    # feature standardization
    mu, sigma = find_mu_and_sigma(train_X, X_col_names)

    train_X = apply_normalization(train_X, mu, sigma)
    # no intercept term added, scikit-learn will do it
    logger.debug('train_X normalized without intercept (first 2 rows)\n{}'.format(train_X[range(2), :]))

    X_col_names.insert(0, '')  # prepend empty label to keep column names aligned

    test_X = apply_normalization(test_X, mu, sigma)
    # no intercept term added, scikit-learn will do it
    logger.debug('test_X normalized without intercept (first 2 rows)\n{}'.format(test_X[range(2), :]))

    # polynomial features/interactions
    poly = preprocessing.PolynomialFeatures(2)
    train_X = poly.fit_transform(train_X)
    test_X = poly.fit_transform(test_X)
    logger.debug('test_X with polynomial features (first 2 rows)\n{}'.format(test_X[range(2), :]))

    # apply regularized polynomial regression with built-in cross validation to choose the best value for the
    # hyperparameter alpha, basically it chooses the best regularization parameter by itself
    alphas = (0.1, 0.5, 1.0, 5.0, 10.0, 25.0, 50.0)
    clf = linear_model.RidgeCV(alphas)
    clf.fit(train_X, train_y)
    logger.debug('clf.alpha_ {}'.format(clf.alpha_))
    predictor = lambda x: clf.predict(x)  # function we use to predict the result

    # draw a graph showing the mean error for each #attacks
    atks_cnt_col = info_col_names.index('#atkd')
    atk_sizes, test_costs = calc_cost_by_atk_size(test_X, test_y, test_info, atks_cnt_col, predictor)
    logger.debug('atk_sizes = {}'.format(atk_sizes))
    plot_2D_line(atk_sizes, test_costs, '% of attacked nodes', 'Avg abs prediction error', 'test set')

    full_X = np.append(train_X, test_X, axis=0)
    full_y = np.append(train_y, test_y, axis=0)
    full_info = np.append(train_info, test_info, axis=0)

    for cur_seed in range(20):
        info_filter = {info_col_names.index('instance'): 0, info_col_names.index('seed'): cur_seed}
        indep_var_vals, results, predictions = \
            find_scenario_results_and_predictions(full_X, full_y, full_info, info_filter, atks_cnt_col, predictor)
        plot_scenario_performances(indep_var_vals, results, predictions, '# attacked nodes', '% dead nodes')


def run():
    # setup logging
    global logger
    log_conf_path = 'logging_base_conf.json'
    with open(log_conf_path, 'rt') as f:
        config = json.load(f)
    logging.config.dictConfig(config)
    logger = logging.getLogger(__name__)

    train_set_fpath = 'Data/train_0-8_0123a.tsv'
    test_set_fpath = 'Data/test_0-8_0123a.tsv'

    data_col_names = ['p_atkd', 'p_atkd_a', 'p_atkd_b',
                      'indeg_c_ab_q_1', 'indeg_c_ab_q_2', 'indeg_c_ab_q_3', 'indeg_c_ab_q_4', 'indeg_c_ab_q_5',
                      'indeg_c_i_q_1', 'indeg_c_i_q_2', 'indeg_c_i_q_3', 'indeg_c_i_q_4', 'indeg_c_i_q_5',
                      'rel_betw_c_q_1', 'rel_betw_c_q_2', 'rel_betw_c_q_3', 'rel_betw_c_q_4', 'rel_betw_c_q_5',
                      'p_atkd_gen', 'p_atkd_ts', 'p_atkd_ds', 'p_atkd_rel', 'p_atkd_cc']
    result_col_name = 'p_dead'
    info_col_names = ['instance', 'seed', '#atkd']

    alpha = 0.01  # the learning rate
    num_iters = 400  # the number of iterations

    train_and_test(train_set_fpath, test_set_fpath, data_col_names, result_col_name, info_col_names, alpha, num_iters)
    solve_and_test(train_set_fpath, test_set_fpath, data_col_names, result_col_name, info_col_names)
    train_and_test_sk(train_set_fpath, test_set_fpath, data_col_names, result_col_name, info_col_names)


run()
