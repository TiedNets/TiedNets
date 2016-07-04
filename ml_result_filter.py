import csv
import random
import shared_functions as sf


def check_split_tolerance(total_item_cnt, chosen_item_cnt, desired_perc, perc_tolerance):
    actual_perc = sf.percent_of_part(chosen_item_cnt, total_item_cnt)
    perc_deviation = abs(actual_perc - desired_perc)
    if perc_deviation > perc_tolerance:
        print('Percentage deviation of {}'.format(perc_deviation))


# open the tsv file (set correct parameters)
# divide in training set and test set
#   divide examples on the value of a column
#   shuffle
# for each of them, pick a percentage
# throw away useless columns
# save

# TODO: offer option to keep or strip header in files (currently stripping)
# TODO: offer an alternative split that does not slice examples first

my_random = random.Random(128)
in_fpath = 'C:/Users/Agostino/Documents/OctaveProjects/TiedNetsLearner/Data/atks_on_ab0123_2-26_401s_26-51_201s.txt'
train_fpath = 'C:/Users/Agostino/Desktop/train_0123ab.tsv'
cv_fpath = 'C:/Users/Agostino/Desktop/cv_0123ab.tsv'
test_fpath = 'C:/Users/Agostino/Desktop/test_0123ab.tsv'

# how to subdivide the set of examples
p_train = 0.8  # training set
p_cv = 0.0  # cross validation
p_test = 0.2  # test set
p_tolerance = 0.05  # maximum acceptable deviation from intended percentage

col_name = 'p_atkd'
examples_by_val = {}
tot_ex_cnt = 0

with open(in_fpath, 'r') as ml_file:
    csvreader = csv.reader(ml_file, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
    header = csvreader.next()
    col_num = header.index(col_name)
    for example in csvreader:
        col_val = example[col_num]
        if col_val not in examples_by_val:
            examples_by_val[col_val] = []
        examples_by_val[col_val].append(example)
        tot_ex_cnt += 1

    for col_val in examples_by_val:
        my_random.shuffle(examples_by_val[col_val])

train_ex_cnt = cv_ex_cnt = test_ex_cnt = 0

with open(train_fpath, 'wb') as train_file, open(cv_fpath, 'wb') as cv_file, open(test_fpath, 'wb') as test_file:
    train_writer = csv.writer(train_file, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
    cv_writer = csv.writer(cv_file, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
    test_writer = csv.writer(test_file, delimiter='\t', quoting=csv.QUOTE_MINIMAL)

    for col_val in examples_by_val:
        all_sets = sf.percentage_split(examples_by_val[col_val], [p_train, p_cv, p_test])

        train_set = all_sets[0]
        train_ex_cnt += len(train_set)
        for example in train_set:
            train_writer.writerow(example)

        cv_set = all_sets[1]
        cv_ex_cnt += len(cv_set)
        for example in cv_set:
            cv_writer.writerow(example)

        test_set = all_sets[2]
        test_ex_cnt += len(test_set)
        for example in test_set:
            test_writer.writerow(example)


check_split_tolerance(tot_ex_cnt, train_ex_cnt, p_train, p_tolerance)
check_split_tolerance(tot_ex_cnt, cv_ex_cnt, p_cv, p_tolerance)
check_split_tolerance(tot_ex_cnt, test_ex_cnt, p_test, p_tolerance)
