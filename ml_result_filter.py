import os
import csv
from bisect import bisect
import random
import shared_functions as sf

__author__ = 'Agostino Sturaro'


def check_split_tolerance(total_item_cnt, chosen_item_cnt, desired_perc, perc_tolerance, where):
    actual_perc = sf.percent_of_part(chosen_item_cnt, total_item_cnt)
    perc_deviation = abs(actual_perc - desired_perc)
    if perc_deviation > perc_tolerance:
        print('Percentage deviation of {} for {}'.format(perc_deviation, where))


# assumes all files have a single header line and only keeps the header of the first file
def merge_files_with_headers(input_file_paths, output_file_path):
    first_file = True
    with open(output_file_path, 'w') as output_file:
        for input_fpath in input_file_paths:
            with open(input_fpath, 'r') as input_file:
                if first_file is True:
                    header_line = next(input_file)
                    output_file.write(header_line)
                    first_file = False
                else:
                    if next(input_file) != header_line:
                        raise RuntimeError('Header mismatch between {} and {}'.format(input_file_paths[0], input_fpath))
                for line in input_file:
                    output_file.write(line)

                # if there was no empty last line, add one
                if not line.endswith('\n') and not line.endswith('\r\n'):
                    output_file.write('\n')


# add a column containing the labels for the values in another column
# TODO: add a parameter to specify what type to cast to
def label_col_values(input_fpath, output_fpath, col_to_label, label_col_name, thresholds, labels):
    with open(input_fpath, 'r') as input_file, open(output_fpath, 'w') as output_file:
        csvreader = csv.reader(input_file, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        csvwriter = csv.writer(output_file, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        header = csvreader.next()
        if col_to_label not in header:
            raise ValueError('The column to label does not exist, check the value of col_to_label')
        if label_col_name in header:
            raise ValueError('The column to add already exist, check the value of label_col_name')
        if len(labels) != len(thresholds) + 1:
            raise ValueError('The number of labels must be equal to the number of thresholds + 1')

        if sorted(header) == header:
            ins_idx = bisect(header, label_col_name)
        else:
            ins_idx = header.index(col_to_label) + 1
        header.insert(ins_idx, label_col_name)
        csvwriter.writerow(header)
        col_num = header.index(col_to_label)

        # TODO: fix this code (labelling)
        for row in csvreader:
            col_val = float(row[col_num])  # type cast here
            label_num = 0

            for thresh in thresholds:
                if col_val >= thresh:
                    label_num += 1
                else:
                    break
            label = labels[label_num]
            row.insert(ins_idx, label)
            csvwriter.writerow(row)


# values must be a list of strings, because values are read as text
def remove_col_values(input_fpath, output_fpath, col_name, values):
    with open(input_fpath, 'r') as input_file, open(output_fpath, 'w') as output_file:
        csvreader = csv.reader(input_file, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        csvwriter = csv.writer(output_file, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        header = csvreader.next()
        csvwriter.writerow(header)
        col_num = header.index(col_name)

        for row in csvreader:
            col_value = row[col_num]
            if col_value not in values:
                csvwriter.writerow(row)


def filter_file_cols(input_fpath, output_fpath, wanted_col_names):
    with open(input_fpath, 'r') as input_file, open(output_fpath, 'w') as output_file:
        csvreader = csv.reader(input_file, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        csvwriter = csv.writer(output_file, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        header = csvreader.next()
        wanted_col_nums = []
        for col_name in wanted_col_names:
            col_num = header.index(col_name)
            wanted_col_nums.append(col_num)
        wanted_col_nums.sort()

        for row in csvreader:
            filtered_row = [row[col_num] for col_num in wanted_col_nums]
            csvwriter.writerow(filtered_row)


# keep only one row with for each value on a given column
def filter_duplicates_on_col(input_fpath, output_fpath, duplicates_col_name):
    with open(input_fpath, 'r') as input_file, open(output_fpath, 'w') as output_file:
        csvreader = csv.reader(input_file, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        csvwriter = csv.writer(output_file, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        header = csvreader.next()
        csvwriter.writerow(header)
        duplicates_col_num = header.index(duplicates_col_name)
        seen = set()
        duplicate_cnt = 0
        for row in csvreader:
            col_val = row[duplicates_col_num]
            if col_val not in seen:
                csvwriter.writerow(row)
                seen.add(col_val)
            else:
                duplicate_cnt += 1
                print('Found duplicate val {}'.format(col_val))
    print('Found {} duplicates'.format(duplicate_cnt))


# open the tsv file (set correct parameters)
# divide in training set and test set
#   divide examples on the value of a column
#   shuffle
# for each of them, pick a percentage
# throw away useless columns
# save

# TODO: offer an alternative split that does not slice examples first

# my_random = random.Random(130)
# my_random = random.Random(131)
# my_random = random.Random(132)
my_random = random.Random(130)

# input_fpath = '/home/agostino/Documents/Sims/netw_a_0-100/0-100_union/train_union.tsv'
# output_fpath = '/home/agostino/Documents/Sims/netw_a_0-100/0-100_union/equispaced_train_union.tsv'
# col_name = '#atkd_a'
# vals_to_remove = ['1', '2', '3', '4', '6', '7', '8', '9']
# remove_col_values(input_fpath, output_fpath, col_name, vals_to_remove)
# exit(0)


# input_folder = '/home/agostino/Documents/Simulations/test_mp_10/500_nodes_10_subnets_results/'
# batches = [0, 1]
# input_folder = '/home/agostino/Documents/Simulations/test_mp_10/500_nodes_20_subnets_results/'
# batches = [2, 3]
# input_folder = '/home/agostino/Documents/Simulations/test_mp_10/2000_nodes_20_subnets_results/'
# batches = [4, 5]
# input_folder = '/home/agostino/Documents/Simulations/test_mp_10/2000_nodes_40_subnets_results/'
# batches = [0, 1, 2]

# input_folder = '/home/agostino/Documents/Simulations/test_mp_11/'
# batches = [0, 1]
# input_folder = '/home/agostino/Documents/Simulations/test_mp_11/'
# batches = [2, 3]

# input_folder = '/home/agostino/Documents/Simulations/test_mp_12/'
# batches = [0, 1, 2, 3, 4, 5, 6, 7]
# input_folder = '/home/agostino/Documents/Simulations/test_mp_13/'
# batches = [0]
# input_folder = '/home/agostino/Documents/Simulations/test_mp_13/'
# batches = [1]
# input_folder = '/home/agostino/Documents/Simulations/test_mp_14/'
# batches = [0, 1, 2, 3]
# input_folder = '/home/agostino/Documents/Simulations/test_mp_14/'
# batches = [4, 5, 6, 7]

# input_folder = '/home/agostino/Documents/Simulations/test_mp_12b/'
# batches = [6, 7]
# input_folder = '/home/agostino/Documents/Simulations/test_mp_12b/'
# batches = [6, 7]
# input_folder = '/home/agostino/Documents/Simulations/test_mp_14b/'
# batches = [2, 3, 4, 5, 6]
input_folder = '/home/agostino/Documents/Simulations/test_mp_12c/'
batches = range(0, 8)

input_file_paths = []
for batch_num in batches:
    input_file_paths.append(os.path.join(input_folder, 'ml_stats_{}.tsv'.format(batch_num)))

merged_fname = 'merged.tsv'
merged_fpath = os.path.join(input_folder, merged_fname)
merge_files_with_headers(input_file_paths, merged_fpath)

# col_to_label = 'p_dead'
# label_col_name = 'dead_lvl'
# extra_cols_fname = 'ext_2000_n_40_s.tsv'
# extra_cols_fpath = os.path.join(input_folder, extra_cols_fname)
# label_col_values(merged_fpath, extra_cols_fpath, col_to_label, label_col_name, [0.3], ['low', 'total'])
#
in_fpath = merged_fpath
# in_fpath = extra_cols_fpath
# base_out_fname = '500_n_10_s.tsv'
# base_out_fname = '1000_n_20_s.tsv'
# base_out_fname = '2000_n_40_s.tsv'
base_out_fname = '1000_n_20_s.tsv'
train_fpath = os.path.join(input_folder, 'train_' + base_out_fname)
cv_fpath = os.path.join(input_folder, 'cv_' + base_out_fname)
test_fpath = os.path.join(input_folder, 'test_' + base_out_fname)
#
# with open(extra_cols_fpath, 'r') as extra_file:
#     rdr = csv.DictReader(extra_file, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
#     row_cnt = 0
#     for row in rdr:
#         print('p_dead = {}, dead_lvl = {}'.format(row['p_dead'], row['dead_lvl']))
#         row_cnt += 1
#         if row_cnt > 10:
#             break

# how to subdivide the set of examples
p_train = 0.8  # training set
p_cv = 0.0  # cross validation/home/agostino/Documents/train_0-7_union.tsv
p_test = 0.2  # test set
p_tolerance = 0.05  # maximum acceptable deviation from intended percentage

# subdivide examples based on the value they have in this column
splitting_col_name = '#atkd_a'
examples_by_val = {}
tot_ex_cnt = 0

with open(in_fpath, 'r') as ml_file:
    csvreader = csv.reader(ml_file, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
    header = csvreader.next()
    splitting_col_num = header.index(splitting_col_name)
    for example in csvreader:
        splitting_val = example[splitting_col_num]
        if splitting_val not in examples_by_val:
            examples_by_val[splitting_val] = []
        examples_by_val[splitting_val].append(example)
        tot_ex_cnt += 1

    for splitting_val in examples_by_val:
        my_random.shuffle(examples_by_val[splitting_val])

train_ex_cnt = cv_ex_cnt = test_ex_cnt = 0

with open(train_fpath, 'w') as train_file, open(cv_fpath, 'w') as cv_file, open(test_fpath, 'w') as test_file:
    train_writer = csv.writer(train_file, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
    train_writer.writerow(header)
    cv_writer = csv.writer(cv_file, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
    cv_writer.writerow(header)
    test_writer = csv.writer(test_file, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
    test_writer.writerow(header)

    for splitting_val in sorted(examples_by_val.keys(), key=sf.natural_sort_key):
        all_sets = sf.percentage_split(examples_by_val[splitting_val], [p_train, p_cv, p_test])

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


check_split_tolerance(tot_ex_cnt, train_ex_cnt, p_train, p_tolerance, 'train')
check_split_tolerance(tot_ex_cnt, cv_ex_cnt, p_cv, p_tolerance, 'cv')
check_split_tolerance(tot_ex_cnt, test_ex_cnt, p_test, p_tolerance, 'test')
