import csv
from itertools import izip


# works for tsv files, make sure the two input files have the same number of lines
def add_cols_from_file(main_fpath, add_fpath, output_fpath):
    with open(main_fpath, 'r') as main_file, open(add_fpath, 'r') as add_file, open(output_fpath, 'w') as out_file:
        main_reader = csv.reader(main_file, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        add_reader = csv.reader(add_file, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        out_writer = csv.writer(out_file, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        main_header = main_reader.next()
        add_header = add_reader.next()
        out_header = main_header + add_header
        out_writer.writerow(out_header)
        for main_row, add_row in izip(main_reader, add_reader):
            out_row = main_row + add_row
            out_writer.writerow(out_row)


for i in range(0, 8):
    main_fpath = '/home/agostino/Documents/Simulations/test_mp/ml_stats_{}.tsv'.format(i)
    add_fpath = '/home/agostino/Documents/Simulations/test_mp/missing_col_{}.tsv'.format(i)
    output_fpath = '/home/agostino/Documents/Simulations/test_mp/ml_stats_fix_{}.tsv'.format(i)
    add_cols_from_file(main_fpath, add_fpath, output_fpath)
