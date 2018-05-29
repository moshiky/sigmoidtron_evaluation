
import csv
import random


class DatasetUtils(object):

    @staticmethod
    def smooth_series(series, smoothing_level):
        # init indexes
        last_change_index = 0
        changes_found = 0

        # iterate series
        for i in range(len(series)):
            if series[i] > series[last_change_index]:
                changes_found += 1
                if changes_found == smoothing_level:
                    changes_found = 0
                    val_diff = series[i] - series[last_change_index]
                    step_size = float(val_diff) / (i - last_change_index)
                    for j in range(1, i - last_change_index):
                        series[last_change_index + j] = series[last_change_index] + j * step_size
                    last_change_index = i

    @staticmethod
    def parse_csv(file_path, should_shuffle, min_record_length):
        # read file
        with open(file_path, 'rt') as file_handle:
            reader = csv.reader(file_handle)
            file_lines = list(reader)

        record_list = [
            [float(line_value) for line_value in line]
            for line in file_lines if len(line) > min_record_length
        ]
        if should_shuffle:
            random.shuffle(record_list)

        return record_list
