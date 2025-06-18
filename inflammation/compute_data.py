"""Module containing mechanism for calculating standard deviation between datasets."""

import glob
import os
import numpy as np

from inflammation import models, views


class CSVDataSource:
    """Class to create an instance for CSV data files"""
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_inflammation_data(self):
        """Gets all the inflammation data from CSV files within a directory"""
        data_file_paths = glob.glob(os.path.join(self.data_dir, 'inflammation*.csv'))
        if len(data_file_paths) == 0:
            raise ValueError(f"No inflammation data CSV files found in path {self.data_dir}")
        data = map(models.load_csv, data_file_paths)

        return list(data)


class JSONDataSource:
    """Class to create an instance for JSON data files"""
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_inflammation_data(self):
        """Gets all the inflammation data from JSON files within a directory"""
        data_file_paths = glob.glob(os.path.join(self.data_dir, 'inflammation*.json'))
        if len(data_file_paths) == 0:
            raise ValueError(f"No inflammation data JSON files found in path {self.data_dir}")
        data = map(models.load_json, data_file_paths)

        return list(data)


def compute_standard_deviation_by_day(data):
    """Calculates the standard deviation by day between datasets."""
    means_by_day = map(models.daily_mean, data)
    means_by_day_matrix = np.stack(list(means_by_day))

    daily_standard_deviation = np.std(means_by_day_matrix, axis=0)

    return daily_standard_deviation


def analyse_data(dataset):
    """Calculates the standard deviation by day between datasets.
    Gets all the inflammation data from CSV files within a directory, works out the mean
    inflammation value for each day across all datasets, then visualises the
    standard deviation of these means on a graph."""
    daily_standard_deviation = compute_standard_deviation_by_day(dataset)

    return daily_standard_deviation


def plot_data(std_dev):
    """Plot standard deviation data"""
    graph_data = {
        'standard deviation by day': std_dev,
    }

    views.visualize(graph_data)
