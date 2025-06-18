"""Module containing models representing patients and their data.

The Model layer is responsible for the 'business logic' part of the software.

Patients' data is held in an inflammation table (2D array) where each row contains 
inflammation data for a single patient taken over a number of days 
and each column represents a single day across all patients.
"""

import json
import glob
import os
import numpy as np


def load_csv(filename):
    """Load a Numpy array from a CSV

    :param filename: Filename of CSV to load
    """
    return np.loadtxt(fname=filename, delimiter=',')


def load_json(filename):
    """Load a Numpy array from a JSON

    :param filename: Filename of JSON to load
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    return np.array(data)


def daily_mean(data):
    """Calculate the daily mean of a 2d inflammation data array."""
    return np.mean(data, axis=0)


def daily_max(data):
    """Calculate the daily max of a 2d inflammation data array."""
    return np.max(data, axis=0)


def daily_min(data):
    """Calculate the daily min of a 2d inflammation data array."""
    return np.min(data, axis=0)


class CSVDataSource:
    """Class to create an instance for CSV data files"""
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_inflammation_data(self):
        """Gets all the inflammation data from CSV files within a directory"""
        data_file_paths = glob.glob(os.path.join(self.data_dir, 'inflammation*.csv'))
        if len(data_file_paths) == 0:
            raise ValueError(f"No inflammation data CSV files found in path {self.data_dir}")
        data = map(load_csv, data_file_paths)

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
        data = map(load_json, data_file_paths)

        return list(data)


def compute_standard_deviation_by_day(data):
    """Calculates the standard deviation by day between datasets."""
    means_by_day = map(daily_mean, data)
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
