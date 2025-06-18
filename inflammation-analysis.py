#!/usr/bin/env python3
"""Software for managing and analysing patients' inflammation data in our imaginary hospital."""

import argparse
import os

from inflammation import models, views


def main(args):
    """The MVC Controller of the patient inflammation data system.

    The Controller is responsible for:
    - selecting the necessary models and views for the current task
    - passing data between models and views
    """
    infiles = args.infiles
    if not isinstance(infiles, list):
        infiles = [args.infiles]


    if args.full_data_analysis:
        files = [f for f in os.listdir(infiles[0]) if os.path.isfile(os.path.join(infiles[0], f))]
        extensions = []
        for f in files:
            ext = os.path.splitext(f)[1]
            if ext and ext[1:] not in extensions:
                extensions.append(ext[1:])
        
        dataset = []
        for ext in extensions:
            if ext == 'json':
                data_type = models.JSONDataSource(os.path.dirname(infiles[0]))
                data = data_type.load_inflammation_data()
                dataset.extend(data)
            elif ext == 'csv':
                data_type = models.CSVDataSource(os.path.dirname(infiles[0]))
                data = data_type.load_inflammation_data()
                dataset.extend(data)
            else:
                print(f'Unsupported data file format: .{ext}')

        if dataset:
            results = models.analyse_data(dataset)
            views.plot_data(results)
        else:
            raise ValueError("No supported data file formats found in directory.")

        return

    for filename in infiles:
        inflammation_data = models.load_csv(filename)

        view_data = {
            'average': models.daily_mean(inflammation_data),
            'max': models.daily_max(inflammation_data),
            'min': models.daily_min(inflammation_data)
        }

        views.visualize(view_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='A basic patient inflammation data management system')

    parser.add_argument(
        'infiles',
        nargs='+',
        help='Input CSV(s) containing inflammation series for each patient')

    parser.add_argument(
        '--full-data-analysis',
        action='store_true',
        dest='full_data_analysis')

    args = parser.parse_args()

    main(args)
