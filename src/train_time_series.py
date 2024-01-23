import matplotlib

from train_base import init_options

matplotlib.use(backend="AGG")
from time_series.time_series_options.experiment_options import ExperimentOptions
from time_series.train_time_series import run

if __name__ == "__main__":

    options = ExperimentOptions
    opts = init_options(options)

    run(opts)
