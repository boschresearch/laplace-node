
import matplotlib

from aphynity.aphinity_options.aphinityoptions import AphinityOptions
from aphynity.train_aphynity import run
from train_base import init_options

matplotlib.use(backend="AGG")

if __name__ == "__main__":

    options = AphinityOptions
    opts = init_options(options)

    run(opts)
