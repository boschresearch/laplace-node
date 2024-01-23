from mcmc_time_series.mcmcnode import main
from mcmc_time_series.options.mcmc_options import MCMCOptions
from train_base import init_options

if __name__ == "__main__":

    options = MCMCOptions
    opts = init_options(options)
    main(opts)

