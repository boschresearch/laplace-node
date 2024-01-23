import os

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from tueplots import bundles, figsizes, fontsizes

from cluster_functionalities.util import get_results_dir
from mcmc_time_series.plotting.plot_uncertainty import MCMCFinalSolutionUncertaintyPlot
from mcmc_time_series.plotting.plot_vector_field import MCMCVectorFieldPlotUncertainty
from time_series.plotting.plot_final_solution_uncertainty import FinalSolutionUncertaintyPlot
from time_series.plotting.plot_final_solution_uncertainty_sample import SampleUncertaintyPlot
from time_series.plotting.plot_vector_field_sample import VectorFieldPlotSample
from time_series.plotting.plot_vector_field_uncertainty import VectorFieldPlotUncertainty

plt.style.use("paper")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

bundle = bundles.icml2022()
figsize_icml_ = figsizes.icml2022(column="half", nrows=2)
figsize_icml_["figure.figsize"] = (4.1, 2.5)
# plt.rcParams.update(fontsizes.icml2022())
plt.rcParams.update(figsize_icml_)

fig_path = "C:\\Users\\otk2rng\\Documents\\Submissions\\ICML2022\\icml-2022\\fig"
exp_dir = get_results_dir()
exp_dir = os.path.join(exp_dir, "ICML",  "mcmc_a46f0ffe")
mcmc_half = "run_dataset_lotka_volterra_half_hidden_dim_16_num_samples_2000"
mcmc_half = os.path.join(exp_dir, mcmc_half)

fig_path = "C:\\Users\\otk2rng\\Documents\\Submissions\\ICML2022\\icml-2022\\fig"
exp_dir = get_results_dir()
exp_dir = os.path.join(exp_dir, "ICML",  "sample")
sample_full = "run"
sample_full = os.path.join(exp_dir, sample_full)

exp_dir = get_results_dir()
exp_dir = os.path.join(exp_dir, "ICML", "lotka_volterra_bf5af12d")
full = "run_dataset_lotka_volterra_half"
full = os.path.join(exp_dir, full)

fig, axs = plt.subplots(2, 3)
p_max = 0.2

plot = SampleUncertaintyPlot(sample_full)
ax = axs[0, 0]
ax.set_title("A1   Laplace sample", loc="left")
plot.plot(ax=ax)
ax.set_ylabel("$x, y$")
ax.set_ylim([-1, 3])

plot = FinalSolutionUncertaintyPlot(full)
ax = axs[0, 1]
ax.set_title("B1    Laplace linearized", loc="left")
plot.plot(ax=ax)
ax.set_ylabel("")
ax.set(yticklabels=[])
ax.set_ylim([-1, 3])

plot = MCMCFinalSolutionUncertaintyPlot(mcmc_half)
ax = axs[0, 2]
ax.set_title("C1    HMC", loc="left")
plot.plot(ax=ax)
ax.set_ylabel("")
ax.set(yticklabels=[])
ax.legend(ncol=2)
ax.set_ylim([-1, 3])

plot = VectorFieldPlotSample(sample_full)
ax = axs[1, 0]
ax.set_title("A2", loc="left")
pc = plot.plot(ax=ax)
ax.set_ylabel("$y$")
ax.set_xlabel("$x$")
ax.legend(ncol=2)
pc.set_clim(0, p_max)

plot = VectorFieldPlotUncertainty(full)
ax = axs[1, 1]
ax.set_title("B2", loc="left")
pc = plot.plot(ax=ax)
ax.set_ylabel("")
ax.set_xlabel("$x$")
ax.set(yticklabels=[])
pc.set_clim(0, p_max)

plot = MCMCVectorFieldPlotUncertainty(mcmc_half)
ax = axs[1, 2]
ax.set_title("C2", loc="left")
pc = plot.plot(ax=ax)
ax.set_ylabel("")
ax.set_xlabel("$x$")
ax.set(yticklabels=[])
ax.legend(ncol=2)
pc.set_clim(0, p_max)
fig.colorbar(pc, ax=ax, shrink=1., location="right", orientation="vertical")

fig.savefig(os.path.join(fig_path, "sample.png"), dpi=500)
fig.savefig(os.path.join(fig_path, "sample.pdf"), dpi=500)
