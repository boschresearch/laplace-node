# PyTorch Implementation for ODE and Neural ODE solvers
The experiments based on this library are fully supported to run on a single GPU. 
By default, the device is set to cpu. 

## Installation

To install all necessary packages via conda in an environment `ode-software` execute
```
conda env create -f requirements.yml
```
To install all necessary packages with gpu in an environment `ode-software` execute
```
conda env create -f requirements-gpu.yml
```
## Training

### Time Series

To run code with different options you can use
```
python train_time_series.py --<option_name> <option_argument>
```


