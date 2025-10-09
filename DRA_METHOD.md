# Drag Reduction Modelling Approach

The optimiser computes drag reduction effects directly from the Burger equation.
Supporting utilities in [`dra_utils.py`](dra_utils.py) expose helpers to convert
between drag-reduction percentages and the PPM lacing required at the measured
flow velocity and pipe diameter. These helpers are imported by
[`pipeline_model.py`](pipeline_model.py) and used throughout the hydraulic and
optimisation routines to enforce the minimum-PPM floors and to cost DRA usage.

Although several viscosity CSV files are packaged with the repository, the
optimiser does not read any of them when evaluating drag reduction. All DRA
conversions are calculated analytically via the Burger-equation helpers.
