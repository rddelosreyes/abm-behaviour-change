This code accompanies the following paper:

**Delos Reyes, R., Lyons Keenan, H., & Zachreson, C. (2024). _An agent-based model of behaviour change calibrated to reversal learning data_. arXiv. [https://doi.org/10.48550/arXiv.2406.14062](https://doi.org/10.48550/arXiv.2406.14062)**

The experimental data used for calibration and validation can be found in the `data` folder:
  - `data/target_data_single.csv`: preprocessed single reversal learning data of large-brained guppies from [Buechel et al. (2018)](https://datadryad.org/stash/dataset/doi:10.5061/dryad.cm503)
  - `data/target_data_series.csv`: preprocessed serial reversal learning data of large-brained guppies from [Boussard et al. (2020)](https://datadryad.org/stash/dataset/doi:10.5061/dryad.5mkkwh72s)

The simulation results reported in the paper can be found in the `experiments` folder:
  - `experiments/run/single/output_1_20240621_000000`: contains the data for the model dynamics shown in Figure 3 (see `plot_figure3.ipynb` for plotting code)
  - `experiments/calibrate/single/output_1_20240621_000000`: contains the data for the single reversal learning result shown in Figure 4 (see `plot_figure4.ipynb` for plotting code)
  - `experiments/calibrate/serial/output_1_20240621_000000`: contains the data for the serial reversal learning result shown in Figure 5 (see `plot_figure5.ipynb` for plotting code)

Alternatively, to run your own experiments, please see the commands in the `run.sh` file. The configuration files in the `config` folder are identical to those used for our experiments with the exception of the `ABC_SAMPLE_COUNT` parameter, which is set to `10` rather than `1000` so that the experiment is faster to run. 

Cite as:
```
@misc{delosreyes2024agentbased,
      title={An agent-based model of behaviour change calibrated to reversal learning data}, 
      author={Roben {Delos Reyes} and Hugo {Lyons Keenan} and Cameron Zachreson},
      year={2024},
      eprint={2406.14062},
      archivePrefix={arXiv},
      primaryClass={id='q-bio.QM' full_name='Quantitative Methods' is_active=True alt_name=None in_archive='q-bio' is_general=False description='All experimental, numerical, statistical and mathematical contributions of value to biology'}
}
```
