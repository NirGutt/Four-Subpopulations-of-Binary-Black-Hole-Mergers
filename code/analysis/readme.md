# Population-analysis workflow

This directory contains the code and configuration required to reproduce the
hierarchical population analysis.

The analysis is performed with
[`gwpopulation`](https://colmtalbot.github.io/gwpopulation/) and
[`gwpopulation_pipe`](https://docs.ligo.org/RatesAndPopulations/gwpopulation_pipe/).
The latter provides the command-line workflow used to collect the event
posteriors, evaluate the selection function, run the population inference, and
post-process the result.

For complete descriptions of the software and command-line options, see:

- [`gwpopulation` documentation](https://colmtalbot.github.io/gwpopulation/);
- [`gwpopulation_pipe` documentation](https://docs.ligo.org/ratesandpopulations/gwpopulation_pipe/);
- [`gwpopulation_pipe` command-line options](https://docs.ligo.org/ratesandpopulations/gwpopulation_pipe/configuration.html).

## This Directory contents

```text
code/analysis/
├── README.md
├── config/       # gwpopulation_pipe configuration and Bilby prior files
├── data/         # downloaded event posteriors and injections script
├── models/       # custom population-model implementation
└── postprocessing/      # post postprocessing scripts
```

The large GWTC event posterior and injection files are not stored in Git.
`data/` is populated by the download script.

## Software environment

Run all commands below from the top-level directory of the repository.

Create the Conda environment specified in `code/environment.yml`:

```bash
conda env create --file code/environment.yml
conda activate populations_env
```

The environment name in the second command must match the `name` entry in
`code/environment.yml`.


The full analysis was tested on Rocky Linux 8.10 using one NVIDIA
A100-SXM4-80GB GPU and the JAX backend.

## Download the input data

Download the event-level posterior samples and cumulative search injections
with:

```bash
bash code/analysis/data/download_all_gwtc_data.sh code/analysis/data
```

The script downloads the required GWTC posterior and injection files.

## Analysis configuration

The analysis is defined by two files (the names are used only for example here):

```text
code/analysis/config/analysis.ini
code/analysis/config/population.prior
```

The INI file specifies:

- the event-posterior and injection locations;
- the parameters extracted from each event;
- the custom model functions;
- the prior file;
- the sampler and sampler settings;

An example INI file is provided under `code/analysis/config/`. However, it will need to be adapted to the specific model and analysis configuration.

The model implementation is stored under `code/analysis/models/` and is passed
to `gwpopulation_pipe` through the `source-files` configuration option.


## Run a complete workflow

Generate the data-collection, analysis, and post-processing workflow with:

```bash
gwpopulation_pipe code/analysis/config/analysis.ini  
```
The main `gwpopulation_pipe` executable performs three tasks:

1. `gwpopulation_pipe_collection` reads the individual-event
   posterior samples and prepares the selection-function data.
2. `gwpopulation_pipe_analysis` performs the hierarchical Bayesian inference.
3. `gwpopulation_pipe_plot` creates the standard diagnostic and
   post-processing plots.

Depending on the settings in `analysis.ini`, the command generates either a
local Bash workflow or an HTCondor workflow. For an HTCondor run, submit the
DAG produced in the run directory using the command printed by
`gwpopulation_pipe`.

## Run only the inference stage

If the event posteriors and injection data have already been collected, the
inference stage can be launched directly:

```bash
gwpopulation_pipe_analysis \
    code/analysis/config/analysis.ini \
    --run-dir code/analysis/runs/gwtc5_four_subpopulations \
    --label gwtc5_four_subpopulations \
    --vt-file code/analysis/runs/gwtc5_four_subpopulations/data/injections.pkl
```

This is the main executable used by the HTCondor analysis job. 

## Outputs

The run directory contains:

- the collected event posteriors;
- the prepared injection data;
- the Bilby population-inference result;
- sampler and pipeline logs;


The final population-level posterior is found in the `results/` directory. 

Paper-specific figures are generated from that released posterior by the scripts under
`code/postprocessing/`.

The full inference takes a long time (~24h) on a GPU and is stochastic. Meaning, a successful reproduction should agree well with
the posterior distributions and credible intervals; however, the newly generated result file is not expected to be byte-for-byte identical.



