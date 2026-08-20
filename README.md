# WiCleanData

The source code of mining **WiCleanData**, a refined version of Wikidata with a consistent taxonomy and free from type constraint violations. This project is licensed under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

## Installation

```bash
pip install -r requirements.txt
```

For running LLMs, please place `access_token_read_hf=<token>` in a `.env` file at the project root. All pipeline paths are defined in [`pipeline_config.py`](pipeline_config.py). To use a different data directory, do:
```bash
export WICLEAN_DATA_DIR=/path/to/your/data
```


## Refining Pipeline

All data are stored under [`data/`](data/) (see the local [README](data/README.md) for details).

The automatic cleaning pipeline runs in three stages: taxonomy cleaning, constraints cleaning, and facts cleaning. Each stage consumes outputs from the previous one. To reproduce the results:

```bash
make all                 # taxonomy → constraints → facts
# Or, stage by stage:
make taxonomy
make constraints
make facts
```

`make constraints` and `make facts` run their upstream stages first. To run only one stage on existing outputs:

```bash
make facts SKIP_DEPS=1
```

Equivalently, each stage can be launched from its own script:

```bash
bash src/taxonomy/run.sh      # taxonomy cleaning
bash src/constraints/run.sh   # constraints cleaning
bash src/facts/run.sh         # facts cleaning
```
Note: in `src/taxonomy/run.sh`, extraction and final refinement run on CPU; the LLM inference and rewire steps typically require a GPU.

## Evaluation

- Intrinsic evaluation lives in `src/analysis/`. After the pipeline has finished:

```bash
make analysis
# or: bash src/analysis/run.sh
```

- Extrinsic evaluation uses [KGrEaT](https://github.com/dwslab/kgreat) to assess WiCleanData on downstream tasks such as classification and recommendation. Create a new knowledge graph named `WiCleanData` under the `kg/` folder of the KGrEaT repository, place the data into `kg/WiCleanData/`, and configure `config.yaml`. Please see the KGrEaT repository for further setup and usage details.

## Website

`src/website/` provides two lightweight viewers:

- [`constraint_viewer/`](src/website/constraint_viewer/) — summarizes property type constraints
- [`taxonomy_viewer/`](src/website/taxonomy_viewer/) — visualizes the class hierarchy DAG for a given class

To run either viewer, execute:

```bash
pip install -r requirements.txt
python3 src/server.py <port>
```

See each viewer's local README for details.

The WiCleanData website is also publicly available at: https://wicleandata.r2.enst.fr/

## Citation