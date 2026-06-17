# WiCleanData

The source code of mining **WiCleanData**, a refined version of Wikidata with a consistent taxonomy and free from type constraint violations. This project is licensed under the MIT license

## Installation

```bash
pip install -r requirements.txt
```

## Refining Pipeline

The automatic cleaning pipeline runs in three stages: taxonomy cleaning, constraints cleaning and finally facts cleaning. All data and intermediate outputs are stored in the [`data/`](data/) folder (see the local README for details). 
To reproduce the results, run the following stages in order. Each stage consumes outputs from the previous one.

```bash
bash src/taxonomy/cleanTaxonomy.sh
python cleanConstraints.py --taxonomy ../../data/wicleanTaxonomy.txt
bash src/facts/run.sh
```

Note: You may need to run `python src/taxonomy/llm_infer_rewire.py --llm $llms$` if the rewired links have not been checked during taxonomy refinement.

## Evaluation

Intrinsic evaluation uses `src/analysis/` to measure taxonomy semantic coherence, structural distance, and robustness. Extrinsic evaluation uses [KGrEaT](https://github.com/dwslab/kgreat) to assess WiCleanData on downstream tasks such as classification and recommendation. See the KGrEaT repository for setup and usage details.

## Website

The `src/website/` directory provides two lightweight web viewers: `constraint_viewer/`, which summarizes property type constraints, and `taxonomy_viewer/`, which visualizes the class hierarchy DAG for a given class. To run either viewer, execute `python3 src/server.py <port>` from within its directory (see the local README for details).

The WiCleanData website is also publicly available at: https://wicleandata.r2.enst.fr/

