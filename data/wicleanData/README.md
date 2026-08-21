# WiCleanData

Cleaned taxonomy, constraints, and facts produced by the pipeline. Run `make all` (or the stage scripts under `src/`) to populate intermediate folders and final files.

## Layout

```
wicleanData/
├── facts/                              # Intermediate extracted facts
├── instTypes/                          # Instance types after filtering / retyping
├── constraints/                        # Unused constraint types (case study)
└── statistics/                         # Intrinsic evaluation outputs
```
The final cleaned data will be in `wicleanData/`


## File description

| File | Description |
|------|-------------|
| `wicleanTaxonomy.txt` | Cleaned taxonomy: `childQID,parentQID` |
| `wicleanLabels.txt` | Class labels: `QID\t"label"` |
| `wiclean_mapping.txt` | Remapping from removed/merged classes to retained parents: `childQID,parentQID` |
| `subject_constraints_types_clean.csv` | Subject-type constraints after cleaning |
| `value_constraints_types_clean.csv` | Value-type constraints after cleaning |
| `wiclean_facts.tsv` | Final fact triples after type-constraint checking (produced by the facts stage) |

**Due to space limit, the full cleaned data can be found here: [data](https://drive.google.com/drive/folders/1NLvsvY-R40SZ_TOvRH4ZwQkSvNNBVLal?usp=sharing).**