# ML Workflow for TS-Geometry Prediction

This repository contains the Python scripts used to reproduce the machine-learning analyses for Reaction B and Reaction C reported in the manuscript:

**Hammett-Guided Machine Learning for Transition State Geometry Prediction in Nitrosobenzene Additions**

Jingxing Cheng,<sup>a</sup> Pauline Bianchi,<sup>a</sup> and Jean-Christophe M. Monbaliu<sup>a,b,</sup>*

Affiliations: \
<sup>a</sup> Center for Integrated Technology and Organic Synthesis (CiTOS), MolSys Research Unit, University of Liège
    B6a, Room 3/19, Allée du Six Août 13, 4000 Liège (Sart Tilman), Belgium
    Homepage: www.citos.uliege.be
    E-mail: jc.monbaliu@uliege.be\
<sup>b</sup> WEL Research Institute, Avenue Pasteur 6, B-1300 Wavre, Belgium

##
Each regression model uses the corresponding Reaction A transition-state geometry together with a single numerical descriptor to predict one geometric parameter, bond length, angle, or dihedral, for Reaction B or Reaction C.

The workflow separates **descriptor screening** from **model development**. Descriptor-screening scripts compare a broad candidate set of Hammett-related and DFT-derived substituent and rank them by LOOCV Q². The machine-learning scripts then build the final Hammett-based models (σp+, σp, σp− by default), export fitted equations, and support downstream external validation.

The machine-learning workflow and implementation were developed by Jingxing Cheng under the supervision of Jean-Christophe M. Monbaliu.

*E-mail: <jc.monbaliu@uliege.be>*

## Repository contents

```text
TS_geometry_ML_prediction/
├── README.md
└──TS_geometry_ML_prediction/
       ├── datasets/
       │   ├── reaction_B_dataset.csv
       │   ├── reaction_B_external-set.csv
       │   ├── reaction_C_dataset.csv
       │   └── reaction_C_external-set.csv
       ├── reaction_B_descriptor_screening.py
       ├── reaction_B_machine_learning_model.py
       ├── reaction_B_method_comparison.py
       ├── reaction_B_external_test_prediction.py
       ├── reaction_C_descriptor_screening.py
       ├── reaction_C_machine_learning_model.py
       ├── reaction_C_method_comparison.py
       ├── reaction_C_external_test_prediction.py
       ├── results/
       ├── reaction_B_descriptor_screening
       ├── reaction_B_ml_results
       ├── reaction_B_method_comparison
       ├── reaction_B_external_test_predictions
       ├── reaction_C_descriptor_screening
       ├── reaction_C_ml_results
       ├── reaction_C_method_comparison
       └── reaction_C_external_test_predictions
```
## How to cite this material

DOI: 10.5281/zenodo.21376455

## Scripts

- `reaction_B_descriptor_screening.py`: compares candidate Reaction B substituent descriptors, including Hammett-related parameters (for example σp+, σp, σp−) and DFT-derived descriptors (for example EHOMO, ELUMO, η, ω, N, f-N, NN, ωN, B1, B5, L, and v%), and ranks them by bond-length LOOCV Q².
- `reaction_B_machine_learning_model.py`: builds the final Hammett-based Reaction B models (σp+, σp, σp− by default). Unlike the descriptor-screening script, it focuses on Hammett constants, selects the best descriptor per target, and exports LOOCV metrics, model equations, predictions, parity plots, and software versions for downstream use.
- `reaction_B_method_comparison.py`: compares Ridge, Lasso, Elastic Net, and Bayesian Ridge regression for Reaction B.
- `reaction_B_external_test_prediction.py`: applies fixed Reaction B equations to an external test set without refitting.
- `reaction_C_descriptor_screening.py`: compares candidate Reaction C nucleophile descriptors, including Hammett-related parameters (for example σp+, σp, σp−) and DFT-derived descriptors (for example EHOMO, ELUMO, η, ω, N, B1, B5, L, v%, f-c, Nc, and ωc), and ranks them by dihedral LOOCV Q².
- `reaction_C_machine_learning_model.py`: builds the final Hammett-based Reaction C models (σp+, σp, σp− by default). Unlike the descriptor-screening script, it focuses on Hammett constants, selects the best descriptor per target, and exports LOOCV metrics, model equations, predictions, parity plots, and software versions for downstream use.
- `reaction_C_method_comparison.py`: compares Ridge, Lasso, Elastic Net, and Bayesian Ridge regression for Reaction C.
- `reaction_C_external_test_prediction.py`: applies fixed Reaction C equations to an external test set without refitting.

## Environment

Python 3.9 or later is recommended.

Required packages:

```text
numpy
pandas
matplotlib
scikit-learn
```

Create and activate an isolated environment:

```bash
python -m venv .venv
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib scikit-learn
```

## Typical workflow

1. Prepare the input CSV file using the required column names or one of the accepted aliases defined in the corresponding script.
2. Run the descriptor-screening script for Reaction B or Reaction C to compare Hammett-related and DFT-derived descriptors and identify the most predictive candidates.
3. Run the Hammett-focused model-development script for Reaction B or Reaction C to fit the final Ridge models and export equations.
4. Run the method-comparison script to compare the available regression approaches, if required.
5. Run the external-test script to evaluate the fixed equations on an independent external test set.
6. Inspect the generated metrics, predictions, equations, and parity plots.

## Running the scripts

Run each script from the `TS_geometry_ML_prediction/` directory:

```bash
python reaction_B_descriptor_screening.py --input datasets/reaction_B_dataset.csv --output results/reaction_B_descriptor_screening
python reaction_B_machine_learning_model.py --input datasets/reaction_B_dataset.csv --output results/reaction_B_ml_results --alpha 0.1
python reaction_B_method_comparison.py --input datasets/reaction_B_dataset.csv --output results/reaction_B_method_comparison --descriptor "σp+"
python reaction_B_external_test_prediction.py --input datasets/reaction_B_external-set.csv

python reaction_C_descriptor_screening.py --input datasets/reaction_C_dataset.csv --output results/reaction_C_descriptor_screening
python reaction_C_machine_learning_model.py --input datasets/reaction_C_dataset.csv --output results/reaction_C_ml_results --alpha 0.1
python reaction_C_method_comparison.py --input datasets/reaction_C_dataset.csv --output results/reaction_C_method_comparison --descriptor "σp+"
python reaction_C_external_test_prediction.py --input datasets/reaction_C_external-set.csv
```

The scripts currently expect their input files and configuration to follow the paths and names defined in the source code. Update those paths when necessary before execution.

## Model and validation

Bond lengths, angles, and dihedrals are modeled separately.

Internal validation is performed using leave-one-out cross-validation (LOOCV). Training metrics and LOOCV metrics are reported separately to distinguish model fit from cross-validated predictive performance.

## Input data

Input files must be CSV tables containing:

- compound identifiers;
- substituent labels;
- Reaction A transition-state geometries;
- target Reaction B or Reaction C geometries;
- selected numerical descriptor columns.

Accepted column aliases are defined near the top of each script. Preserve the original column names or use one of the listed aliases.

## Outputs

### Descriptor-screening scripts

Reaction B ranks descriptors by bond-length LOOCV Q². Reaction C ranks descriptors by dihedral LOOCV Q² and includes a `rank` column in `descriptor_ranking.csv`.
- `descriptor_target_metrics.csv`: LOOCV Q² and sample size for each descriptor/target combination
- `descriptor_ranking.csv`: ranked summary table used to build the figure
- `figure.png`
- `figure.pdf`

### Model-development scripts

- `model_metrics.csv`
- `model_predictions.csv`
- `model_equations.txt`
- `best_model_parity.png`
- `best_model_parity.pdf`
- `software_versions.json`

### Method-comparison scripts

- `method_target_metrics.csv`
- `method_summary.csv`
- `loocv_predictions.csv`
- `fitted_model_coefficients.csv`
- comparison and ranking figures

### External-test scripts

- `external_test_predictions.csv`
- `external_test_metrics.csv`
- `external_test_parity.png`
- `external_test_parity.pdf`

## Reproducibility notes

- Model-development scripts use deterministic LOOCV and do not rely on a random train/test split.
- Method-comparison hyperparameters are fixed to reproduce the reported analysis.
- External-test scripts apply fixed model coefficients and do not refit the models.
- For small external test sets, interpret R<sup>2</sup> together with MAE, RMSE, residuals, and parity plots.
- Software-version differences may lead to minor numerical or graphical variations.

## Data and software availability

The scripts are author-generated and may be shared together with the associated machine-readable data. Quantum-chemistry output files, optimized coordinates, and processed datasets are provided in the Supporting Information accompanying the manuscript.

## Acknowledgements 

This work is supported by the Walloon Region as part of the funding for the FRFS-WEL-T strategic axis. The authors acknowledge the WEL Research Institute (grant WEL-T-CR-2023 A – 05, "Smart Flow Systems"). Computational resources were provided by the “Consortium des Équipements de Calcul Intensif” (CÉCI), funded by the “Fonds de la Recherche Scientifique de Belgique” (F.R.S.-FNRS) under Grant No. 2.5020.11a and by the Walloon Region. 

## License

The code in this repository is licensed under the MIT License. You are free to use, modify, and distribute the code under the terms of this license.
