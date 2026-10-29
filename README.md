# Prodigy ML Task 01

This project trains a linear regression model to predict house sale prices using selected features from the Ames Housing dataset.

## What the project does

[`ML-TASK-01.py`](./ML-TASK-01.py) performs the following steps:

1. Loads `train.csv`.
2. Uses these input features:
   - `GrLivArea` - above-ground living area in square feet
   - `BedroomAbvGr` - number of bedrooms above ground
   - `FullBath` - number of full bathrooms
3. Uses `SalePrice` as the target.
4. Splits the data into 80% training and 20% evaluation sets with `random_state=42`.
5. Trains a scikit-learn `LinearRegression` model.
6. Reports mean squared error and R-squared score.
7. Predicts the price of a sample house with 2,000 square feet, 3 bedrooms, and 2 full bathrooms.

The notebook version is available in [`ML_Task01.ipynb`](./ML_Task01.ipynb).

## Project files

| File | Description |
| --- | --- |
| [`ML-TASK-01.py`](./ML-TASK-01.py) | Executable training and evaluation script |
| [`ML_Task01.ipynb`](./ML_Task01.ipynb) | Notebook version of the task |
| [`train.csv`](./train.csv) | Training data: 1,460 rows and 81 columns, including `SalePrice` |
| [`test.csv`](./test.csv) | Unlabelled test data: 1,459 rows and 80 columns |
| [`sample_submission.csv`](./sample_submission.csv) | Submission template with 1,459 rows |
| [`data_description.txt`](./data_description.txt) | Descriptions of the dataset columns |

## Requirements

- Python 3.12.1 was used for verification.
- pandas 2.2.3
- scikit-learn 1.9.0

If the packages are not already installed, install them with:

```powershell
python -m pip install pandas scikit-learn
```

## Run the project

Open PowerShell in this project directory and run:

```powershell
python .\ML-TASK-01.py
```

Run the command from the project directory because the script reads `train.csv` using a relative path.

## Verified output

The script was run successfully from the project directory. The observed output included:

```text
Mean Squared Error: 2806426667.25
R-squared Score: 0.6341
Predicted House Price: $240,377.51
```

The exact metrics can vary if the data or model configuration changes.

## Notes

This is a baseline model using only three numeric features and no preprocessing or hyperparameter tuning. Its R-squared score should therefore be treated as a starting point rather than a production-quality estimate. The provided `test.csv` and `sample_submission.csv` are included for possible future submission generation, but the current script evaluates the model using a held-out portion of `train.csv`.
