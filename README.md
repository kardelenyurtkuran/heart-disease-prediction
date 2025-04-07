# Heart Disease Prediction Project

## Overview
This project aims to predict heart disease using various machine learning models. The dataset includes features such as `age`, `chest pain type`, `cholesterol`, `oldpeak`, `ST slope`, `sex_1`, and `exercise angina_1`. The goal is to achieve high F1-score while minimizing False Negatives (FN), as missing a heart disease diagnosis can be critical in the healthcare domain.

## Project Structure
- **data/**: Contains the training and test datasets (`X_train_data.csv`, `y_train_data.csv`, `X_test_data.csv`, `y_test_data.csv`).
- **models/**: Trained models are saved here (e.g., `knn_model.pkl`, `random_forest_model.pkl`).
- **results/**: Best hyperparameters and visualizations are saved here (e.g., `best_params_knn.json`, `random_forest_tree.png`).
- **scripts/**: Training scripts for each model (e.g., `train_knn.py`, `train_random_forest.py`).
- **feature_engineering.ipynb**: Data preprocessing and feature scaling (already applied to the dataset).
- **evaluate_models.py**: Script to evaluate model performance and generate confusion matrices.

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd heart-disease-prediction
   ```
2. Create a virtual environment (optional but recommended):
    ```bash
     python -m venv venv
     source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3. Install the required dependencies:
    ```bash
     pip install -r requirements.txt
    ```
## Usage
1. Prepare the Data:
Ensure the dataset files (`X_train_data.csv`, `y_train_data.csv`, `X_test_data.csv`, `y_test_data.csv`) are in the `data/` directory.
The dataset is already preprocessed and scaled (via `feature_engineering.ipynb`).

2. Train a Model:
Run the training script for the desired model. For example, to train the KNN model:
    ```bash
     python scripts/train_knn.py
    ```
    This will perform Grid Search, save the best model to `models/knn_model.pkl`, and save the best hyperparameters to `results/best_params_knn.json`.

3. Evaluate Models:
Run the evaluation script to compare model performance:
    ```bash
       python evaluate_models.py
    ```

    This will output performance metrics (accuracy, precision, recall, F1-score) and confusion matrices for all trained models.

## Model Performance
The following table summarizes the performance of all trained models:

| Model              | Train Accuracy | Test Accuracy | Train Precision | Test Precision | Train Recall | Test Recall | Train F1 Score | Test F1 Score |
|--------------------|----------------|---------------|-----------------|----------------|--------------|-------------|----------------|---------------|
| KNN                | 0.868383405    | 0.851428571   | 0.868531798     | 0.851583504    | 0.868383405  | 0.851428571 | 0.868105069    | 0.851055474   |
| Naive Bayes        | 0.845493562    | 0.834285714   | 0.845689115     | 0.836190476    | 0.845493562  | 0.834285714 | 0.845563564    | 0.834557155   |
| Logistic Regression| 0.859799714    | 0.822857143   | 0.862130498     | 0.822714021    | 0.859799714  | 0.822857143 | 0.858876052    | 0.822541453   |
| Decision Tree      | 0.859799714    | 0.828571429   | 0.860311006     | 0.828405272    | 0.859799714  | 0.828571429 | 0.859340479    | 0.828379295   |
| Random Forest      | 0.927038627    | 0.845714286   | 0.927569527     | 0.846095238    | 0.927038627  | 0.845714286 | 0.926870995    | 0.845203828   |
| ANN                | 0.898426323    | 0.828571429   | 0.89838481      | 0.829931973    | 0.898426323  | 0.828571429 | 0.898359237    | 0.828818703   |
| SVM                | 0.878397711    | 0.851428571   | 0.887336037     | 0.852226767    | 0.878397711  | 0.851428571 | 0.876705524    | 0.850808375   |

### Key Insights:
- **Best Model**: KNN and SVM achieved the highest test F1-score (0.851), followed by Random Forest (0.845).
- **Healthcare Context**: In the healthcare domain, a test F1-score of 0.851 (approximately 15% error rate) is not sufficient, as False Negatives (FN) can be critical. A target F1-score of >0.95 and recall >0.95 is recommended.
- **Feature Importance**: Across models, `ST slope`, `chest pain type`, and `oldpeak` were consistently the most important features for predicting heart disease. For example:

#### Random Forest Feature Importance
| Feature            | Importance  |
|--------------------|-------------|
| ST slope          | 0.307832    |
| chest pain type   | 0.148015    |
| oldpeak           | 0.125594    |

#### Logistic Regression Coefficients
| Feature            | Importance  |
|--------------------|-------------|
| ST slope          | 0.045109    |
| chest pain type   | 0.031527    |
| exercise angina_1 | 0.028410    |

#### Naive Bayes Permutation Importance
| Feature            | Importance  |
|--------------------|-------------|
| ST slope          | 0.045109    |
| chest pain type   | 0.031527    |
| exercise angina_1 | 0.028410    |

## Future Improvements
- **More Data**: Collect more data to improve model generalization. The current dataset might be small, and healthcare problems often require thousands of samples for better performance.
- **Advanced Models**: Try gradient boosting methods like XGBoost or LightGBM to achieve higher performance. Example for XGBoost:

```python
from xgboost import XGBClassifier
    param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1],
    'scale_pos_weight': [1, 2, 3]  # To reduce FN
}
```
- **Threshold Tuning**: Adjust the prediction threshold to further reduce False Negatives (FN). See the "Adjust Threshold" section in Usage.
- **Feature Engineering**: Explore new features or interactions (e.g., age and cholesterol interaction) to capture more complex patterns.
- **Domain Expertise**: Collaborate with a cardiologist to validate feature importance and determine acceptable error rates (e.g., FN < 5%).
- **Additional Metrics**: Evaluate models using ROC-AUC and Precision-Recall AUC, which are more informative for imbalanced datasets in healthcare.

## License
This project is licensed under the MIT License - see the  file for details.
