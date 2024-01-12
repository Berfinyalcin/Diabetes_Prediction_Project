# Diabetes Prediction Project
This project contains data from a large dataset maintained by the National Institute of Diabetes and Digestive and Kidney Diseases in the United States. It is a subset of the data used for a diabetes study conducted on Pima Indian women residing in Phoenix, the 5th largest city in the state of Arizona, aged 21 and above. The dataset comprises 768 observations and 8 numerical independent variables. The target variable, "outcome," indicates whether the diabetes test result is positive (1) or negative (0).
## Variables
- Pregnancies: Number of pregnancies.
- Glucose: Glucose level.
- BloodPressure: Blood pressure.
- SkinThickness: Skin thickness.
- Insulin: Insulin level.
- BMI: Body mass index.
- DiabetesPedigreeFunction: A function calculating the likelihood of diabetes based on family history.
- Age: Age (in years).
- Outcome: Information about whether the individual has diabetes or not. Affected (1) or not affected (0).
## Project Objectives
This project aims to predict whether Pima Indian women have diabetes or not using the features in the dataset. Fundamental feature engineering techniques and a logistic regression model will be employed for model training, evaluation, and validation.
## Libraries Used
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
## Development Steps
### Data Exploration and Cleaning:
Cleanse missing or nonsensical values, understand the dataset.
### Feature Engineering:
Apply various engineering techniques on existing features.
### Model Training and Evaluation:
Train a logistic regression model and assess its performance.
### Model Validation:
Validate the model's accuracy using Holdout and 10-Fold Cross Validation methods.
### Prediction for a New Observation:
Utilize the trained model to predict diabetes for a new observation.

