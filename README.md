# Titanic_survival_prediction_ML
<br>
This README outlines the steps involved in building and evaluating a machine learning model, starting from preprocessing to final evaluation.

### 1. Preprocessing

### 1.1 Loading the Data
- Train dataset: Contains features and target variable (`Survived` for Titanic dataset).
- Test dataset: Contains features to predict the target variable.

### 1.2 Handling Missing Values
- **`Age`**: Filled with the median.
- **`Embarked`**: Filled with the mode.
- **`Fare`**: Filled with the median.
- **`Cabin`**: Created a binary feature `HasCabin` and dropped the original column.

### 1.3 Feature Engineering
- Added new features:
  - `FamilySize` = `SibSp` + `Parch` + 1
  - `IsAlone`: Whether the passenger is traveling alone.
  - `Title`: Extracted title from the `Name` feature.
  - `FarePerPerson` = `Fare` / `FamilySize`
  - `AgeGroup`: Binned age into categories like Child, Teen, Adult, Senior.
  - `TicketPrefix`: Extracted the prefix from the `Ticket` feature.

### 1.4 Handling Categorical Features
- Converted categorical features (`Sex`, `Embarked`, etc.) to numerical using encoding techniques like:
  - Label Encoding
  - One-Hot Encoding (if required for specific models).

### 1.5 Scaling Numerical Features
- Applied log transformation to reduce skewness for `Age` and `Fare`:
  ```python
  combined_data['Age_log'] = np.log1p(combined_data['Age'])
  combined_data['Fare_log'] = np.log1p(combined_data['Fare'])
