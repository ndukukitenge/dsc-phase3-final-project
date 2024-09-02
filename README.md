# dsc-phase3-final-project
Final project
# Phase 3 Project 


## 1. Project Setup
- Import necessary libraries (pandas, numpy, sklearn, matplotlib, seaborn)
- Load the dataset
- Set random seed for reproducibility

## 2. Data Exploration and Preprocessing
- Examine the dataset (head, info, describe)
- Check for missing values and handle them
- Explore data distributions and correlations
- Perform feature engineering if necessary
- Split the data into features (X) and target variable(s) (y)

## 3. Logistic Regression
- Prepare the data (ensure binary target variable)
- Create and train the model
- Make predictions on the test set
- Evaluate the model (accuracy, precision, recall, F1-score)
- Plot the ROC curve and calculate AUC
- Analyze coefficients and their significance

## 4. Conclusion and Future Work
- Summarize key findings
- Discuss limitations of the current approach
- Suggest potential improvements or additional models to try
## Student details
Nduku Kitenge
DSC-PT07 

## Business Problem

The goal of this project is to predict which pumps are functional, which need some repairs and which dont work at all. 

The challenge from DrivenData. (2015). Pump it Up: Data Mining the Water Table. Retrieved [Month Day Year] from https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table  with data from Taarifa and the Tanzanian Ministry of Water. The goal of this project is to predict one of these three classes based on a number of variables about what kind of pump is operating, when it was installed, and how it is managed. A smart understanding of which waterpoints will fail can improve maintenance operations and ensure that clean, potable water is available to communities across Tanzania.


## 1. Project Setup
- Import necessary libraries (pandas, numpy, sklearn, matplotlib, seaborn)
- Load the dataset
#import neccessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#Load training dataset
sub_df = pd.read_csv('./SubmissionFormat.csv',index_col=0)
training_df = pd.read_csv('./TrainingSetValues.csv',index_col=0)
test_df = pd.read_csv('./TestSetValues.csv',index_col=0)
t_label_df = pd.read_csv('TrainingSetLabels.csv',index_col=0)



## 2. Data Exploration and Preprocessing
- Examine the dataset (head, info, describe)
- Check for missing values and handle them
- Explore data distributions and correlations
- Perform feature engineering eg. add pump age
- Split the data into features (X) and target variable(s) (y)
# Function to display dataset info
def display_dataset_info(df, name):
    print(f"\n=== {name} ===")
    print(f"Shape: {df.shape}")
    print("\nInfo:")
    print(df.info())
    print("\nDescription:")
    print(df.describe())
    print("\nHead:")
    print(df.head())
    print("\n" + "="*40)

# Display info for each dataset
display_dataset_info(sub_df, "Submission Format")
display_dataset_info(training_df, "Training Set Values")
display_dataset_info(test_df, "Test Set Values")
display_dataset_info(t_label_df, "Training Set Labels")
#check for missing values in training dataset
training_df.isna().sum()


Dropping columns and reasons why:
Scheme name - Too many null values 
wpt_name - Too many unique values
management - management group covers this
quality group - quality covers this
quantity group
extracton type group 

Numerical columns:
num private - dont know what this means. 

Observations
Funder and installer are similar, but funder has less null values than installer. 
Waterpoint type and water point are similar - Will drop waterpoint type. 


#Check categorical columns
training_df.select_dtypes(include=['object']).columns
#check numerical columns
training_df.select_dtypes(include=['int64', 'float64']).columns
#Define the list of columns to drop
columns_drop = ['wpt_name','lga','ward','quality_group','extraction_type_class','management','source','waterpoint_type_group','num_private']

# Drop unnecessary columns
training_df = training_df.drop(columns_drop,axis=1)

#Print remaining columns
print(training_df.columns)
#explore data distrubitions and correlations of the numerical- training data
corr_matrix = training_df.corr()

# Plot the correlation matrix using Seaborn for numerical values.
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True, linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()
# merge with training labels data using ID as connector
merged_df = pd.merge(training_df, t_label_df, on='id', how='outer', indicator=True)
print(merged_df.head(20))
#check for missing values in merged dataset
merged_df.isna().sum()
#Treat null values
missing_value_columns = ['funder', 'installer', 'subvillage', 'public_meeting','scheme_name', 'scheme_management', 'permit']

# Check the value counts
for col in missing_value_columns:
    print(merged_df[col].value_counts())
# Remove rows with missing values in 'funder', 'installer' and 'scheme_management' columns
merged_df.dropna(subset=['funder','installer', 'scheme_management'], axis=0, inplace=True)
Replacing the missing values for public meeting and permit with False. Assuming that the information doesnt exist. 
# Fill missing values in public meeting and permit'
for col in ['public_meeting', 'permit']:
    merged_df[col] = merged_df[col].fillna(False)

#Fill in missing values in 'scheme_name', subvillage
for col2 in ['scheme_name', 'subvillage']:
    merged_df[col2] = merged_df[col2].fillna('None')



# Confirm there are no more missing values
merged_df.isna().sum()
# Ensure the 'installation_year' is in numeric format (not datetime, just the year)
merged_df['construction_year'] = pd.to_numeric(training_df['construction_year'], errors='coerce')

# Calculate pump age
current_year = pd.Timestamp.now().year
merged_df['pump_age'] = current_year - merged_df['construction_year']

# Handle cases where installation_year might be missing or incorrect (e.g., 0 or negative values)
merged_df['pump_age'] = merged_df['pump_age'].apply(lambda x: x if x > 0 else None)
# Plotting box plots of some numerical columns
columns = ['amount_tsh', 'gps_height', 'population','pump_age']
plt.figure(figsize=(20, 10))
sns.boxplot(data=[merged_df[col] for col in columns])
plt.title("Numerical columns sample box plot", fontsize=13)
plt.ylabel("Numerical Value")
plt.xticks(range(0,4), columns)
plt.show()
# Check whether there are duplicates
merged_df.duplicated(keep = 'first').sum()

# Change the data type of public_meeting and permit columns to binary for classification
merged_df[['public_meeting', 'permit']] = merged_df[['public_meeting', 'permit']].astype(int)
# Check the new data types
merged_df.info()
## 3. Logistic Regression
- Prepare the data (ensure binary target variable)
- Create and train the model
- Make predictions on the test set
- Evaluate the model (accuracy, precision, recall, F1-score)
- Plot the ROC curve and calculate AUC
- Analyze coefficients and their significance
# Assign status_group column to y series
y = merged_df['status_group']

# Drop status_group and _merge to create X dataframe
X = merged_df.drop(['status_group','_merge'], axis=1)

# Print first 5 rows of X
X.head()
#Check categorical columns in merged set
merged_df.select_dtypes(include=['object']).columns
#check numerical columns in merged set
merged_df.select_dtypes(include=['int64', 'float64']).columns
# Create lists of categorical, numerical, and binary columns
category_column = ['funder', 'installer', 'basin', 'region', 'scheme_management', 'scheme_name',
       'extraction_type_group', 'management_group', 'payment_type',
       'water_quality', 'quantity_group', 'source_type',
       'source_class', 'waterpoint_type']

numerical_column = ['amount_tsh', 'gps_height', 'longitude', 'latitude', 'region_code',
       'district_code', 'population', 'construction_year', 'pump_age']

binary_column = ['public_meeting', 'permit']
#create dummies for categorical colums
X= pd.get_dummies(X, columns=category_column)
X

# Split the data into training and test sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize the Decision Tree model
#dt = DecisionTreeClassifier(random_state=42)

# Train the model
#dt.fit(X_train, y_train)

# Predict on the test set
#y_pred = dt.predict(X_test)

# Evaluate the model
#print("Accuracy:", accuracy_score(y_test, y_pred))
#print("Classification Report:\n", classification_report(y_test, y_pred))
#print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

