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

## 4. Decision Trees

- Prepare the data
- Create and train the model
- Make predictions on the test set
- Evaluate the model (accuracy, precision, recall, F1-score)
- Visualize the tree structure
- Analyze feature importance

## 5. Model Comparison

- Compare performance metrics across all models
- Discuss strengths and weaknesses of each approach
- Recommend the best model(s) for the problem at hand

## 6. Conclusion and Future Work
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
from sklearn.feature_selection import RFE

#Load training dataset
sub_merged_df = pd.read_csv('./SubmissionFormat.csv',index_col=0)
training_merged_df = pd.read_csv('./TrainingSetValues.csv',index_col=0)
test_merged_df = pd.read_csv('./TestSetValues.csv',index_col=0)
t_label_merged_df = pd.read_csv('TrainingSetLabels.csv',index_col=0)



## 2. Data Exploration and Preprocessing
- Examine the dataset (head, info, describe)
- Check for missing values and handle them
- Explore data distributions and correlations
- Perform feature engineering eg. add pump age
- Split the data into features (X) and target variable(s) (y)
# Function to display dataset info
def display_dataset_info(merged_df, name):
    print(f"\n=== {name} ===")
    print(f"Shape: {merged_df.shape}")
    print("\nInfo:")
    print(merged_df.info())
    print("\nDescription:")
    print(merged_df.describe())
    print("\nHead:")
    print(merged_df.head())
    print("\n" + "="*40)

# Display info for each dataset
display_dataset_info(sub_merged_df, "Submission Format")
display_dataset_info(training_merged_df, "Training Set Values")
display_dataset_info(test_merged_df, "Test Set Values")
display_dataset_info(t_label_merged_df, "Training Set Labels")
#check for missing values in training dataset
training_merged_df.isna().sum()


## Dropping columns and reasons why:
- Scheme name - Too many null values 
- wpt_name - Too many unique values
- management - management group covers this
- quality group - quality covers this
- quantity group
- extracton type group 

Numerical columns:
- num private - dont know what this means. 

Observations
- Funder and installer are similar, but funder has less null values than installer. 
- Waterpoint type and water point are similar - Will drop waterpoint type. 


#Check categorical columns
training_merged_df.select_dtypes(include=['object']).columns
#check numerical columns
training_merged_df.select_dtypes(include=['int64', 'float64']).columns
#Define the list of columns to drop
columns_drop = ['date_recorded', 'funder', 'wpt_name', 'subvillage', 'lga', 
 'ward', 'recorded_by', 'scheme_name', 'extraction_type', 
 'extraction_type_group', 'management', 'payment', 'quality_group', 
 'quantity', 'source', 'source_type', 'waterpoint_type', 'num_private', 
 'region_code', 'district_code']

# Drop unnecessary columns
training_merged_df = training_merged_df.drop(columns_drop,axis=1)

#Print remaining columns
print(training_merged_df.columns)
#explore data distrubitions and correlations of the numerical- training data
corr_matrix = training_merged_df.corr()

# Plot the correlation matrix using Seaborn for numerical values.
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True, linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()
# merge with training labels data using ID as connector
merged_df = pd.merge(training_merged_df, t_label_merged_df, on='id', how='outer', indicator=True)
print(merged_df.head(20))
#check for missing values in merged dataset
merged_df.isna().sum()
#Treat null values
missing_value_columns = ['installer', 'public_meeting', 'scheme_management', 'permit']

# Check the value counts
for col in missing_value_columns:
    print(merged_df[col].value_counts())
# Remove rows with missing values in 'funder', 'installer' and 'scheme_management' columns
merged_df.dropna(subset=['installer', 'scheme_management'], axis=0, inplace=True)
Replacing the missing values for public meeting and permit with False. Assuming that the information doesnt exist. 
# Fill missing values in public meeting and permit'
for col in ['public_meeting', 'permit']:
    merged_df[col] = merged_df[col].fillna(False)



Remove dimensionality of unique values in installer column:
# Replace close variations and misspellings in the installer column

merged_df['installer'] = merged_df['installer'].replace(to_replace = ('Central government', 'Tanzania Government',
                                          'Cental Government','Tanzania government','Cebtral Government', 
                                          'Centra Government', 'central government', 'CENTRAL GOVERNMENT', 
                                          'TANZANIA GOVERNMENT', 'TANZANIAN GOVERNMENT', 'Central govt', 
                                          'Centr', 'Centra govt', 'Tanzanian Government', 'Tanzania', 
                                          'Tanz', 'Tanza', 'GOVERNMENT', 
                                          'GOVER', 'GOVERNME', 'GOVERM', 'GOVERN', 'Gover', 'Gove', 
                                          'Governme', 'Governmen', 'Got', 'Serikali', 'Serikari', 'Government',
                                          'Central Government'), 
                                          value = 'Central Government')

merged_df['installer'] = merged_df['installer'].replace(to_replace = ('IDARA', 'Idara ya maji', 'MINISTRY OF WATER',
                                          'Ministry of water', 'Ministry of water engineer', 'MINISTRYOF WATER', 
                                          'MWE &', 'MWE', 'Wizara ya maji', 'WIZARA', 'wizara ya maji',
                                          'Ministry of Water'), 
                                          value ='Ministry of Water')

merged_df['installer'] = merged_df['installer'].replace(to_replace = ('District COUNCIL', 'DISTRICT COUNCIL',
                                          'Counc','District council','District Counci', 
                                          'Council', 'COUN', 'Distri', 'Halmashauri ya wilaya',
                                          'Halmashauri wilaya', 'District Council'), 
                                          value = 'District  Council')

merged_df['installer'] = merged_df['installer'].replace(to_replace = ('District water depar', 'District Water Department', 
                                          'District water department', 'Distric Water Department'),
                                          value = 'District Water Department')

merged_df['installer'] = merged_df['installer'].replace(to_replace = ('villigers', 'villager', 'villagers', 'Villa', 'Village',
                                          'Villi', 'Village Council', 'Village Counil', 'Villages', 'Vill', 
                                          'Village community', 'Villaers', 'Village Community', 'Villag',
                                          'Villege Council', 'Village council', 'Villege Council', 'Villagerd', 
                                          'Villager', 'VILLAGER', 'Villagers',  'Villagerd', 'Village Technician', 
                                          'Village water attendant', 'Village Office', 'VILLAGE COUNCIL',
                                          'VILLAGE COUNCIL .ODA', 'VILLAGE COUNCIL Orpha', 'Village community members', 
                                          'VILLAG', 'VILLAGE', 'Village Government', 'Village government', 
                                          'Village Govt', 'Village govt', 'VILLAGERS', 'VILLAGE WATER COMMISSION',
                                          'Village water committee', 'Commu', 'Communit', 'commu', 'COMMU', 'COMMUNITY', 
                                           'Comunity', 'Communit', 'Kijiji', 'Serikali ya kijiji', 'Community'), 
                                          value ='Community')

merged_df['installer'] = merged_df['installer'].replace(to_replace = ('FinW', 'Fini water', 'FINI WATER', 'FIN WATER',
                                          'Finwater', 'FINN WATER', 'FinW', 'FW', 'FinWater', 'FiNI WATER', 
                                          'FinWate', 'FINLAND', 'Fin Water', 'Finland Government'), 
                                          value ='Finnish Government')

merged_df['installer'] = merged_df['installer'].replace(to_replace = ('RC CHURCH', 'RC Churc', 'RC', 'RC Ch', 'RC C', 'RC CH',
                                          'RC church', 'RC CATHORIC', 'Roman Church', 'Roman Catholic',
                                          'Roman catholic', 'Roman Ca', 'Roman', 'Romam', 'Roma', 
                                          'ROMAN CATHOLIC', 'Kanisa', 'Kanisa katoliki'), 
                                          value ='Roman Catholic Church')

merged_df['installer'] = merged_df['installer'].replace(to_replace = ('Dmdd', 'DMDD'), value ='DMDD') 

merged_df['installer'] = merged_df['installer'].replace(to_replace = ('TASA', 'Tasaf', 'TASAF 1', 'TASAF/', 'TASF',
                                          'TASSAF', 'TASAF'), value ='TASAF') 

merged_df['installer'] = merged_df['installer'].replace(to_replace = ('RW', 'RWE'), value ='RWE')

merged_df['installer'] = merged_df['installer'].replace(to_replace = ('SEMA CO LTD', 'SEMA Consultant', 'SEMA'), value ='SEMA')

merged_df['installer'] = merged_df['installer'].replace(to_replace = ('DW E', 'DW#', 'DW$', 'DWE&', 'DWE/', 'DWE}', 
                                         'DWEB', 'DWE'), value ='DWE')

merged_df['installer'] = merged_df['installer'].replace(to_replace = ('No', 'NORA', 'Norad', 'NORAD/', 'NORAD'), 
                                          value ='NORAD') 

merged_df['installer'] = merged_df['installer'].replace(to_replace = ('Ox', 'OXFARM', 'OXFAM'), value ='OXFAM') 

merged_df['installer'] = merged_df['installer'].replace(to_replace = ('PRIV', 'Priva', 'Privat', 'private', 'Private company',
                                          'Private individuals', 'PRIVATE INSTITUTIONS', 'Private owned',
                                          'Private person', 'Private Technician', 'Private'), 
                                          value ='Private') 

merged_df['installer'] = merged_df['installer'].replace(to_replace = ('Ch', 'CH', 'Chiko', 'CHINA', 'China',
                                            'China Goverment'), value ='Chinese Goverment')

merged_df['installer'] = merged_df['installer'].replace(to_replace = ('Unisef','Unicef', 'UNICEF'), value ='UNICEF')
                                          
merged_df['installer'] = merged_df['installer'].replace(to_replace = ('Wedeco','WEDEKO', 'WEDECO'), value ='WEDECO')

merged_df['installer'] = merged_df['installer'].replace(to_replace = ('Wo','WB', 'Word Bank', 'Word bank', 'WBK',
                                          'WORDL BANK', 'World', 'world', 'WORLD BANK', 'World bank',
                                          'world banks', 'World banks', 'WOULD BANK', 'World Bank'), 
                                          value ='World Bank')
                                          
merged_df['installer'] = merged_df['installer'].replace(to_replace = ('Lga', 'LGA'), value ='LGA')

merged_df['installer'] = merged_df['installer'].replace(to_replace = ('World Division', 'World Visiin', 
                                         'World vision', 'WORLD VISION', 'world vision', 'World Vission', 
                                          'World Vision'), 
                                          value ='World Vision')

merged_df['installer'] = merged_df['installer'].replace(to_replace = ('Local', 'Local technician', 'Local  technician',
                                         'local  technician', 'LOCAL CONTRACT', 'local fundi', 
                                         'Local l technician', 'Local te', 'Local technical', 'Local technical tec',
                                         'local technical tec', 'local technician', 'Local technitian',
                                         'local technitian', 'Locall technician', 'Localtechnician',
                                         'Local Contractor'), 
                                          value ='Local Contractor')
                                          
merged_df['installer'] = merged_df['installer'].replace(to_replace = ('DANID', 'DANNY', 'DANNIDA', 'DANIDS', 
                                         'DANIDA CO', 'DANID', 'Danid', 'DANIAD', 'Danda', 'DA',
                                         'DENISH', 'DANIDA'), 
                                          value ='DANIDA')

merged_df['installer'] = merged_df['installer'].replace(to_replace =('Adrs', 'Adra', 'ADRA'), value ='ADRA')
                                          
merged_df['installer'] = merged_df['installer'].replace(to_replace = ('Hesawa', 'hesawa', 'HESAW', 'hesaw',
                                          'HESAWQ', 'HESAWS', 'HESAWZ', 'hesawz', 'hesewa', 'HSW',
                                          'HESAWA'),
                                          value ='HESAWA')

merged_df['installer'] = merged_df['installer'].replace(to_replace = ('Jaica', 'JAICA', 'Jica', 'Jeica', 'JAICA CO', 'JALCA',
                                          'Japan', 'JAPAN', 'JAPAN EMBASSY', 'Japan Government', 'Jicks',
                                          'JIKA', 'jika', 'jiks', 'Embasy of Japan in Tanzania', 'JICA'), 
                                          value ='JICA')

merged_df['installer'] = merged_df['installer'].replace(to_replace = ('KKT', 'KK', 'KKKT Church', 'KkKT', 'KKT C',
                                          'KKKT'), value ='KKKT')

merged_df['installer'] = merged_df['installer'].replace(to_replace = ('0', 'Not Known', 'not known', 'Not kno'), value ='Unknown')

# Retain top 20 installers as unique entries
top_20_installers = merged_df['installer'].value_counts(normalize=True).head(20).index.tolist()  


merged_df['installer'] = [value if value in top_20_installers else "OTHER" for value in merged_df['installer']]
# Confirm there are no more missing values
merged_df.isna().sum()
# Ensure the 'installation_year' is in numeric format (not datetime, just the year)
merged_df['construction_year'] = pd.to_numeric(training_merged_df['construction_year'], errors='coerce')

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
In the above I found that the outliers were minimal and opted not to treat them.
# Check whether there are duplicates
merged_df.duplicated(keep = 'first').sum()

# Change the data type of public_meeting and permit columns to binary for classification
merged_df[['public_meeting', 'permit']] = merged_df[['public_meeting', 'permit']].astype(int)
# Check the new data types
merged_df.info()
Explore data distributions:
# Plot distribution of target variable.
fig, ax = plt.subplots(figsize=(5, 3))
sns.countplot(merged_df['status_group'])
x_labels = merged_df['status_group'].unique()

# Add labels
plt.title('Countplot of Predictions')
plt.xlabel('Pump Function')
ax.set_xticklabels(x_labels, fontsize=9)
plt.ylabel('Predictions')
plt.show()
# Histogram of continuous variables
continuous = ['amount_tsh','gps_height','longitude','latitude','population','construction_year','pump_age']
fig = plt.figure(figsize=(10, 10))
for i, col in enumerate(continuous):
    ax = plt.subplot(4, 4, i+1)
    merged_df[col].plot(kind='hist', ax=ax, title=col)
plt.tight_layout()
plt.show()
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
category_column = ['installer', 'basin', 'region', 'scheme_management',
       'extraction_type_class', 'management_group', 'payment_type',
       'water_quality', 'quantity_group', 'source_class',
       'waterpoint_type_group',]

numerical_column = ['amount_tsh', 'gps_height', 'longitude', 'latitude', 'population',
       'construction_year', 'pump_age']

binary_column = ['public_meeting', 'permit']
#create dummies for categorical colums
X= pd.get_dummies(X, columns=category_column)
X
# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Logistic Regression model
log_reg = LogisticRegression(max_iter=1000)

# Train the model
log_reg.fit(X_train, y_train)

# Predict on the test set
y_pred_logreg = log_reg.predict(X_test)
# Plotting Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_logreg), annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred_logreg)
print(f"Accuracy: {accuracy}")
print("Classification Report:\n", classification_report(y_test, y_pred_logreg))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_logreg))


I interpret the results of the logistic regression as follows:
## 1. Overall Accuracy:
   The model correctly classifies about 70.8% of the cases in the test set, which is moderately good but shows room for improvement. 

## 2. Class-wise Performance:
   - **Functional (Majority Class)**:
     - **Precision**: 0.70 – When the model predicts a pump as "functional," it is correct 70% of the time.
     - **Recall**: 0.87 – The model captures 87% of all actual "functional" pumps.
     - **F1-Score**: 0.77 – This represents a balance between precision and recall, indicating good performance for this class.
   - **Functional Needs Repair (Minority Class)**:
     - **Precision**: 0.42 – Only 42% of the time when the model predicts "functional needs repair" is it correct.
     - **Recall**: 0.01 – The model only captures 1% of actual "needs repair" cases. This shows a major issue in identifying pumps that need repair.
     - **F1-Score**: 0.01 – A very low score, suggesting the model is poor at distinguishing this class.
   - **Non-functional**:
     - **Precision**: 0.74 – 74% of the predicted "non-functional" pumps are correctly classified.
     - **Recall**: 0.61 – The model identifies 61% of actual "non-functional" pumps.
     - **F1-Score**: 0.67 – Indicates decent performance but lower recall.

## 3. Confusion Matrix:
   - **Functional**: The model correctly classified 4901 out of 5620 "functional" pumps, but misclassified 716 as "non-functional."
   - **Functional Needs Repair**: The model struggles heavily here, correctly identifying only 5 out of 743 pumps that need repair, misclassifying most as either "functional" (580) or "non-functional" (158).
   - **Non-functional**: The model correctly identifies 2449 of the 4023 non-functional pumps, but misclassifies 1570 as "functional."

### Insights:
   - The model does well at predicting "functional" and "non-functional" pumps but performs very poorly at identifying pumps that need repair.
   - The **class imbalance** (with the "functional needs repair" class being underrepresented) is likely contributing to the poor performance on this class.
   - Since recall for "functional needs repair" is very low, this indicates that most pumps requiring repair are misclassified as either "functional" or "non-functional."

## 4. Decision Trees

- Prepare the data
- Create and train the model
- Make predictions on the test set
- Evaluate the model (accuracy, precision, recall, F1-score)
- Visualize the tree structure
- Analyze feature importance



# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize the Decision Tree model
dt = DecisionTreeClassifier(random_state=42)

# Train the model
dt.fit(X_train, y_train)

# Predict on the test set
y_pred_dt = dt.predict(X_test)

# Generate the confusion matrix
cm_dt = confusion_matrix(y_test, y_pred_dt)

# Plot the confusion matrix using seaborn heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm_dt, annot=True, fmt="d", cmap="Blues", cbar=False, 
            xticklabels=dt.classes_, yticklabels=dt.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Classification Report:\n", classification_report(y_test, y_pred_dt))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))
### Interpretation of findings from decision tree: 
## 1. Overall Accuracy:
   - **Accuracy**: 0.7531 (75.31%) means that 75.31% of the water pumps were correctly classified as either functional, needing repair, or non-functional. While this is a decent result, it can still improve, particularly in distinguishing between the different classes.

## 2. Precision, Recall, and F1-Score:

   - **Functional Pumps**:
     - **Precision**: 0.79 (79%) means that, when the model predicted a pump to be functional, 79% of the time, it was correct.
     - **Recall**: 0.80 (80%) means that the model correctly identified 80% of the truly functional pumps.
     - **F1-Score**: 0.80 indicates a good balance between precision and recall, reflecting strong performance for this class.
   
   - **Functional but Needs Repair**:
     - **Precision**: 0.37 (37%) shows that when the model predicted a pump to need repairs, it was only correct 37% of the time. This indicates a high number of false positives for this class.
     - **Recall**: 0.36 (36%) means that the model only identified 35% of the pumps that actually need repair.
     - **F1-Score**: 0.36 indicates weak performance, as the model struggles to accurately classify pumps that need repair. This suggests that this class is the most challenging for the model to predict.

   - **Non-Functional Pumps**:
     - **Precision**: 0.77 (77%) means that when the model predicted a pump was non-functional, 77% of the predictions were correct.
     - **Recall**: 0.76 (76%) shows that the model correctly identified 77% of the truly non-functional pumps.
     - **F1-Score**: 0.77 indicates solid performance for this class, similar to the functional pumps.

## 3. Macro and Weighted Averages:
   - **Macro Average**: These metrics are the unweighted averages across all classes:
     - **Precision**: 0.64 (64%)
     - **Recall**: 0.64 (64%)
     - **F1-Score**: 0.64 (64%)
     - These lower values indicate that the model performs worse on minority classes, particularly on pumps that need repair.
   
   - **Weighted Average**: These metrics are weighted by the number of samples in each class:
     - **Precision**: 0.75 (75%)
     - **Recall**: 0.75 (75%)
     - **F1-Score**: 0.75 (75%)
     - These values are higher since the model performs well on the majority class (`functional`), which has a larger representation in the dataset.

## 4. Confusion Matrix:
   - **Functional Pumps**: O4,490 pumps were correctly classified as functional, but 331 were misclassified as "functional needs repair" and 799 as "non-functional."
   - **Functional but Needs Repair**: Only 270 were correctly classified, with 352 misclassified as functional and 121 as non-functional. This class has the highest misclassification rate, indicating difficulty in distinguishing between functional needs repair and the other two categories.
   - **Non-Functional Pumps**: 3,062 pumps were correctly identified as non-functional, but 824 were incorrectly classified as functional, and 137 as needing repair.

## Key Insights:
1. **Strength**: The model performs well on the majority classes (`functional` and `non-functional` pumps) .
   
2. **Weakness**: The model struggles significantly with the `functional needs repair` class. This indicates that the model has difficulty distinguishing pumps that require minor repairs from those that are functional or non-functional.



## 5. Model Comparison

- Compare performance metrics across all models
- Discuss strengths and weaknesses of each approach
- Recommend the best model(s) for the problem at hand


# Define the data for the table
data = {
    'Metric': ['Accuracy', 'Functional Precision', 'Functional Recall', 'Functional F1-Score',
               'Functional Needs Repair Precision', 'Functional Needs Repair Recall', 'Functional Needs Repair F1-Score',
               'Non-functional Precision', 'Non-functional Recall', 'Non-functional F1-Score'],
    'Decision Tree': [0.757, 0.79, 0.80, 0.80, 
                      0.37, 0.35, 0.36, 
                      0.77, 0.77, 0.77],
    'Logistic Regression': [0.708, 0.70, 0.87, 0.77, 
                            0.42, 0.01, 0.01, 
                            0.74, 0.61, 0.67]
}

# Create the DataFrame
results_df = pd.DataFrame(data)

# Display the DataFrame
results_df

When we compare the results of the Logistic Regression and Decision Tree models. We find the following: 

1. **Accuracy**:
   - The Decision Tree (75.8%) outperforms Logistic Regression (70.8%) in terms of accuracy.

2. **Class Performance**:
   - **Functional Pumps**:
     - Both models perform reasonably well, but the Decision Tree model has a slightly better balance between precision and recall, with a higher F1-score.
   - **Functional Needs Repair**:
     - Both models struggle, but the Decision Tree performs better, with a recall of 0.35 compared to just 0.01 for Logistic Regression.
     - Logistic Regression is almost entirely ineffective at identifying "needs repair" pumps, whereas the Decision Tree, while still underperforming, does capture a small portion.
   - **Non-functional Pumps**:
     - The Decision Tree is also better at identifying "non-functional" pumps, with both higher recall and F1-score.

3. **Confusion Matrix**:
   - The Decision Tree distributes misclassifications more effectively across classes, particularly in the minority class ("needs repair").
   - Logistic Regression over-classifies pumps as functional and fails to capture "needs repair" instances.

## Strengths & Weaknesses:
- **Decision Tree**:
  - **Strengths**:
    - Performs better at handling class imbalance.
    - More interpretable for understanding complex decision boundaries.
    - Handles non-linear relationships well.
  - **Weaknesses**:
    - Prone to overfitting, especially with noisy data or small datasets.
    - May require tuning to avoid over-complex models.

- **Logistic Regression**:
  - **Strengths**:
    - Simpler and computationally efficient.
    - Well-suited for linearly separable data.
  - **Weaknesses**:
    - Struggles with imbalanced data, especially in the "needs repair" class.
    - Assumes linear relationships between features and target, which may not fit this dataset.

## Recommendation:
Given the results, the **Decision Tree** is the better model for this problem, particularly because:
- It outperforms Logistic Regression in overall accuracy.
- It handles the "functional needs repair" class significantly better.
- It captures more complex, non-linear relationships, which are likely present in the dataset.

## Possible Next Steps:
   1. **Addressing Class Imbalance**: Techniques like oversampling the "functional needs repair" class, undersampling the "functional" class, or using a more balanced metric such as the F1-score might help improve the model’s ability to correctly identify pumps needing repair.
   2. **Feature Engineering**: I would explore creating more features or transforming existing ones to improve model performance.
   3. **Try Other Models**: I would consider using more complex models like Random Forest or Gradient Boosting to improve classification performance, particularly for the minority class.

