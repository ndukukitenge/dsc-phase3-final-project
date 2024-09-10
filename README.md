# Phase 3 Project Tanzania Wells 

### Outline
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


## Business Problem

The goal of this project is to predict which pumps are functional, which need some repairs and which dont work at all. 

The challenge from DrivenData. (2015). Pump it Up: Data Mining the Water Table. Retrieved [Month Day Year] from https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table  with data from Taarifa and the Tanzanian Ministry of Water. The goal of this project is to predict one of these three classes based on a number of variables about what kind of pump is operating, when it was installed, and how it is managed. A smart understanding of which waterpoints will fail can improve maintenance operations and ensure that clean, potable water is available to communities across Tanzania.


## 1. Project Setup
- Import necessary libraries (pandas, numpy, sklearn, matplotlib, seaborn)
- Load the dataset

## 2. Data Exploration and Preprocessing
- Examine the dataset (head, info, describe)
- Check for missing values and handle them
- Explore data distributions and correlations
- Perform feature engineering eg. add pump age
- Split the data into features (X) and target variable(s) (y)

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

I found that the outliers were minimal and opted not to treat them and dropped duplicate amounts.

## 3. Logistic Regression
- Prepare the data (ensure binary target variable)
- Create and train the model
- Make predictions on the test set
- Evaluate the model (accuracy, precision, recall, F1-score)
- Plot the ROC curve and calculate AUC
- Analyze coefficients and their significance

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

