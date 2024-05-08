# Statistical_Learning
From Browsing to Buying: Predicting Purchase Intentions in Online Shoppers

### PROBLEM SETTING & DEFINITION:
 
In today's digital era, e-commerce has transformed consumer behavior, necessitating businesses to predict user purchasing patterns for better engagement and sales optimization. Machine learning (ML) algorithms offer a potent tool to analyze vast datasets and unveil patterns that shape consumer decisions.
 
The e-commerce sector, characterized by its rapid growth, faces the challenge of discerning whether users are inclined to make purchases. Understanding customer behavior hinges on factors like demographics, browsing history, and external influences such as promotions.
 
This project's significance lies in revolutionizing how businesses interact with customers. Machine Learning techniques enable companies to anticipate user preferences, personalize offerings, and optimize marketing efforts, ultimately enhancing customer satisfaction and loyalty. Moreover, data-driven insights empower businesses to maintain competitiveness by staying ahead of market trends.
 
 
### GOAL:
 
The primary goal of this project is to develop a robust machine-learning model capable of accurately predicting whether a user is likely to purchase a product or not. This predictive capability enables businesses to optimize their marketing strategies, personalize user experiences, and ultimately increase sales conversion rates. Additionally, the project aims to provide valuable insights into consumer behavior, empowering businesses to make data-driven decisions and stay competitive in the rapidly evolving e-commerce landscape.
 
### DATASET:  
 
We have chosen our dataset from the UCI repository as follows: https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset
 
### ATTRIBUTE INFORMATION:
 
This dataset, which has 18 features and 12,330 observations, is believed to be the greatest because it contains so many highly valuable indicators like Revenue and Bounce Rate, Exit Rate, Page Value
 
1. Administrative: Counts the number of administrative pages visited by the user during the session.
2. ⁠Administrative Duration: Total time spent by the user on administrative pages during the session.
3. Informational: Number of informational pages visited by the user in the session.
4. ⁠Informational Duration: Total time spent by the user on informational pages during the session.
5. ⁠Product Related: Counts the product-related pages visited by the user during the session.
6. Product Related Duration: Total time spent by the user on product-related pages during the session.
7. ⁠Bounce Rate: Percentage of visitors who leave the site after viewing only the landing page, without any further interaction.
8. ⁠Exit Rate: For all pageviews to the page, the percentage that were the last in the session.
9. Page Value: Average value of a page that a user visited before completing an ecommerce transaction.
10. ⁠Special Day: Indicates the closeness of the site visit to a specific special day which may influence purchase decisions.
11. Operating System: The operating system used by the visitor.
12. Browser: The browser used by the visitor.
13. Region: The geographical region from where the session originated.
14. Traffic Type: The type of traffic that led the visitor to the website.
15. ⁠Visitor Type: Indicates whether the visitor is new or returning.
16. Weekend: A Boolean value indicating whether the visit occurred on a weekend.
17. Month: The month of the year in which the session occurred.  
18. Revenue : Binary target feature (Label)
 
### METHOD OF IMPLEMENTATION:
 
The online shoppers purchasing intention project is going to be started with the data preprocessing to address missing / inappropriate data by either deleting them or filling them with averages or medians and then normalizing the numerical values for consistency. Exploratory Data Analytics is going to be conducted on preprocessed data to examine the data and identify any outliers if exists which helps in understanding the datasets attributes better. After this, we choose key features based on insights from EDA using methods like correlation analysis to identify and eliminate redundant features.  The objective of the project is then implemented using Machine Learning Classification techniques such as logistic regression, Decision Trees, Naïve Bayes, SVM etc. to classify online customer conversion.
 ``` 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import RobustScaler

from imblearn.over_sampling import SMOTE

from dataclasses import dataclass

from scipy import optimize
from scipy.optimize import Bounds

from tqdm import tqdm

import sys

from scipy.stats import chi2_contingency

import random

from sklearn.metrics import roc_curve, roc_auc_score, auc 
```
```
#EDA
df.info()
df.describe()
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 12330 entries, 0 to 12329
Data columns (total 18 columns):
 #   Column                   Non-Null Count  Dtype  
---  ------                   --------------  -----  
 0   Administrative           12330 non-null  int64  
 1   Administrative_Duration  12330 non-null  float64
 2   Informational            12330 non-null  int64  
 3   Informational_Duration   12330 non-null  float64
 4   ProductRelated           12330 non-null  int64  
 5   ProductRelated_Duration  12330 non-null  float64
 6   BounceRates              12330 non-null  float64
 7   ExitRates                12330 non-null  float64
 8   PageValues               12330 non-null  float64
 9   SpecialDay               12330 non-null  float64
 10  Month                    12330 non-null  object 
 11  OperatingSystems         12330 non-null  int64  
 12  Browser                  12330 non-null  int64  
 13  Region                   12330 non-null  int64  
 14  TrafficType              12330 non-null  int64  
 15  VisitorType              12330 non-null  object 
 16  Weekend                  12330 non-null  bool   
 17  Revenue                  12330 non-null  bool   
dtypes: bool(2), float64(7), int64(7), object(2)
memory usage: 1.5+ MB
```
# Check for null values in the data dictionary
print(df.isnull().sum())
Administrative             0
Administrative_Duration    0
Informational              0
Informational_Duration     0
ProductRelated             0
ProductRelated_Duration    0
BounceRates                0
ExitRates                  0
PageValues                 0
SpecialDay                 0
Month                      0
OperatingSystems           0
Browser                    0
Region                     0
TrafficType                0
VisitorType                0
Weekend                    0
Revenue                    0
dtype: int64
```
###Exploring Features
#Since data has numerical and categorical columns, exploring them separately
numerical_cols = df.select_dtypes(exclude=['object', 'bool']).columns.tolist()
```
# Creating a histogram for each numerical column
for col in numerical_cols:
    plt.figure(figsize=(4, 3))  # Set the figure size for better visibility
    df[col].hist(bins=30)  # You can adjust the number of bins based on your data distribution
    plt.title(f'Histogram of {col}')  # Set title with column name
    plt.xlabel(col)  # Set x-axis label to column name
    plt.ylabel('Frequency')  # Set y-axis label to Frequency
    plt.grid(False)  
    plt.show()
```
#Since data has numerical and categorical columns, exploring them separately
numerical_cols = df.select_dtypes(exclude=['object', 'bool']).columns.tolist()

# Creating a histogram for each numerical column
for col in numerical_cols:
    plt.figure(figsize=(4, 3))  # Set the figure size for better visibility
    df[col].hist(bins=30)  # You can adjust the number of bins based on your data distribution
    plt.title(f'Histogram of {col}')  # Set title with column name
    plt.xlabel(col)  # Set x-axis label to column name
    plt.ylabel('Frequency')  # Set y-axis label to Frequency
    plt.grid(False)  
    plt.show()
