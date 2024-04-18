<!--

author:   Daniel Schwartz
email:    des338@drexel.edu
version:  0.0.0
current_version_description: Initial version
module_type: standard
docs_version: 3.0.0
language: en
narrator: UK English Female
mode: Textbook

title: Python Lesson on Regression for Machine Learning

comment:  Understand the basics of linear regression, including how it works and when to use it in Python.

long_description: Understand linear regression in Python as a powerful and versatile tool that can be used to solve a wide variety of problems. By understanding the key concepts and techniques involved in linear regression, you can build and deploy models that can accurately predict the target variable of interest.

estimated_time_in_minutes: 20

@pre_reqs

This module assumes learners have been exposed to introductory statistics, Algebra, and probability.
There are coding exercises in Python, so programming experience is required.

@end

@learning_objectives  

-   Understand the concept of linear regression and its applications in machine learning
-   Learn how to implement the linear regression algorithm in Python
-   Apply linear regression to a real-world dataset

@end

good_first_module: false
data_task: data_analysis
collection: machine_learning
coding_required: true
coding_level: basic
coding_language: python

@sets_you_up_for

@end

@depends_on_knowledge_available_in

@end

@version_history 

Previous versions: 

- [x.x.x](link): that version's current version description
- [x.x.x](link): that version's current version description
- [x.x.x](link): that version's current version description
@end

import: https://raw.githubusercontent.com/arcus/education_modules/main/_module_templates/macros.md
import: https://raw.githubusercontent.com/arcus/education_modules/pyodide_testing/_module_templates/macros_python.md
import: https://raw.githubusercontent.com/LiaTemplates/Pyodide/master/README.md
-->

# Python Lesson on Regression for Machine Learning

@overview

## What is linear regression?
- Linear regression is a supervised machine learning algorithm that learns to predict a continuous target variable based on one or more predictor variables. Linear regression models the relationship between the target variable and the predictor variables using a linear equation.
- In the case of linear regression, the target variable is a continuous variable. In a supervised learning problem, the machine learning algorithm is given a set of training data and asked to learn a function that can map the input variables to the output variable. The training data consists of pairs of input and output variables. The algorithm learns the function by finding the best fit line to the data. Once the algorithm has learned the function, it can be used to make predictions on new data. To make a prediction, the algorithm simply plugs the values of the input variables into the function.
- Linear regression stands out as a favored supervised learning algorithm due to its straightforward implementation and comprehensibility. Additionally, its versatility enables its application across diverse problem domains.



### Illustrative Toy Example in a Medical Setting

Let's consider a small dataset showing the relationship between a patient's daily medication dosage and their blood pressure:

| **Medication Dosage (mg) (X)** | **Systolic Blood Pressure (mmHg) (Y)** |
|:------------------------------:|:--------------------------------------:|
|               10               |                   140                  |
|               15               |                   130                  |
|               25               |                   115                  |
|                5               |                   150                  |
|               20               |                   120                  |


**Line of Best Fit:** After analysis, let's say we find the equation of the line of best fit to be:
$ Y = -2X + 160 $

**Prediction:**  If a doctor prescribes a dosage of 30 mg, we can use the equation to predict the patient's systolic blood pressure:
$ Y = (-2 * 30) + 160 = 100 $
Based on our model, we would predict the patient's systolic blood pressure to be approximately 100 mmHg. It's crucial to remember that in real medical scenarios, linear regression is rarely this simple. There are often many other factors influencing a patient's health outcomes. However, this example illustrates the basic concept of how linear regression can identify trends in medical data. Linear regression is a tool, but it should always be used in conjunction with a doctor's expertise and clinical judgment.




Which of the following is NOT a characteristic of linear regression?


[( )] Linear regression models the relationship between the target variable and the predictor variables using a linear equation.
[( )] Linear regression is a supervised learning algorithm.
[( )] Linear regression is a simple to implement and understand algorithm.
[(X)] Linear regression can be used to predict categorical variables.
[( )] Linear regression is a versatile algorithm that can be used to solve a variety of problems.
***
<div class = "answer">
This question presents a deeper challenge as it requires a solid understanding of linear regression's characteristics. To answer correctly, you need to identify the feature that doesn't align with linear regression. The incorrect option, "Linear regression can be used to predict categorical variables," deviates from the typical usage of linear regression, which is primarily for continuous variables. Understanding this distinction enhances your comprehension of linear regression's scope and limitations.

</div>
***


### Applications of linear regression in machine learning
Linear Regression can be used for a variety of tasks, such as: 

-   **Prediction:**  Linear regression can be used to predict a wide range of continuous variables, such as house prices, stock prices, customer churn, and medical outcomes.

### Applications of linear regression in biomedical research
Linear regression finds extensive application in biomedical research, offering insights into various domains, such as:

- **Disease prognosis:** Linear regression aids in predicting the progression of diseases based on patient demographics, biomarkers, and clinical data. For instance, it can forecast the advancement of cancer stages or the deterioration of chronic conditions like diabetes.  
    - A specific example of this in research can be found in ["A longitudinal study defined circulating microRNAs as reliable biomarkers for disease prognosis and progression in ALS human patients"](https://www.nature.com/articles/s41420-020-00397-6) In the realm of disease prognosis, longitudinal research has illuminated the potential of circulating microRNAs as dependable biomarkers for assessing disease progression and prognosis in ALS patients. By integrating patient demographics, biomarkers, and clinical data, linear regression models can be leveraged to forecast the trajectory of diseases, akin to predicting cancer stages or the progression of chronic ailments like diabetes.  

- **Treatment efficacy:** Linear regression assists in evaluating the effectiveness of medical treatments by analyzing patient response data. Researchers can utilize it to assess the impact of medications, therapies, or interventions on disease outcomes and patient well-being.  
    - ["Meta-analysis of the Age-Dependent Efficacy of Multiple Sclerosis Treatments"](https://www.frontiersin.org/journals/neurology/articles/10.3389/fneur.2017.00577/full) demonstrates a specific application of linear regression.  This study uses linear regression to determine how the effectiveness of Multiple Sclerosis treatments changes as patients age.  


- **Genetic studies:** Linear regression plays a pivotal role in genetic research by exploring associations between genetic variants and phenotypic traits. It helps identify genetic markers linked to disease susceptibility, treatment response, and disease progression, contributing to personalized medicine approaches.  
    - The article ["Prediction of Gene Expression Patterns With Generalized Linear Regression Model"](https://www.frontiersin.org/journals/genetics/articles/10.3389/fgene.2019.00120/full) describes a method using generalized linear regression to predict how gene expression levels change in response to the binding intensity of the Oct4 transcription factor. This model aids researchers in understanding the complex regulatory mechanisms behind cell reprogramming and development.  

- **Public health analysis:** Linear regression facilitates the analysis of population-level health trends, aiding in the identification of risk factors, disease clusters, and health disparities. It enables researchers to model the impact of interventions, policies, and socio-economic factors on public health outcomes.  
    - The article ["Regression Analysis for COVID-19 Infections and Deaths Based on Food Access and Health Issues"](https://www.mdpi.com/2227-9032/10/2/324) investigates the relationship between food access, pre-existing health conditions, and the severity of COVID-19 outcomes. Researchers used regression models to discover potential correlations that could inform future pandemic preparedness efforts.

- **Epidemiological modeling:** Linear regression serves as a fundamental tool in epidemiology for modeling disease spread and understanding risk factors. It assists in forecasting disease outbreaks, estimating transmission rates, and evaluating interventions' effectiveness in controlling infectious diseases.  
    - The article ["SEIR and Regression Model based COVID-19 outbreak predictions in India"](https://arxiv.org/abs/2004.00958) utilizes a combination of SEIR modeling and regression analysis to forecast COVID-19 outbreaks in India, providing valuable insights into disease spread dynamics. This approach contributes to epidemiological modeling by showcasing how linear regression, alongside SEIR models, aids in predicting disease outbreaks, estimating transmission rates, and assessing the effectiveness of interventions, thereby informing proactive measures to control infectious diseases.



By leveraging linear regression in these contexts, biomedical researchers can glean valuable insights into disease mechanisms, treatment strategies, and public health interventions, ultimately advancing healthcare practices and improving patient outcomes.



## Linear Regression Algorithm
Linear regression works by fitting a linear equation to the data.

The linear equation is represented by the following formula: 

```
y = b0 + b1 * x1 + b2 * x2 + ... + bn * xn
``` 

where:  

- `y` is the target variable
- `b0` is the bias term
- `bi` is the coefficient for the ith predictor variable
- `xi` is the ith predictor variable

The coefficients of the linear equation are estimated using the ordinary least squares (OLS) method. The OLS method minimizes the sum of the squared residuals, which are the differences between the predicted values and the actual values of the target variable. Once the linear regression model is trained, it can be used to make predictions on new data. To make a prediction, we simply plug the values of the predictor variables into the linear equation. 

<div class = "learn-more">
<b style="color: rgb(var(--color-highlight));">Learning connection</b><br>

To learn more about Linear Regression and for a visual explanation, watch [Linear Regression, Clearly Explained!!!](https://www.youtube.com/watch?v=nk2CQITm_eo).

</div>


Which of the following is NOT a component of the linear regression formula?


[( )] Target variable
[( )] Bias term
[( )] Coefficient for the ith predictor variable
[( )] ith predictor variable
[(X)] Variance of the target variable
***
<div class = "answer">

The variance of the target variable is not a component of the linear regression formula. The linear regression formula is used to predict the mean value of the target variable, not the variance.

</div>
***



### Understanding Machine Learning Techniques

Before diving into the example, it's valuable to understand some key concepts used in machine learning. These techniques help us build more accurate and reliable models for prediction.

- **Splitting Data (Training and Testing):**  Machine learning models 'learn' from data. We divide our dataset into two parts:  

    - **Training set:** This part is used to train the model, allowing it to find patterns. 
    - **Testing set:** This is held-out data used to evaluate how well our model performs on unseen examples. This prevents overfitting, where the model becomes too specific to the training data and performs poorly on new data.  

- **Recoding Categorical Predictors:** Many machine learning models work best with numerical data. Categorical features (like 'gender' or 'treatment group') need to be converted into numbers. Label encoding is a common technique, where each category is assigned a unique numerical label.  
- **Scaling Continuous Predictors:** When features have vastly different scales (e.g., age vs. body temperature), some models might be biased towards features with larger ranges. Scaling brings features into a similar range, often between 0 and 1, or standardizing them to have a mean of 0 and a standard deviation of 1. This ensures all features are treated fairly during training.  
- **Evaluating Model Predictions (MSE):** We need ways to tell how well our model is doing. Mean Squared Error (MSE) is a common metric. It calculates the average squared difference between the predicted values and the actual true values.  A lower MSE means our predictions are closer to the real targets.  

#### Why do we use these techniques?

- **Improved Accuracy:** These steps help our model identify true patterns and relationships within the data and not just memorize specific examples from the training set.  
- **Preventing Overfitting:** By testing the model on unseen data, we ensure it generalizes well to new situations.  
- **Fair Feature Influence:** Scaling makes sure no single feature dominates the model's predictions due to differences in measurement ranges.  

Let's continue with our example, keeping these concepts in mind.


    
### Python Implementation of Linear Regression

To implement linear regression in Python using Scikit-learn, we can follow these steps:

1.  Import the necessary libraries:
```
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
```

2.  Load the data:
```
# Load the data as a NumPy array
data = np.loadtxt("data.csv", delimiter=",")

# Split the data into features and target variable
X = data[:, :-1]
y = data[:, -1]
```

3.   Split the data into 80% training and 20% testing sets
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

4.  Create the linear regression model instance
```
model = LinearRegression()
```

5.  Fit the linear regression model to the training data
```
model.fit(X_train, y_train)
```

6.  Make predictions on the testing set
```
y_pred = model.predict(X_test)
```

7.  Evaluate the model using the mean squared error (MSE)
```
mse = np.mean((y_pred - y_test)**2)
```

8.  Print the MSE
```
print("MSE:", mse)
```

9.  Make predictions on new data:
```
# New data point
new_data = np.array([[1000, 3, 2]])

# Make a prediction on the new data point
y_pred = model.predict(new_data)

# Print the prediction
print("Prediction:", y_pred[0])

```

This is a basic example of how to implement linear regression in Python using Scikit-learn. There are many other ways to implement linear regression in Python, but this is a good starting point.

Here are some additional tips for implementing linear regression in Python:

-   Make sure to scale the data before training the model. This will help to ensure that all features have equal importance in the model.
-   Use a validation set to evaluate the model and tune the hyperparameters. This will help to prevent overfitting.
-   Use regularization techniques, such as L1 or L2 regularization, to prevent overfitting.
-   Interpret the coefficients of the linear regression model to understand the relationship between the predictor variables and the target variable.
    

### Applying Linear Regression to a Real-World Dataset
To apply linear regression to a real-world dataset, we can follow these steps: 

- **Choose a dataset:** The dataset should have at least one continuous target variable and one or more predictor variables. 
- **Prepare the data:** This may involve cleaning the data, handling missing values, and scaling the data. 
- **Split the data into training and testing sets:** This will help to prevent overfitting.
- **Train the linear regression model:** Use the training set to fit the model to the data. 
- **Evaluate the model on the testing set:** This will give you an estimate of how well the model will generalize to new data. 
- **Interpret the results:** Examine the coefficients of the model to understand the relationship between the predictor variables and the target variable. 
- **Make predictions on new data:** Use the trained model to make predictions on new data points.

### Important Notes
Linear regression is a powerful machine learning algorithm, but it has some limitations. Here are some of the most important limitations of linear regression: 

- **Linearity assumption:** Linear regression assumes that the relationship between the target variable and the predictor variables is linear. If the relationship is non-linear, then linear regression will not be able to accurately predict the target variable.
- **Overfitting:** Linear regression is prone to overfitting, which occurs when the model learns the training data too well and is unable to generalize to new data. Overfitting can be prevented by using regularization techniques such as L1 or L2 regularization.
- **Outliers:** Linear regression is sensitive to outliers, which are data points that are significantly different from the rest of the data. Outliers can have a large impact on the parameters of the linear regression model and can lead to inaccurate predictions.
- **Collinearity:** Linear regression is also sensitive to collinearity, which occurs when two or more predictor variables are highly correlated with each other. Collinearity can make it difficult to interpret the results of the linear regression model and can lead to inaccurate predictions.

[True/False] Linear regression is sensitive to collinearity.


[(X)] True
[( )] False
***
<div class = "answer">

This question is designed to test the test-taker's understanding of the concept of collinearity and its impact on linear regression models. Collinearity is a serious problem in linear regression because it can make it difficult to interpret the results of the model and can lead to inaccurate predictions.

</div>
***


[True/False] Overfitting can be prevented by using regularization techniques.


[(X)] True
[( )] False
***
<div class = "answer">

This question is designed to test the test-taker's understanding of the concept of overfitting and how to prevent it. Overfitting is a common problem in machine learning, and it is important to be able to identify and prevent it. Regularization techniques such as L1 and L2 regularization can be used to prevent overfitting in linear regression models.

</div>
***





### Real World Code Example

The Streptomycin for Tuberculosis dataset originates from a groundbreaking clinical trial published in 1948, often recognized as the first modern randomized clinical trial. It comprises data from a prospective, randomized, placebo-controlled study investigating the efficacy of streptomycin treatment for pulmonary tuberculosis. The dataset includes variables such as participant ID, study arm (Streptomycin or Control), doses of Streptomycin and Para-Amino-Salicylate in grams, gender, baseline conditions (categorized as good, fair, or poor), oral temperature at baseline, erythrocyte sedimentation rate at baseline, presence of lung cavitation on chest X-ray at baseline, streptomycin resistance at 6 months, radiologic outcomes at 6 months, numeric rating of chest X-ray at month 6, and a dichotomous outcome indicating improvement. These variables provide comprehensive information for analyzing the effectiveness of streptomycin treatment for tuberculosis, allowing for various statistical analyses such as logistic regression modeling.



1.  Install Packages:
```python @Pyodide.exec

import pandas as pd
import io
from pyodide.http import open_url
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error





```

2.  Load the data:
```python
# Load dataset and read to pandas dataframe
url = "https://github.com/arcus/education_modules/linear_regression/python_linear_regression/data/strep_tb.csv"

url_contents = open_url(url)
text = url_contents.read()
file = io.StringIO(text)
df = pd.read_csv(file)

# Analyze data and features
df.info()

# SECTION 1: Data Loading and Preparation
df = pd.read_csv('./data/strep_tb.csv')  # Load the data from a CSV file

# SECTION 2: Encoding Categorical Features
categorical_cols = [
    'arm', 'gender', 'baseline_condition', 'baseline_temp',
    'baseline_esr', 'baseline_cavitation', 'strep_resistance', 'radiologic_6m'
]
 # Create a LabelEncoder for transforming columns
le = LabelEncoder() 

# Apply label encoding to each categorical column
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Select features for clustering
features = ['age', 'baseline', 'number3m', 'number12m']
X = df[features]

# Fill missing values with the mean of each column
X.fillna(X.mean(), inplace=True)

# Standardize the feature values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```
@Pyodide.eval


3.  Compute Regression:
```python
# Feature Selection and Target Definition
features = [
    'patient_id', 'arm', 'dose_strep_g', 'dose_PAS_g', 'gender',
    'baseline_condition', 'baseline_temp', 'baseline_esr',
    'baseline_cavitation', 'strep_resistance', 'radiologic_6m', 'rad_num'
]
target = 'improved'

# Separate inputs (features) and output (target variable)
inputs = df[features]
output = df[target]

# SECTION 4: Linear Regression Modeling
model = LinearRegression()  # Create the regression model
model.fit(inputs, output)   # Train the model 
```
@Pyodide.eval


4.  Evaluate Model:
```python
# Predict data
predictions = model.predict(inputs) 

# Analyze predictions
print('R-squared:', r2_score(output, predictions))  
print('Mean Squared Error:', mean_squared_error(output, predictions))  
```
@Pyodide.eval

If the K-Means algorithm identified distinct clusters with minimal overlap, it suggests there might be three underlying patient groups regarding polyp count progression:

- **Cluster 1 (Low Progression):** This cluster might represent participants who have a relatively low number of polyps at 3 months and a stable or slightly increased number at 12 months. This could be associated with effective treatment or naturally slow polyp growth.
- **Cluster 2 (Moderate Progression):** This cluster could include participants with a moderate number of polyps at 3 months and a somewhat steeper increase by 12 months. This might indicate a less effective treatment or a faster natural growth rate for polyps.
- **Cluster 3 (High Progression):** This cluster might contain participants with a high number of polyps at 3 months and a substantial increase by 12 months. This could be linked to factors like a particularly aggressive polyp type or treatment resistance.

**While clustering provides valuable insights into potential patient subgroups, further analysis of treatment effects and other relevant features is necessary to fully understand the underlying factors influencing polyp count progression.**

    


## Conclusion

By the end of this module, you'll have gained a solid grasp of linear regression and its practical implementation in Python. You'll be equipped to apply linear regression techniques to real-world datasets, enabling you to make predictions and uncover valuable insights. With this knowledge, you'll be well-prepared to embark on your journey into the world of data analysis and machine learning.


## Additional Resources

## Feedback

@feedback
