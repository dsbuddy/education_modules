import pandas
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# SECTION 1: Data Loading and Preparation
df = pandas.read_csv('./data/strep_tb.csv')  # Load the data from a CSV file

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

# SECTION 3: Feature Selection and Target Definition
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

# SECTION 5: Model Evaluation
predictions = model.predict(inputs) 
print('R-squared:', r2_score(output, predictions))  
print('Mean Squared Error:', mean_squared_error(output, predictions))  