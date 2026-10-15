# Restaurant Ratings Prediction & Cuisine Classification

**A Machine Learning Project | Cognifyz Technologies Internship**

Exploring restaurant ratings, service features, pricing patterns, and cuisine data using Python and machine learning to uncover business insights and evaluate predictive performance.

## Project Overview

This project applies machine learning techniques to **9,551 restaurant records** from the Cognifyz Technologies internship dataset. The goal was to investigate whether restaurant ratings could be predicted from basic listing attributes and whether cuisine information could be used to classify restaurants.

## Dashboard Overview-Data cleaning
![Dashboard Overview](https://github.com/oloruntayobolade/Machine-Learning-Project/blob/main/Script_Overview1-data-cleaning.png)

Two modelling tasks were explored:

- **Restaurant Rating Prediction:** A Linear Regression model using five restaurant features to examine patterns associated with rating levels.
- **Cuisine Classification:** A Logistic Regression baseline using frequency-encoded cuisine information to classify restaurant records.

The project combines data preparation, exploratory analysis, feature encoding, model development, performance evaluation, and business interpretation.

An important finding was that price range had the strongest positive association with rating levels among the features examined. The cuisine classification experiment also demonstrated the limitations of using a single frequency-encoded feature to distinguish restaurants.

## Project Objectives

- Prepare and transform restaurant data for machine learning.
- Investigate how pricing and service features relate to restaurant ratings.
- Build and evaluate a Linear Regression model for rating prediction.
- Explore cuisine-based classification using Logistic Regression.
- Interpret model coefficients and performance metrics.
- Translate analytical findings into practical recommendations for restaurant owners and food platforms.
- Identify modelling limitations and opportunities for improvement.

## Tools & Technologies

| Technology | Application |
|---|---|
| Python | Data analysis and machine learning |
| pandas | Data manipulation and preparation |
| NumPy | Numerical operations |
| scikit-learn | Model development, preprocessing, and evaluation |
| Matplotlib | Data visualization |
| Jupyter Notebook | Interactive analysis and experimentation |

## Dataset Overview

The dataset contains **9,551 restaurant records and 22 columns**, covering restaurant identifiers, location, cuisines, pricing, service availability, and ratings.

Key features used in the analysis included:

- `Price range`
- `Has Table booking`
- `Has Online delivery`
- `Is delivering now`
- `Switch to order menu`
- `Rating text`
- `Cuisines`
- Restaurant name and city information for relevant analytical tasks

The rating target was ordinal-encoded from Poor to Excellent, with rating levels represented numerically from 0 to 5.

## Project Workflow

### 1. Data Preparation

- Loaded the dataset into pandas.
- Examined missing values across columns.
- Filled missing cuisine entries using the most frequent cuisine value.
- Converted Yes/No service fields into binary numerical variables.
- Encoded rating categories into an ordered numerical representation.
- Transformed multi-valued cuisine entries into frequency-based numerical features for the classification experiment.

### 2. Exploratory Data Analysis

Investigated relationships between restaurant ratings, price ranges, service options, and locations.

The analysis explored whether higher-rated restaurants tended to occupy higher price ranges and whether service features showed positive or negative associations with rating levels.

### 3. Feature Engineering

Prepared numerical inputs for the machine learning models.

For rating prediction, five features were used: table booking, online delivery, delivering now, switch to order menu, and price range.

For cuisine classification, cuisine entries were transformed into a single frequency-based score. This provided a simple baseline but substantially limited the information available to distinguish restaurants.

### 4. Model Development

**Model 1: Linear Regression**

Used to estimate restaurant rating levels from pricing and service-related features.

**Model 2: Logistic Regression**

Used as a baseline classification approach, with restaurant names as the target labels and a single frequency-encoded cuisine feature as the input.

The cuisine classification experiment used an 80/20 train-test split with `random_state=42` and feature standardization fitted on the training data.

### 5. Model Evaluation

The models were evaluated using the metrics documented in the project report:

- R-squared (\(R^2\))
- Mean Squared Error (MSE)
- Classification accuracy

The evaluation focused not only on numerical performance but also on whether the chosen features and target variables were appropriate for the prediction task.

## Model Results

### Experiment 1: Restaurant Rating Prediction

| Evaluation metric | Result |
|---|---:|
| R-squared (\(R^2\)) | 0.246 |
| Mean Squared Error (MSE) | 0.891 |
| Root Mean Squared Error (RMSE) | Approximately 0.94 rating levels |

The model explained approximately **24.6% of the variation in restaurant rating levels** using the five selected features.

The RMSE of approximately 0.94 indicates that the model's predictions had an error magnitude of around 0.94 rating levels on the encoded target scale.

#### Feature Coefficients

| Feature | Coefficient | Interpretation |
|---|---:|---|
| Price range | +0.63 | Strongest positive association |
| Online delivery | +0.19 | Positive association |
| Table booking | -0.30 | Negative association |
| Is delivering now | -0.23* | Negative after controlling for the other features |
| Switch to order menu | 0.00 | No contribution in the fitted model |

*The delivering-now coefficient changed direction when other features were included, indicating that its relationship with ratings may overlap with other variables.*

**Key insight:** Price range had the largest positive coefficient among the features examined. However, these coefficients describe associations in the fitted model; they do not establish that changing a restaurant's price or service options will cause its ratings to change.

### Experiment 2: Cuisine Classification

| Evaluation metric | Result |
|---|---:|
| Model | Logistic Regression |
| Training records | 7,640 |
| Test records | 1,911 |
| Correct test predictions | 41 |
| Accuracy | 2.15% |

The baseline model achieved **2.15% accuracy**, correctly predicting 41 of the 1,911 restaurant names in the test set.

This result exposed two major limitations:

1. **Limited feature information:** A single cuisine-frequency score cannot adequately represent the differences between restaurants with similar cuisine profiles.
2. **High-cardinality target:** Predicting restaurant names creates a classification problem with many possible labels, making the task difficult with such limited input information.

A Random Forest approach using TF-IDF features from cuisine text was also configured as an alternative experiment in the project. Its comparative performance should be evaluated before drawing conclusions about whether it improves classification.

**Key insight:** Feature representation and target selection are critical to classification performance. A more suitable next experiment would predict a restaurant's primary cuisine category rather than its name.

## Key Findings

### 1. Price Range Was the Strongest Rating Predictor

Price range had a coefficient of +0.63, substantially higher than the next strongest positive coefficient, online delivery at +0.19.

This suggests that price range was the most influential feature among the five inputs in the fitted linear regression model.

### 2. Online Delivery Showed a Positive Association

Online delivery had a coefficient of +0.19, indicating a modest positive relationship with predicted rating levels after accounting for the other included features.

Further analysis would be needed to establish whether this pattern holds across restaurant types, cities, and cuisine categories.

### 3. Table Booking Was Negatively Associated with Ratings

The table-booking feature had a coefficient of -0.30.

This finding provides a useful question for further investigation: do restaurants offering table booking differ in price, customer expectations, cuisine, location, or other factors that influence ratings?

### 4. Some Features Added Little Predictive Value

The `Switch to order menu` feature had a coefficient of 0.00 in the fitted model. The delivering-now feature also showed an unstable relationship when other variables were included.

These findings highlight the importance of reviewing feature usefulness and overlap before using a model to guide business decisions.

### 5. Cuisine Classification Needed a Better Approach

The 2.15% classification accuracy demonstrated that a single frequency-encoded cuisine score was insufficient for distinguishing restaurant names.

This informed the next-step recommendation to use richer text features and a more meaningful classification target.

## Business Recommendations

Based on the analysis, the following actions could be explored:

- **Benchmark ratings within price ranges:** Compare restaurants with similar pricing to make performance comparisons more meaningful.
- **Investigate online delivery:** Explore whether the positive association persists across locations, cuisine types, and price categories.
- **Study table-booking patterns:** Combine ratings with customer reviews and additional restaurant features to understand the negative association.
- **Improve feature selection:** Reassess features with limited contribution or unstable coefficients.
- **Use richer data for rating prediction:** Add cuisine, city, votes, and average cost where available.
- **Redesign the cuisine classification task:** Predict primary cuisine categories using text-based features and evaluate performance with macro F1-score and a confusion matrix.
- **Strengthen model validation:** Use held-out test data and cross-validation to better estimate how well the rating model generalizes to unseen restaurants.

## Limitations and Future Improvements

This project served as an initial modelling exercise and identified several opportunities to strengthen the analysis.

| Limitation | Proposed improvement |
|---|---|
| Rating model \(R^2\) of 0.246 | Add informative features and test alternative regression algorithms |
| Rating model evaluation requires stronger validation | Use a held-out test set and cross-validation |
| Unrated restaurants were encoded as an ordinal category | Separate unrated records or model rating availability independently |
| Cuisine classification predicted restaurant names | Reframe the task around primary cuisine categories |
| Single frequency-encoded cuisine feature | Explore TF-IDF, multi-hot encoding, or other text representations |
| Potential character-encoding issues | Reload the source with the appropriate encoding and validate affected fields |
| Unstable delivering-now coefficient | Investigate overlap with online delivery and assess multicollinearity |

## Skills Demonstrated
- Data cleaning and missing-value handling
- Exploratory data analysis
- Feature engineering and categorical encoding
- Linear Regression
- Logistic Regression
- Random Forest and TF-IDF experimentation
- Train-test splitting and feature standardization
- Regression and classification evaluation
- Interpretation of model coefficients
- Critical evaluation of machine learning limitations
- Translating model results into business recommendations

## Conclusion
This project demonstrates a practical machine learning workflow for exploring restaurant ratings and cuisine data, from data preparation and feature engineering to model evaluation and business interpretation.
The rating prediction experiment identified price range as the strongest positive association among the selected features, while the cuisine classification experiment showed the limitations of a weak feature representation and an overly specific target.

Beyond model building, the project reinforced an important principle of data science: **a useful model depends not only on the algorithm, but also on the quality of the features, the suitability of the target, and the strength of the evaluation process.**
---
**Author:** Bolade Oloruntayo  
**Project:** Cognifyz Technologies Machine Learning Internship  
**Focus Areas:** Data Analysis | Machine Learning | Predictive Analytics | Business Intelligence

## NOTE
**To read the python script, doenload and open with a browser (Chrome suggested)**
