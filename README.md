# README

## Project Title

Car Evaluation System Using Support Vector Machine (SVM)

## Description

This project uses the Support Vector Machine (SVM) algorithm to classify cars into different evaluation categories based on various features such as buying price, maintenance cost, number of doors, passenger capacity, luggage boot size, and safety level.

## Objective

To build a machine learning model that predicts the quality of a car as:

* Acceptable (acc)
* Good (good)
* Unacceptable (unacc)
* Very Good (vgood)

## Dataset

The project uses the UCI Car Evaluation Dataset.

Attributes:

1. Buying Price
2. Maintenance Cost
3. Number of Doors
4. Persons Capacity
5. Luggage Boot Size
6. Safety

Target:

* Class

## Technologies Used

* Python
* Pandas
* Scikit-learn

## Modules Used

* pandas
* LabelEncoder
* StandardScaler
* train_test_split
* SVC
* accuracy_score
* classification_report

## Steps Involved

1. Load the dataset.
2. Convert categorical values into numerical values using Label Encoding.
3. Split the dataset into training and testing data.
4. Standardize the features.
5. Train the SVM classifier.
6. Predict car evaluation classes.
7. Evaluate the model using accuracy and classification report.

## Output

The system predicts the evaluation category of a car and displays the model accuracy and classification report.

## Conclusion

The Car Evaluation System successfully classifies cars into different categories using the SVM algorithm. The model provides accurate predictions and helps in assessing the quality of cars based on their attributes.
