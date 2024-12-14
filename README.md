# FindDefault-Prediction-of-Credit-Card-fraud

**Fraud Detection with Machine Learning**

This project applies machine learning techniques to detect fraudulent transactions in credit card data. The notebook implements data preprocessing, class imbalance handling, and model evaluation to develop an effective solution for binary classification.

**Table of Contents**
  •	Dataset
•	Project Workflow
•	Key Techniques
•	Models Implemented
•	Evaluation Metrics
•	Dependencies
•	How to Run
•	Results

**Dataset**
The dataset includes anonymized features obtained through Principal Component Analysis (PCA) and contains two untransformed features: Time and Amount. The target variable is Class, where:
•	0 represents non-fraudulent transactions
•	1 represents fraudulent transactions

**Project Workflow**
1.	Data Exploration and Visualization
•	Analyzed class distribution and identified significant class imbalance.
•	Visualized features and their distributions.

2.	Data Preprocessing
•	Scaled the Amount feature using StandardScaler.
•	Dropped the Time feature as it is irrelevant.

3.	Train-Test Split
•	Split the dataset into training (70%) and testing (30%) sets using train_test_split.

4.	Class Imbalance Handling
•	Applied Synthetic Minority Oversampling Technique (SMOTE) to balance the dataset.

5.	Model Implementation and Evaluation
•	Experimented with Decision Trees and Random Forests.
•	Compared models using various evaluation metrics.
Key Techniques
•	Synthetic Minority Oversampling Technique (SMOTE): Augmented the minority class to address class imbalance.
•	Feature Scaling: Normalized the range of numerical features.
•	Model Persistence: Serialized the trained model and dataset using pickle for future use.
Models Implemented
•	Decision Tree Classifier
•	Random Forest Classifier
Evaluation Metrics
•	Accuracy
•	Precision
•	Recall
•	F1 Score
•	Confusion Matrix
Dependencies
•	Python
•	Jupyter Notebook
•	Libraries:
o	pandas
o	numpy
o	seaborn
o	matplotlib
o	scikit-learn
o	imbalanced-learn
o	pickle
How to Run

1.	Clone this repository.

2.	Install dependencies using:

pip install -r requirements.txt

3.	Open the notebook in Jupyter:

jupyter notebook Untitled.ipynb

4.	Run all cells sequentially.


Results

•	The Random Forest classifier outperformed the Decision Tree classifier in both imbalanced and resampled datasets.

•	Using SMOTE, the Random Forest achieved over 99% accuracy, effectively classifying fraudulent transactions with minimal misclassifications.

Acknowledgments

This project leverages the creditcard.csv dataset and techniques for handling imbalanced datasets.
