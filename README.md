# Machine Learning: Fake News Detection

## Description
This project implements various machine learning models to detect fake news. It involves data preprocessing, feature extraction using TF-IDF, and classification using several algorithms to identify whether a given news article is real or fake.

## Features
- Data loading and initial exploration
- Data cleaning and preprocessing (lowercase conversion, punctuation removal, stopword removal, stemming)
- Feature extraction using CountVectorizer and TfidfTransformer
- Implementation and evaluation of multiple classification models:
    - Naive Bayes
    - Logistic Regression
    - Decision Tree
    - Random Forest
    - Support Vector Machine (SVM)
- Confusion matrix visualization for each model

## Installation
To set up the project, clone the repository and install the required Python libraries. It is recommended to use a virtual environment.

```bash
git clone https://github.com/rahuldembani16/Machine-Learning-ML-Fake-News-Detection
cd Machine-Learning-ML-Fake-News-Detection
pip install pandas numpy matplotlib seaborn scikit-learn nltk
```

## Usage
To run the analysis and train the models, execute the `FakeNewsDetection.ipynb` Jupyter Notebook. You can open it using Jupyter Notebook or JupyterLab.

```bash
jupyter notebook FakeNewsDetection.ipynb
```

The notebook will guide you through the data loading, preprocessing, model training, and evaluation steps.

## Results
The following accuracy scores were achieved by the different models:

- **Naive Bayes:** 94.82%
- **Logistic Regression:** 99.04%
- **Decision Tree:** 99.72%
- **Random Forest:** 99.58%
- **SVM:** 99.58%

Visualizations of the data distribution and model performance (confusion matrices) are available in the notebook and as image files in the repository:

- `img1.png`
- `img2.png`
- `img3.png`
- `img4.png`
- `img5.png`
- `img6.png`
- `img7.png`
- `img8.png`
- `img9.png`

## Contact
For any questions or inquiries, please contact Rahool Dembani at rahool.dembani16@gmail.com.
