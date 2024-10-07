# Sentiment Analysis of Amazon Reviews Using BERT and Transformer Models

## Overview

This project focuses on sentiment analysis of Amazon customer reviews using the BERT (Bidirectional Encoder
Representations from Transformers) model. The goal is to classify reviews as positive or negative based on their textual
content. By leveraging state-of-the-art transformer architectures, this project aims to achieve high accuracy in
sentiment classification, making it beneficial for businesses looking to gauge customer opinions.

## Table of Contents

- [Project Description](#project-description)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Usage](#usage)

## Project Description

Sentiment analysis is a crucial task in natural language processing (NLP) that involves determining the emotional tone
behind a series of words. In this project, we utilize the BERT model, which has shown remarkable performance in various
NLP tasks. The model is trained on a large dataset of Amazon reviews, allowing it to learn contextual relationships
between words in a review and improve sentiment classification accuracy.

Through this project, we aim to:

- Demonstrate the effectiveness of transformer models in understanding context.
- Provide insights into customer sentiment, which can aid businesses in decision-making.
- Implement a pipeline that can be reused for future NLP tasks.

## Key Features

- **Transformer Architecture**: Utilizes BERT, a transformer-based model that excels in understanding context and
  semantics in text, leading to superior performance in sentiment classification tasks.
- **Data Preprocessing**: Includes comprehensive text cleaning and tokenization processes to prepare the dataset for
  training, ensuring high-quality input for the model.
- **Batch Processing**: Efficiently handles large datasets with batch tokenization and saving, enhancing performance
  during training and reducing computational load.
- **Robust Training Pipeline**: Implements a well-structured training pipeline with clear validation and testing phases
  to ensure model reliability and prevent overfitting.
- **Prediction Functionality**: Provides functionality to predict sentiment for new reviews, allowing for real-time
  analysis of customer feedback.
- **Extensible Framework**: Designed to be easily extended for additional NLP tasks, making it a versatile resource for
  future projects.

## Dataset

The dataset used in this project is
the [Amazon Reviews Dataset](https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews). It contains a vast number
of customer reviews along with their associated ratings. The dataset includes:

- Reviews from various product categories.
- Ratings that have been converted into sentiment labels (positive and negative).

The dataset is pre-processed to ensure that only relevant information is retained for sentiment analysis. The
pre-processing steps include removing duplicates, filtering out irrelevant reviews, and encoding sentiment labels based
on review ratings.

## Usage

Install the required packages:

 ```bash
   pip install -r requirements.txt
 ```

- First, download and extract the dataset by running the `Dataset/download_dataset.ipynb` file
- Prepare the data for training the model by running the `data_preprocessing.ipynb` file
- Note: Note that due to the large amount of data in the code, only a quarter of it is processed in
  the `data_preprocessing.ipynb` file

```
# Loading 1 quarter of the dataset in the data_preprocessing.ipynb file
train_dataset = pd.read_csv(f'{project_path}/Dataset/train.csv', names=column_names, nrows=1000000)
test_dataset = pd.read_csv(f'{project_path}/Dataset/test.csv', names=column_names, nrows=100000)
```

```
# Loading all data in the data_preprocessing.ipynb file
train_dataset = pd.read_csv(f'{project_path}/Dataset/train.csv', names=column_names)
test_dataset = pd.read_csv(f'{project_path}/Dataset/test.csv', names=column_name)
```

- Training the model with processed data in the `train.ipynb` file
- Using the trained model with the test `test.ipynb`

Note: If the model is fully trained with all the data, it can achieve extremely high accuracy.
