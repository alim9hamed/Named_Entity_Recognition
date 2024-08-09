# Named Entity Recognition (NER) Project

This project aims to build a Named Entity Recognition (NER) model using a dataset from Kaggle, specifically the Groningen Meaning Bank (GMB). The dataset consists of thousands of sentences with words tagged for named entity recognition tasks. The model is trained to predict named entities in English text.

## Project Overview

### Dataset

- **Source:** Kaggle
- **Dataset Name:** Groningen Meaning Bank (GMB)
- **Description:** Contains thousands of sentences with each word annotated for named entities. This dataset is commonly used for training NER models.

### Project Stages

1. **Import Libraries**
   - Essential libraries for data handling, model building, and evaluation.

2. **Loading Data**
   - Load the dataset into the environment and prepare it for analysis.

3. **Exploratory Data Analysis (EDA) & Text Transformation**
   - **Tokens Integration:** Combine tokens into sentences.
   - **Tokens Indexing:** Convert tokens into numerical indices.
   - **Padding Tokens:** Ensure uniform input length by padding sequences.
   - **Splitting Data:** Divide the dataset into training and testing sets.

4. **Building and Testing Model**
   - Construct and train an NER model using TensorFlow and Keras.
   - Evaluate the model's performance on both training and validation sets.

5. **Evaluation**
   - Analyze model performance metrics including accuracy and loss on test data.

## Model Performance

- **Training Accuracy:** Min: 0.964, Max: 0.992, Current: 0.992
- **Validation Accuracy:** Min: 0.984, Max: 0.986, Current: 0.986
- **Training Loss:** Min: 0.026, Max: 0.142, Current: 0.026
- **Validation Loss:** Min: 0.045, Max: 0.056, Current: 0.046
- **Test Accuracy:** 0.986
- **Test Loss:** 0.046

## Getting Started

### Prerequisites

- Python 3.x
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Other necessary libraries (listed in `requirements.txt`)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/ner-project.git
   cd ner-project
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. **Load the dataset:**

   Ensure that the dataset is available at the specified path or update the path in the code.

2. **Run the Jupyter Notebook:**

   ```bash
   jupyter notebook
   ```

   Open the provided Jupyter notebook and follow the instructions to train and evaluate the model.

### Results

The model has achieved high accuracy and low loss on both validation and test datasets, demonstrating its effectiveness in identifying named entities in text.

## Contributing

Feel free to contribute to the project by submitting issues, pull requests, or suggestions for improvement.


## Acknowledgements

- [Groningen Meaning Bank (GMB) dataset](https://www.kaggle.com/datasets/abhinavwalia95/entity-annotated-corpus)
