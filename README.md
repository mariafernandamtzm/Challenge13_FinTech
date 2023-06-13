# Challenge13_FinTech

Venture Funding with Deep Learning

This repository contains a machine learning project focused on predicting the success of venture funding applicants using deep learning techniques. The project utilizes Python as the main programming language and leverages TensorFlow and Keras libraries for building and training neural network models.

Problem Statement
The task is to create a binary classifier model that predicts whether an applicant will be successful if funded by Alphabet Soup, a venture capital firm. The dataset used for training and evaluation contains information about more than 34,000 organizations that have received funding from Alphabet Soup, including whether they became successful.

## Instructions
The project is divided into the following sections:

Data Preparation: In this section, the dataset is preprocessed using Pandas and scikit-learn's StandardScaler(). Categorical variables are encoded using OneHotEncoder, and the numerical variables are scaled. The dataset is split into training and testing sets.

Neural Network Model: A binary classification deep neural network model is designed using TensorFlow and Keras. The model is compiled with the binary_crossentropy loss function, the adam optimizer, and the accuracy evaluation metric. The model is then fit to the training data and evaluated on the testing data.

Model Optimization: Multiple attempts are made to optimize the neural network model by adjusting input data, adding more neurons or hidden layers, using different activation functions, and modifying the number of training epochs. The accuracy scores of each model are compared and evaluated.

Model Export: The final optimized models are saved and exported as HDF5 files for future use. The models are named AlphabetSoup.h5 and include the architecture, weights, and configuration of the neural network.

## Languages and Technologies Used
Python: Main programming language used for implementation.
TensorFlow: Deep learning framework used for building and training neural network models.
Keras: High-level neural networks API used for building and optimizing deep learning models.
Pandas: Library used for data manipulation and preprocessing.
scikit-learn: Library used for data scaling and splitting.
HDF5: Hierarchical Data Format used for saving and exporting the trained models.
Repository Structure
The repository structure is as follows:

## Conclusion 

Throughout the project, we followed a systematic approach to tackle the problem. We started by preparing the data, including preprocessing, encoding categorical variables, and scaling numerical variables. The preprocessed data was then split into training and testing sets.

Next, we designed a deep neural network model using TensorFlow and Keras. We experimented with different architectures, activation functions, and optimization techniques to improve the model's accuracy. The model was compiled with appropriate loss function, optimizer, and evaluation metric. It was then trained on the training data and evaluated on the testing data.

By leveraging the capabilities of deep learning and data analysis, we can make more informed decisions in the field of venture funding, contributing to the success and growth of startups and the venture capital industry as a whole.









