# deep-learning-challenge
Module 21 - Deep learning and neural networks

Note: There are 4 jupyter notebooks, each one a separate attempt at optimizing the neural network used for classification. An .h5 file was generated for each model and has an "_#" at the end of the file name corresponsing to the model attempt.

Overview of the analysis:

The purpose of this analysis was to investigate the predictive ability of dense neural networks by implementing various optimization techniques such as hyperparameter tuning and data engineering. 

Data Preprocessing:

The dataset included information about the operations and success of a list of charities. The targets were binary classifications indicating whether the charity was successful or not. There were 9 input variables, 8 categoritcal and one continuous, which was the ASK_AMT column. Two columns that included the charity name and ID were removed from the dataset as they contained no predictive information. Two categoritcal columns contained many unique values, some of which had a small number of value counts, which were binned as a category of "other" based on thresholds set after inspecting the value counts of all categories in the column. The categoritcal columns were one-hot encoded with the pandas function, getDummies(), then the dataset was split into training and testing subsets using the default train/test fraction of 0.75. The continuous input feature was then standardized to have unit variance and zero mean with the pandas standard scaler. This resulted in a total of 44 input features. 

Compiling, Training, and Evaluating the Model:

Four attempts at building a classification model using a dense neural network were performed. The following is a summary of each attempt:

Model 1

- 3 hidden layers having 100, 30, and 10 nodes, each with the relu activation function; output layer with one node and the sigmoid activation function; 100 training epochs; loss fuction of binary cross-entropy, adam optimizer, and accuracy was used to evalute the model performance  
- Results: Loss: 0.5705615878105164, Accuracy: 0.726064145565033

Model 1

- Optimization technique: drop the correlated columns generated during the one-hot encoding resulting in 37 features instead of 44
- Same schema as model 1
- Loss: 0.5726596713066101, Accuracy: 0.7286297082901001

Model 2

- Optimization technique: Increase the number of hidden layers to 4 and increase the number of nodes per layer
- 4 hidden layers having 500, 250, 100, and 10 nodes, each with the relu activation function; output layer with one node and the sigmoid activation function; 100 training epochs; loss fuction of binary cross-entropy, adam optimizer, accuracy was used to evalute the model performance
- Loss: 0.6032054424285889, Accuracy: 0.7353935837745667

Model 3

- Optimization technique: Use the keras_tuner library and the Hyperband algorithm to optimize the hyperparameters - the number of hidden layers (in addition to the input layer), nodes per layer, and activation function
- Best parameters:
{'num_layers': 3,
 'units_0': 35,
 'activation_0': 'relu',
 'units_1': 40,
 'activation_1': 'relu',
 'units_2': 30,
 'activation_2': 'relu',
 'tuner/epochs': 4,
 'tuner/initial_epoch': 2,
 'tuner/bracket': 4,
 'tuner/round': 1,
 'tuner/trial_id': '0043'}
 - Input/Output layers and compiling parameters: same as previous models
 - Loss: 0.5525076389312744, Accuracy: 0.7329446077346802

Summary: 

Using the 4 classification models, the best accuracy that was achieved was 73%, which did not meet the target of > 75%. The best performance was achieved usng the model with the most hidden layers with the most nodes (Model 3), so perhaps further increasing these values would have yeilded a more accurate model at the cost of training time. Given that this is a classification problem, a simpler machine learning technique such as logistic regression or random forest could be investigated. 