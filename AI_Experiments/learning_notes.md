# Learning Notes 🌺

I have decided to systematically review all the details of deep learning, and organize all important concepts together.

## Data Preprocessing Methods 😱
### Better to Normalize to the same range
* When features are in different ranges, better to normalize them into the same range.
* [sklearn functions][7]

### Text data
* Text data means a column comtains longer text strings, such as comments, topics, etc.
* General Steps
  1. Choose top N toknized words appeared in all the text, N excludes stop words
  2. One-hot encoding: each distinct word as a column, each row reprensents a text, for word appeared in this text, mark it as 1, others mark as 0
  
### Multi-class lables [Python]
* Mehtod 1 - One-hot encoding
  * The labels have N columns, each column represents a class
  * Mark the labeled class as 1
* Method 2 - Integer labels
  * Each class is represented as an integer
  * 1 numpy array is enough for the labels
* [Example for both methods][8]

### How to Preprocessing Testing Data ❣❣
* The parameters used for preprocessing the training data should be generated from the training data only, and be applied to the testing data too.
* For example, when you are using `mean` and `std` to normalize the data, these params should be generated from the training data, and be applied to the testing data.

## Layers & Dimensions 😱😱
### Hidden Layers
* Having more units in a layer, also means having higher dimensional space representaton, will allow you to learn more complex representation.
  * Imagine this allows you to cut an image into smaller pieces for learning.
  * The drawback is more computationally expensive, and you might be learning unnecessary patterns which could even lead to overfitting.
* When there are N classes to predict, the dimensions of all the hidden layers better > N, otherwise the information loss will lead to overfitting
  
### Activation Functions
#### `relu` (rectified linear unit)
* It will zero out negative values.

#### `softmax`
* When it's used in the last layer, usually for N multi-class probability output, the last layer will have N nodes (N > 2).
* The sum of N classes probabilities is 1.

#### `sigmoid`
* When it's used in the last layer, usually for 2 class probability output, the last layer will have 1 node.
* The sum of 2 classes probability is 1.

#### Without Activation Functions
* A layer has 1 node (unit) without activation function equals to a linear function --> Can be used for regression problem.

### Output Layer
* 2 classes
  * 1 node, sigmoid for class probabilities
* N classes (N > 2)
  * N nodes, softmax for class probabilities
* Regression
  * Linear regression - 1 node without activation function

### Optimizer
* `rmsprop` is often good enough

### About K-fold Cross Validation with Keras
* Well, adding this part is because I failed this piece in Unity interview...
  * [Here's the example if the interviewer asks you to implement k-fold][5]
  * [Here's the example if you need k-fold with Keras in the work][6]
* After finding the optimal epoch number from cross validation, train the whole dataset with the optimal number of epoch.
  
### Evaluation Metrics
* [Full list of Keras metrics][4]

#### Loss Functions
* The training process is trying to reduce the loss.
* `crossentropy`
  * Good choice for probability output.
  * It measures the distance between y_true and y_pred probabilities
  * `binary_` one can be used for probability of 2 classes
  * `categorical_` one can be used for probability of 2+ classes, and the label is in categorical format (such as one-hot encoding)
  * `sparse_categorical_` is similar to the use case of `categorical_`, but it's used when the labels are in integer format
  
#### Evaluation Visualization
##### Epoch Plot
* Epoch - Evaluation Metrics plot
  * If there are metrics for training and 1 validation set, plot both curves for comparison will help find the stopping epoch
    * The stopping point is where starts the opposite trending
  * For k-fold cross validation, we can average the validation metrics from all the k folds, and in each fold, training & validation sets generates 1 evaluation value
    * The stopping point is where the trend fully changed
    * Especially useful when the dataset is small


## Overfitting
### Possible Reasons
* Can be because your model has remembered the mapping between the training samples and the targets. This often happens when model capacity is too large.
  * Model capacity is often determined by the number of layers, and the number of units in each layer. Larger the number, higher capacity.
* Can be because there is new samples in the testing set that never appeared in the training set.
* The infrastructure is too complex and learned unnecessary patterns.
* Information loss in the hidden layers.
* Small training dataset.

### Solutions to Avoid Overfitting
* Plot the metrics (such as loss, accuracy, etc.) between training set and validation set, stop training near the epoch that starts to show the opposite trending between training metrics and validation metrics.
  * Note, it doesn't need the 2 curves to be close to each other, we just need to check the moving trending (up or down) for each curve. For example, when the training acccuacy keeps going up, at point A, validation accuracy starts going down, then this point A can be the stopping point even if later validation accuracy could still move up after point A.
* Avoid having the dimensions of the hidden layers < N, N is the number of classes.
* If the training dataset is small, use a smaller NN (1 or 2 hidden layers + 1 last layer), and use cross validation to find the optimal epoch number
* [Regularization][9]
  * Training data only
  * By adding the weight regularization, it's trying to reduce the model complexity
  * L1 regularization, where the cost added is proportional to the absolute value of the weights coefficients
  * L2 regularization, where the cost added is proportional to the square of the value of the weights coefficients
* [Dropout][10]
  * Can be applied to both training and testing data, but in practice, often applied to training data only
  * The core idea is that introducing noise in the output values of a layer can break up coincident patterns that are not significant
  * It's one of the most effective and most commonly used method
  * It randomly drops (i.e. setting to zero) a number of output features of a layer during training
  * Dropout rate is often set between 0.2 and 0.5

## Well Known Datasets
* [Keras packaged datasets][3]

## Reference
* [Deep Learning Python Notebooks][1]
* [Deep Learning with Python][2]

[1]:https://github.com/fchollet/deep-learning-with-python-notebooks
[2]:https://www.manning.com/books/deep-learning-with-python?a_aid=keras&a_bid=76564dff
[3]:https://keras.io/api/datasets/
[4]:https://keras.io/api/metrics/
[5]:https://nbviewer.jupyter.org/github/fchollet/deep-learning-with-python-notebooks/blob/master/3.7-predicting-house-prices.ipynb
[6]:https://www.machinecurve.com/index.php/2020/02/18/how-to-use-k-fold-cross-validation-with-keras/
[7]:https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
[8]:https://nbviewer.jupyter.org/github/fchollet/deep-learning-with-python-notebooks/blob/master/3.6-classifying-newswires.ipynb
[9]:https://nbviewer.jupyter.org/github/fchollet/deep-learning-with-python-notebooks/blob/master/4.4-overfitting-and-underfitting.ipynb#Adding-weight-regularization
[10]:https://nbviewer.jupyter.org/github/fchollet/deep-learning-with-python-notebooks/blob/master/4.4-overfitting-and-underfitting.ipynb#Adding-dropout
