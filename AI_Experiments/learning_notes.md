# Learning Notes 🌺

I have decided to systematically review all the details of deep learning, and organize all important concepts together.

## Data Preprocessing Methods

## Layers & Dimensions 😱😱
### Hidden Units
* Having more hidden units, also means having higher dimensional space representaton, will allow you to learn more complex representation.
  * Imagine this allows you to cut an image into smaller pieces for learning.
  * The drawback is more computationally expensive, and you might be learning unnecessary patterns which could even lead to overfitting.
  
### Activation Functions

### Output Layer

### Optimizer
  
### Evaluation Metrics
* [Full list of Keras metrics][4]
* `crossentropy`
  * Good choice for probability output.
  * It measures the distance between y_true and y_pred probabilities


## Overfitting
### Possible Reasons
* Can be because your model has remembered the mapping between the training samples and the targets.
* Can be because there is new samples in the testing set that never appeared in the training set.
* The infrastructure is too complex and learned unnecessary patterns.

### Solutions to Avoid Overfitting
* Plot the metrics (such as loss, accuracy, etc.) between training set and validation set, stop training near the epoch that starts to show the opposite trending between training metrics and validation metrics.

## Well Known Datasets
* [Keras packaged datasets][3]

## Reference
* [Deep Learning Python Notebooks][1]
* [Deep Learning with Python][2]

[1]:https://github.com/fchollet/deep-learning-with-python-notebooks
[2]:https://www.manning.com/books/deep-learning-with-python?a_aid=keras&a_bid=76564dff
[3]:https://keras.io/api/datasets/
[4]:https://keras.io/api/metrics/
