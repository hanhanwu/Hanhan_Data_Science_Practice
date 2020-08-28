# Learning Notes 🌺

I have decided to systematically review all the details of deep learning, and organize all important concepts together.

## Fun Facts of Deep Learning
* Deep learning can find interesting features itself, so we don't need manually feature engineering in deep learning. 
  * But this is achievable only when the dataset is large enough, higher dimensions of the data (such as images), larger dataset is needed.

## Data Preprocessing Methods 😱
### Shuffle the data before spliting
* When the arrangement of labels not distributed similar for training and testing data, better to shuffle the data before spliting into training and testing.

### Better to Normalize to the same range
* When features are in different ranges, better to normalize them into the same range.
* [sklearn functions][7]

### Text data
* Text data means a column comtains longer text strings, such as comments, topics, etc.
* General Steps
  1. Choose top N toknized words appeared in all the text, N excludes stop words
  2. One-hot encoding: each distinct word as a column, each row reprensents a text, for word appeared in this text, mark it as 1, others mark as 0
  
### Preprocess Image data
#### Raw images to NN readable input 
* [Keras ImageDataGenerator][17] will make below steps easier
  * Convert JPEG to RGB
  * Resize all the images to the same size
  * Convert images into floating point tensors
  * Rescale the pixel values (between 0 and 255) to the [0, 1] interval
  * Batch size is the number of samples in a batch
    * `steps_per_epoch = total training sample size / training batch size`, this is because in each epoch, it runs all the samples
    * `validation_steps = total validation sample size / validation batch size`
  * Seperate data and labels
    * Labels are created based on directories, different classes of images are put in different directories
#### Convert an image to image tensor
* [Example][22]
* The image tensor can be used to understand the output of each activation layer
  
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
  * It seems that `(value - mean)/std` is a common way used to normalize features when they are in different scales. This method doesn't guarantee they are on the same scale, but could make them on similar scales

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
* [Top-n Error][24]
  * We often saw "top-1 error", "top-5" error appeared in papers, here's more explaination

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
  * If the curves look too noisy, you can try methods to smooth out the curves, such as using exponential moving average
    * [See the example at the bottom][21]


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
* [Data Augmentation][18]
  * Training data only
  * "Data augmentation takes the approach of generating more training data from existing training samples, by "augmenting" the samples via a number of random transformations that yield believable-looking image"
  * Even though there is no same image in the training set, with this method, the inputs can still be heavily intercorrelated. Therefore it's necessary to add a `dropout` layer right after flattened data to reduce some dimensions in order to further reduce overfitting.
  
## Convolutional Networks (Convnet)
* Convnets can learn <b>local, translation-invariant features</b>, so they are very data-efficient on perceptual problems. Therefore, even when the dataset is small (such as hundreds of images), you might still get a reasonable results.
* [Keras Convnet][11]
* [Conv2D][12]
  * "Input shape" is `(batch_size, image_height, image_width, image_channels)`, batch_size is optional
    * The width and height tend to shrink when we go deeper in convnet
    * However, in Keras `reshape()`, the order is `(sample_size, channels, width, height)`...
  * When there are multiple layers of Conv2D in a neural net, deeper layer gets larger number of batches, notmrally we choose batch size in a sequence of 32, 64, 128, ...
    * It 
  * `image_channels` is also the image depth. [How to use opencv2 to find image channels][13]
    * If returns 2 dimensions, the image channel is 1, otherwise it's the third number returned in the output
* [MaxPooling2D][16]
  * Pooling Layer - pooling is a down-sampling operation that reduces the dimensionality of the feature map.
  * It performs down-sampling operations to reduce the dimensionality and creates a pooled feature map by sliding a filter matrix over the input matrix.
* [What is `strides`][14]
  * The size (height, width) of the moving window
* Why `Flatten()`
  * `Dense` layer is 1D, so if the output from Conv2D is more than 1D, it needs to be flatterned into 1D before entering into the Dense layer.
* `fit_generator()`
  * `steps_per_epoch = total training sample size / training batch size`, this is because in each epoch, it runs all the samples
  * `validation_steps = total validation sample size / validation batch size`
### Pretrined Convnet
* Better just to reuse the pretrained convolutional base and avoid using the densely-connected classifier. Because
  * The convolutional base has a more generic feature maps while densely-connected features are more specific
  * The convolutional base still contains object location info while densely-connect features do not
* Only choose earlier layers from the trained model if your new dataset has larger difference with the data used for the pretrained model
  * Because the earlier layer capture more generic feature maps
* [Keras pretrained applications][19]
  * [python examples][20]
  * [A bit more description on some of the pre-trained models][25]
* Method 1 - Pretrained Model + Standalone Densely-connected classifier
  * This method is fast and cheap to run, but doesn't allow you to do data augmentation adn therefore may not reach to an optimal accuracy
    * It's possible to do with CPU
  * Need to reshape your new dataset as the output of the pre-trained convnet
  * [See example here][21]
* Method 2 - Extend Pretrained Model
  * It's costly to run, at least need the GPU
  * Before compile and train the model, need to freeze the the convolutional base. "Freezing" a layer or set of layers means preventing their weights from getting updated during training, in order to keep the representations learned by the trained trained model.
    * If you ever modify weight trainability after compilation, you should then re-compile the model, or these changes would be ignored.
  * [See example here][21]
* Fine Tuning
  * Unfreeze a few earlier layers of the frozen model and together being trained with the newly added model parts.
  * It slightly adjusts the more abstract representation of the pretrained model.
  * It is more useful to fine-tune the more specialized features (deeper layers)
  * General Steps:
    *  Add your custom network on top of an already trained base network.
    *  Freeze the base network.
    *  Train the part you added.
    *  Unfreeze some layers in the base network.
    *  Jointly train both these layers and the part you added.
### Convnet Visualization
* Visualize each activation layer helps understand what does each channel learns
* Visualize convnet filters helps understand how the network express the image by resembling edges, colors and textures, etc.
* Visualize the heatmap of classes helps understand which part of the image led to the final classification decisions
* [See examples here][23]


## RNN
* RNN is trying to help analysis on a sequence, so that there could be more context, and therefore they are often used in NLP.
### RNN Pretrained Models
#### Words Embedding
* A word embedding not only converts the word but also identifies the semantics and syntaxes of the word to build a vector representation of this information.
* These models are used to load the embedding layer.
* [Stanford GloVe][27]
  * [Example usage][28]
    * Parse out the word vectors
    * Build an embedding matrix of shape `(max_words, embedding_dim)`, max_words indicates how many top words from the dataset is needed
    * Load the embedding matrix and freeze the embedding layer (layers[0])
#### Sentence Embedding
* Sentence embedding techniques represent entire sentences and their semantic information as vectors. This method could help the machine understand the context better, and when there is large amount of text, it might be more efficient than words embedding.
* [Example usage][29]
  * The major usage in this exmple is to find sentence similarity with cosin similarity, using sentence embedding results
  * 4 types of recommended libraries
    * [doc2vec][30]
    * [sentence transformers][31] and its [full list of pretrained models][32]
    * [Facebook InferSent][33]
    * [Google Universal Sentence Encoder][34]

## Well Known Datasets
* [Keras packaged datasets][3]
* [Cats and Dogs image set][15]

## Reference
* [Deep Learning Python Notebooks][1]
* [Deep Learning with Python][2]

## Interview Questions
I just found some companies like to ask you to implement methods used in deep learning without using numpy, sklearn, no matter whether they need deep learning... This is why I put these here, even though there is no time to check these during the interviews. Personally I don't like this type of interview at all, they gave impossible timeline (unless you have remembered each answer), no one talk with you to help you understand what do the questions really mean, and the solution used here has a big difference with the practical solutions. This type of interviews could be used at most for entry level data scientists but shuld not be used for senior level. Unfortunately, these companies do not have a qualified hiring manager. Sometimes, this world is funny and ridicuclous in a way I can never understand.
* [Implement one-hot encoding][26]
  * This one uses numpy to create the matrix, but you can do it with simple python lists
  * I don't like its second method because of the hashing collision. Instead, I think there are some preprocessing can be done before the encoding here, such as tokenize all the words, removing stop words and other less important tokens (such as those got low tf-idf score)
* [Implement k-fold CV][5]

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
[11]:https://keras.io/api/layers/convolution_layers/
[12]:https://keras.io/api/layers/convolution_layers/convolution2d/
[13]:https://stackoverflow.com/questions/19062875/how-to-get-the-number-of-channels-from-an-image-in-opencv-2
[14]:https://www.quora.com/What-does-stride-mean-in-the-context-of-convolutional-neural-networks
[15]:https://www.kaggle.com/c/dogs-vs-cats/data
[16]:https://keras.io/api/layers/pooling_layers/max_pooling2d/
[17]:https://nbviewer.jupyter.org/github/fchollet/deep-learning-with-python-notebooks/blob/master/5.2-using-convnets-with-small-datasets.ipynb#Data-preprocessing
[18]:https://nbviewer.jupyter.org/github/fchollet/deep-learning-with-python-notebooks/blob/master/5.2-using-convnets-with-small-datasets.ipynb#Using-data-augmentation
[19]:https://github.com/keras-team/keras-applications
[20]:https://keras.io/api/applications/
[21]:https://nbviewer.jupyter.org/github/fchollet/deep-learning-with-python-notebooks/blob/master/5.3-using-a-pretrained-convnet.ipynb
[22]:https://nbviewer.jupyter.org/github/fchollet/deep-learning-with-python-notebooks/blob/master/5.4-visualizing-what-convnets-learn.ipynb#Visualizing-intermediate-activations
[23]:https://nbviewer.jupyter.org/github/fchollet/deep-learning-with-python-notebooks/blob/master/5.4-visualizing-what-convnets-learn.ipynb
[24]:https://www.quora.com/What-does-the-terms-Top-1-and-Top-5-mean-in-the-context-of-Machine-Learning-research-papers-when-report-empirical-results#:~:text=The%20Top%2D1%20error%20is,among%20its%20top%205%20guesses.
[25]:https://www.analyticsvidhya.com/blog/2020/08/top-4-pre-trained-models-for-image-classification-with-python-code/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
[26]:https://nbviewer.jupyter.org/github/fchollet/deep-learning-with-python-notebooks/blob/master/6.1-one-hot-encoding-of-words-or-characters.ipynb
[27]:https://nlp.stanford.edu/projects/glove/
[28]:https://nbviewer.jupyter.org/github/fchollet/deep-learning-with-python-notebooks/blob/master/6.1-using-word-embeddings.ipynb
[29]:https://www.analyticsvidhya.com/blog/2020/08/top-4-sentence-embedding-techniques-using-python/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
[30]:https://radimrehurek.com/gensim/models/doc2vec.html#gensim.models.doc2vec.Doc2Vec
[31]:https://github.com/UKPLab/sentence-transformers
[32]:https://docs.google.com/spreadsheets/d/14QplCdTCDwEmTqrn1LH4yrbKvdogK4oQvYO1K1aPR5M/edit#gid=0
[33]:https://github.com/facebookresearch/InferSent
[34]:https://github.com/tensorflow/tfjs-models/tree/master/universal-sentence-encoder
