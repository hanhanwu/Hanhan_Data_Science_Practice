# Learning Notes 🌺

I have decided to systematically review all the details of deep learning, and organize all important concepts together.

## Fun Facts of Deep Learning
* Deep learning can find interesting features itself, so we don't need manually feature engineering in deep learning. 
  * But this is achievable only when the dataset is large enough, higher dimensions of the data (such as images), larger dataset is needed.
* [How are images saved in computers][44]
  * `N * M * C`
    * N is the number of pixels in height
    * M is the number of pixels in width
    * C is the number of channels
      * grey image (black and white) has 1 channel, RGB image has 3 channels (red, green and blue)
    * Each pixel has a value between 0 and 255, smaller values is darker color, higher value is brighter color (0 is black, 255 is white)  

## [Deep Learning in Time Series / Sequential Analysis Learning Notes][46]

## Tensorflow Keras Tips
### Sequantial Model API vs Functional API
* [Sequantial Model API Example][95]: layers of NN are added into the model in a sequence
  * 1 input and 1 output only 
* [Functional API Example][96]: A layer is an instance that accepts a tensor as an argument and at the same time its output is another tensor that can be used as the argument of another layer. The model is a function between one or more input and output tensors.
* Functional API enables building more complex networks that can't be done in Sequential Model API
### Callbacks
* These callback functions can be called in `model.fit()` [like this][97]
* `checkpoint` helps saving the trained model with best validation data performance
  * To load the saved model, you can call `load_model()` 
* `lr_scheduler` helps reduce the learning rate during training after a certain number of epoches, [like this][98]
  * `lr_schedule()` is called after every epoch during the training 
* `lr_reducer` helps reduce the learning rate by a certain factor if the validation loss hasn't been improved after a certain number of epoches (patience)
  * [Like this example][99], `patience=5`
### [`fit()` vs `fit_generator()` vs `train_on_batch()`][103]
* `fit_generator()` function assumes there is an underlying function that is generating the data for it, such as data generator like [Siamese image pairs generation][102]
* `train_on_batch` function accepts a single batch of data, performs backpropagation, and then updates the model parameters


## Data Preprocessing Methods
### Collected Preprocessing Methods
#### Used in Computer Vision
* [Reshape MNIST digits data to SVHN data format][76]
* [Image Processing using numpy][77]
  * It includes: Opening an Image, Details of an image, Convert numpy array to image, Rotating an Image, Negative of an Image, Padding Black Spaces, Visualizing RGB Channels, Colour Reduction, Trim Image, Pasting With Slice, Binarize Image, Flip Image, An alternate way to Flip an Image, Blending Two Images, Masking Images, Histogram For Pixel Intensity 
* [Multi-threading data generator for object detection][93]
* [Convert RGB images to gray][74] and [reshape the images][75]
* Raw images to NN readable input 
  * [Keras ImageDataGenerator][17] will make below steps easier
    * Convert JPEG to RGB
    * Resize all the images to the same size
    * Convert images into floating point tensors
    * Rescale the pixel values (between 0 and 255) to the [0, 1] interval
    * "Batch size" is the number of samples in each training step
      * Because you can't pass the entire dataset into NN all at once, need multiple batches
      * `training_batchs = total training sample size / training batch size`
      * `validation_batchs = total validation sample size / validation batch size`
      * It's recommended to have batch size to be the power of 2, for GPU optimization purpose
    * 1 "Epoch" will process the entire dataset to update weights
    * "Number of iterations (or number of steps) = total dataset size/batch size", it's the number of steps needed to complete 1 epoch
    * Seperate data and labels
      * Labels are created based on directories, different classes of images are put in different directories
  * [Another example of Keras ImageDataGenerator for data augmentation, multiple choices][100]
* [Image batch preprocess for federated learning][43]
  * For each 28*28 image, it flatten to 784 one dimensional matrix, then divided by 255 to convert the values into [0,1] range because the pixel values are between 0 and 255
  * The preprocessed image can also be reshaped back to 28*28
* [How to convert an image to image tensor][22]
  * The image tensor can be used to understand the output of each activation layer
* [How to generate Siamese images][102]
  * "Siamese" means image pairs, normally is the pair of an image and its transformation
  * The image pairs can be used in evaluating unsupervisde image labeling method, such as MI methods like [IIC][90] or [MINE][92]

### Input Structure
* The image below is showing the data input requirements for MLP, CNN and RNN
  * [MLP input code][47]
  * [CNN input code][48]
  * [RNN input code][49]
<p align="center">
<img src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/images/NN_input_structure.PNG" width="700" height="300" />
</p>

### Shuffle the data before spliting
* When the arrangement of labels not distributed similar for training and testing data, better to shuffle the data before spliting into training and testing.

### Better to Normalize to the same range
* When features are in different ranges, better to normalize them into the same range.
  * For optimization functions, scaling the dependent & independent variables into [-1, 1] range ([0,1] range works too) or [-3, 3] range helps gradient descent, such as makes the converge faster
* [sklearn functions][7]

### How to Preprocessing Testing Data 
* The parameters used for preprocessing the training data should be generated from the training data only, and be applied to the testing data too.
  * This is also true for non-deep-learning models 
* For example, when you are using `mean` and `std` to normalize the data, these params should be generated from the training data, and be applied to the testing data.
  * It seems that `(value - mean)/std` is a common way used to normalize features when they are in different scales. This method doesn't guarantee they are on the same scale, but could make them on similar scales

### Multi-class lables
* Mehtod 1 - One-hot encoding
  * The labels have N columns, each column represents a class
  * Mark the labeled class as 1
* Method 2 - Integer labels
  * Each class is represented as an integer
  * 1 numpy array is enough for the labels
* [Example for both methods][8]

### Preprocess Text data
* Text data means a column contains longer text strings, such as comments, topics, etc.
* General Steps
  1. Choose top N toknized words appeared in all the text, N excludes stop words
  2. One-hot encoding: each distinct word as a column, each row reprensents a text, for word appeared in this text, mark it as 1, others mark as 0
  

## Layers & Dimensions
### Multiple Inputs
* We often see one input in NN, but in fact it allows multiple inputs. 
  * For example, such as `Concatenate()` which concatenate multiple inputs of the same shape at the concatenation axis to form 1 tensor. [Check the code example here][54]
    * The 2 branches in this code are using different "dilation rate". The Dilation rate decides the kernel's receptive field's size. Enlarging the dilation rate is used as a computationally effecient method. [See the defination of dialation rate][55].
    * Meanwhile, using different receptive field sizes for kernels here allows each branch to generate different feature maps.
  * Besides concatenation, we can also do other operations to put multiple inputs together, such as `add`, `dot`, `multiple`, etc.
* Note! The input here doesn't have to be the first layer of NN. Each input can be a sequence of layers, and finally all these inputs merged at a certain layer.
  * <b>It's like an NN has multiple branches, and each branch do different types of work with the same original input data</b> 
* Multiple inputs also has a cost in model complexity and the increasing in parameters

### Hidden Layers
* Having more units in a layer, also means having higher dimensional space representaton, will allow you to learn more complex representation.
  * Imagine this allows you to cut an image into smaller pieces for learning.
  * The drawback is more computationally expensive, and you might be learning unnecessary patterns which could even lead to overfitting.
* When there are N classes to predict, the dimensions of all the hidden layers better > N, otherwise the information loss will lead to overfitting
  
### Activation Functions
* In deep learning, layers like `Dense` only does linear operation, and a sequence of `Dense` only approximate a linear function. 
* Inserting the activation function enables the non-linear mappings.
* [Formula, prod & cons of activation functions][50]
  * `relu` is simple to compute and often used, can only be used in the hidden layers
  * `sigmoid` (output value in 0..1 range) and `tanh` (output value in -1..1 range) can be used in the last layer, for binary classification
    * `tanh` is also popularly used in the hidden layers of RNN 
  * `softmax` is often used in the last layer, usually for N multi-class probability output, the last layer will have N nodes (N > 2). It squashes the output into probabilities by normalizing the prediction, each class gets a probability and the sum is 1.
  * [Some other activation functions][51], such as `elu`, `softplus` and `selu`

### Output Layer
* 2 classes
  * 1 node, sigmoid for class probabilities
* N classes (N > 2)
  * N nodes, softmax for class probabilities
* Regression
  * Linear regression - 1 node without activation function

### Optimizer
* `rmsprop` and `adam` are often used
* It's recommended to start with a larger learning rate, and decrease it as the loss is getting closer to the minimum
* This paper (https://arxiv.org/pdf/2103.05127.pdf) written in 2021 talked about the model complexicity in optimization
#### Loss Function (Objective) & Optimizer & Regularizer
* The goal of deep learning is to reduce the loss function during training
* To minimize this loss value, optimizer is employed to determine how weights and bias should be adjusted during each training step
* Since the trained model will be used beyond the training data, regularizer ensures the trained more is generalizes to the new data 

### About K-fold Cross Validation with Keras
* Well, adding this part is because I failed this piece in Unity interview...
  * [Here's the example if the interviewer asks you to implement k-fold][5]
  * [Here's the example if you need k-fold with Keras in the work][6]
* After finding the optimal epoch number from cross validation, train the whole dataset with the optimal number of epoch.
  
### Evaluation Metrics
* Bigger networks do not always bring better performance
* [Full list of Keras metrics][4]
* [Top-n Error][24]
  * We often saw "top-1 error", "top-5" error appeared in papers.
  * If the correct answer is at least among the classifier’s top k guesses, it is said to be in the Top-k.
  * The Top-k error is the percentage of the time that the classifier did not include the correct class among its top k guesses.

#### Baseline Models
* We often do this in time series prediction. Often choose a baseline model to compare with ML models, and nothing should be worse than this baseline model :P
  * [Some ts baseline model ideas][36], in my previous code, there are a few methods we can borrow as the baseline
    * Naive Forecast (previous day = next day)
    * Naive Average Forecast
    * Moving Average
    * Exponetial Smoothing

#### Loss Functions
* The training process is trying to reduce the loss.
* `crossentropy`
  * Good choice for probability output.
  * It measures the distance between y_true and y_pred probabilities
  * `binary_` one can be used for probability of 2 classes
  * `categorical_` one can be used for probability of 2+ classes, and the label is in categorical format (such as one-hot encoding)
  * `sparse_categorical_` is similar to the use case of `categorical_`, but it's used when the labels are in integer format (nominal)
* `focal loss` vs `cross entropy` vs `balanced cross entropy`
  * Focal loss works better in dealing with the data imbalance issue in object detection of one stage detection
    * Data imbalance in object detection is often caused by much larger amount of background object and only a few objects (foreground) locations
  * Cross-Entropy loss is to penalize the wrong predictions more than to reward the right predictions. But because the final loss is summed over small losses from the entire image, when the data is imbalanced, the loss value won't be accurate to reflect the detected objects.
  * Balanced Cross-Entropy Loss is to add weights to both positive and negative classes. It does help differentiate the positive & negative, but cannot distinguish easy & hard examples
  * Focal loss down-weights easy examples and focus training on hard negatives.
    * After a lot of trials and experiments, researchers have found `∝=0.25 & γ=2` to work best
  * [reference][37]
* Some loss functions' formula
  *  `categorical_crossentropy` and `mean_squared_error` are good choices to be used after `softmax` layer
  *  `binary_corssentropy` is often used after `sigmoid` layer
  *  `mean_squared_error` is can be a choice after `tanh` layer
<p align="center">
<img src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/images/loss_functions.PNG" width="600" height="300" />
</p>

#### Evaluation Visualization
##### Epoch Plot
* Epoch - Evaluation Metrics plot
  * If there are metrics for training and 1 validation set, plot both curves for comparison will help find the stopping epoch
    * The stopping point is where starts the opposite trending between training and validation set
  * For k-fold cross validation, we can average the validation metrics from all the k folds, and in each fold, training & validation sets generates 1 evaluation value
    * The stopping point is where the trend fully changed
    * Especially useful when the dataset is small
  * If the accuracy curves look too noisy, you can try methods to smooth out the curves, such as using exponential moving average
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
  * Use in training data only
  * It helps reduce overfirtting and makes NN more robust to unseen data input
  * In Keras, bias, weights and activation functions can be regularized in each layer
  * L1 and L2
    * L1 regularization, where the cost added is proportional to the absolute value of the weights coefficients
    * L2 regularization, where the cost added is proportional to the square of the value of the weights coefficients
    * Therefore, both L1 and L2 favor smaller param values. However, NN with small params are less sensitive to the data fluctuations
    * No more layer should be added after a layer applied L1 or L2 regularization
  * [Dropout][10]
    * It's one of the most effective and most commonly used regularization method
    * It randomly drops (i.e. setting to zero) a number of output features of a layer during training, to prevent this fraction of units from participating in the next layer
      * This prevents NN from remembering the training data and can be more generalize to unforeseen input data 
    * Dropout rate is often set between 0.2 and 0.5
    * How to use dropout for recurrent NN
      * If we don't use dropout properly in recurrent NN, it will limit the learning instead of doing regularization
      * The same dropout mask (the same pattern of dropped units) should be applied at every timestep, instead of a dropout mask that would vary randomly from timestep to timestep. Because the same dropout mask at every timestep allows the network to properly propagate its learning error through time; a varying random dropout mask would instead disrupt this error signal and be harmful to the learning process.
      * In order to regularize the representations formed by the recurrent gates of layers such as GRU and LSTM, a temporally constant dropout mask should be applied to the inner recurrent activations of the layer
      * In keras, it has `dropout` and `recurrent_dropout` in recurrent layer as the solution
        * `dropout` specifies the dropout rate for input units of the layer
        * `recurrent_dropout` specifies the dropout rate of the recurrent units
* [Data Augmentation][18]
  * Training data only
  * "Data augmentation takes the approach of generating more training data from existing training samples, by "augmenting" the samples via a number of random transformations that yield believable-looking image"
  * Even though there is no same image in the training set, with this method, the inputs can still be heavily intercorrelated. Therefore it's necessary to add a `dropout` layer right after flattened data to reduce some dimensions in order to further reduce overfitting.
* Weight Initialization & Weight Constraint
  * [Tensorflow initializer][41]
  
## Convolutional Networks (Convnet)
* Convnets can learn <b>local, translation-invariant features</b>, so they are very data-efficient on perceptual problems. Therefore, even when the dataset is small (such as hundreds of images), you might still get a reasonable results.
* In CNN, a <b>kernel</b> can be visualized as a window that slides through the whole image from left to right, from top to bottom. 
  * This operation is called as "Convolution", which transforms the input image as a feature map
  * Kernel vs Filter
    * Kernel size is the size of the "grid" of pixels that you convolve, it's like your sliding window.
    * Filters is the number of such sliding windows that you want your network to learn.  
* [Keras Convnet][11]
* [simple CNN pyhton implementation][94], [Simple CNN for MNIST classification][95]
* [Conv2D][12]
  * "Input shape" is `(batch_size, image_height, image_width, image_channels)`, batch_size is optional
    * The width and height tend to shrink when we go deeper in convnet
    * However, in Keras `reshape()`, the order is `(sample_size, channels, width, height)`...
  * When there are multiple layers of Conv2D in a neural net, deeper layer gets larger number of batches, notmrally we choose batch size in a sequence of 32, 64, 128, ...
  * `image_channels` is also the image depth. [How to use opencv2 to find image channels][13]
    * If returns 2 dimensions (`len(img.shape=2`), the image channel is 1 for single channel; otherwise it's the third number returned in the output (also `len(img.shape=3` in this case)
  * `padding=same` will pad the borders with 0 to keep the original image size
  * `kernel` can be non-square, such as `kernel=(3,5)`
* [MaxPooling2D][16]
  * Pooling Layer - pooling is a down-sampling operation that reduces the feature map size
  * It performs down-sampling operations to reduce the dimensionality and creates a pooled feature map by sliding a filter matrix over the input matrix. See [example here][52], every patch of size pool_size * pool_size is reduced to 1 feature map.
    * This also translates to the increase of receptive field field, since now to represent the same size of feature map, you just need a smaller matrix 
  * `MaxPooling2D` chooses the max value from each patch, `AveragePooling2D` chooses the average value from each patch
    * Stratified convolution can do similar work too, such as `Conv2D(strides=2)` will skip every 2 pixels during convolution and will still have the same 50% size reduction effect
  * `pool_size` can be non-square, such as `pool_size=(1,2)`
* [What is `strides`][14]
  * The size (height, width) of the moving window
* Why `Flatten()`
  * `Dense` or `Dropout` layer is 1D, so if the output from Conv2D is more than 1D, it needs to be flatterned into 1D before entering into the Dense layer.
* `fit_generator()`
  * `steps_per_epoch = total training sample size / training batch size`, this is because in each epoch, it runs all the samples
  * `validation_steps = total validation sample size / validation batch size`
### Pretrained Convnet
* Better just to reuse the pretrained convolutional base and avoid using the densely-connected classifier. Because
  * The convolutional base has a more generic feature maps while densely-connected features are more specific
  * The convolutional base still contains object location info while densely-connect features do not
* Only choose earlier layers from the trained model if your new dataset has larger difference with the data used for the pretrained model
  * Because the earlier layer capture more generic feature maps
* [Keras pretrained applications][19]
  * [python examples][20]
  * [A bit more description on some of the pre-trained models][25]
* Method 1 - Pretrained Model + Standalone Densely-connected classifier
  * This method is fast and cheap to run, but doesn't allow you to do data augmentation and therefore may not reach to an optimal accuracy
    * It's possible to do with CPU
  * Need to reshape your new dataset as the output of the pre-trained convnet
  * [See example here][21]
* Method 2 - Extend Pretrained Model
  * It's costly to run, at least need the GPU
  * Before compile and train the model, need to freeze the the convolutional base. "Freezing" a layer or set of layers means preventing their weights from getting updated during training, in order to keep the representations learned by the trained model.
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

## ResNet
* ResNet introduces residual learning, which allows to build a very deep network while addresing the vanishing gradient problem.
  * Gradient Vanishing: Backpropagation follows the chain rule, there is a tendency for the gradient to diminish as it reaches to the shallow layers, due to the multiplication of small numbers (small loss functions and parameter values). However, if the gradients decrease and the parameters cannot update appropriately, then the network will fail to improve its performance.
  * ResNet allows info flows through shortcuts to the shallow layers, in order to relieve the gradient vanishing problem
### [The implementation of ResNet][56]
#### Core
* Core logic of each stack of residual block, and this is how info shortcut happens:
  * `x1 = H(x0)`
    * `H()` is `Conv2D-BN-ReLU` in v1
      * `BN` is Batch Normalization 
      * The benefit of `BN` is to stablizing the learning by normalizing the input to each layer to have 0 mean and unit variance
    * `H()` is `BN-ReLU-Conv2D` in v2
    * By the way, H() is used in VGG, for example a 18-layer VGG has 18 H() operations before the input image is transformed to 18th layer of feature map
  * `x2 = ReLU(F(x1) + x0)` 
    * `F()` is `Conv2D-BN`, also known as "residual mapping"
    * `+` is done by concatenation, element-wise addition
    * The shortcut connection doesn't add extra params nor extra computational complexity
  * Linear projection to match different dimensions
    * `F(x1)` and `x0` should have the same dimentions. When the dimensions are different, then at the end of each residual block (except the first stack in v1), before concatenation, linear projection is applied to match different dimensions
    * Linear projection is using a `Conv2D()` with 1x1 kernel `kernel=1` and `strides=2`
      * NOTE: when strides > 1, it equivalents to skipping pixels during convolution
    * This layer is similar to the "Transition layer" used in DenseNet, which is used to match different dimensions
* `kernel_initializer='he_normal'` to help the convergence when back propagation is taking the place
* ResNet is easier to converge with `Adam` optimization
* The residualBlock in v1 vs v2. V2 could improve the performance a bit.
<p align="center">
<img src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/images/resnet_v1_v2_diff.PNG" width="400" height="250" />
</p>

## DenseNet
* Instead of using shortcuts as what ResNet does, DenseNet allows all the previous feature maps to be the input of the next layer
### [The Implementation of DenseNet][57]
#### Core
* Stacks of Denseblocks bridged by Transition layers
* Within each Stack of Denseblock
  * There are multiple stacks of Bottleneck layers
    * `num_bottleneck_layers = (depth - 4) // (2 * num_dense_blocks)` 
  * Within each Bottleneck layer 
    * "Bottlenck layer" formed by `BN-ReLU-Conv2D(1)-BN-ReLU-Conv2D(3)`
      * By having `BN-ReLU-Conv2D(1)`, bottleneck leyer is trying to prevent the feature maps from growing to be computationally inefficient. 
      * With `Cov2D(1)` with a `filter_size=4*k` to do dimensional reduction
        * `k` is "growth rate", meaning the number of feature maps generated per layer 
    * If there is no data augmentation, then `dropout()` will be added after `BN-ReLU-Conv2D(1)-BN-ReLU-Conv2D(3)`
    * At the end of this bottle layer, concatenate the input and output of `BN-ReLU-Conv2D(1)-BN-ReLU-Conv2D(3)`
  * Transition layer transforms the feature map to the smaller size needed in the next Denseblock
    * The transition layer is made of `BN-Conv2D(1)-AveragingPolling2D`
      * Assuming the feature maps has a dimension with (64, 64, 512)
      * `Conv2D(1)` does the comrepssion work, which reduces the number of feature maps. The compression factor is 0.5 in DenseNet, so the feature maps' dimension will become (64, 64, 256)
      * Averaging Pooling2D does dimensional reduction by halving the feature maps, so in this example, the feature maps' dimension will become (32, 32, 256)
    * Last denseblock doesn't need the transition layer 
* DenseNet converges better with `RMSprop` optimization
* It takes very long time to run DenseNet while the performance may not be much better than ResNet

## RNN
* RNN is trying to help analysis on a sequence, so that there could be more context, and therefore they are often used in NLP.
* RNN can also be applied to images, see [example here][53]
  * Each image can be considered as "height" number of sequences that each sequence with length of "width"
* If CNN is characterized by the convolution of kernels across the input feature map, RNN output is a function not only of the current input but also of the previous output or hidden states
### RNN Tips
* Deal with very long sequence in a cheaper way
  * Add 1 D convnet before RNN layer, the convnet will turn the long input sequence into much shorter (downsampled) sequences of higher-level features. This sequence of extracted features then becomes the input to the RNN part of the network. the performance may not improve but it's cheaper to run with RNN only
  * [check example here][38]
* For all the RNNs, increasing the number of "units" can increase the learning capacity
* Stacking recurrent layers will also increase the learning capacity
  * But for simple problem, it may not necessary
  * Make sure adding more recurrent layers won't make overfitting too bad
  * In keras, `return_sequences=True` for each recorrent layer except the last one
* Bidirectional RNN
  * It's learning both forward and backward of a sequence.
  * Often useful in NLP, to help capture patterns got overlooked in forward learning. However, for time series problems that more recent data has higher influence, forward learning is better.
  * In keras, add `layers.Bidirectional` outside of the recurrent layer
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

## Autoencoder
* Autoencoder is a NN architecture that attempts to find the compressed representation of the given input data. It can learn the code alone without labels and therefore is considered as "unsupervised learning". 
### Main Architecture
* Input (`x`) --> Encoder (`z = f(x)`) --> Latent Vector (`z`) --> Decoder (`x_approx = g(z)`)
  * Latent Vector is a low-dimensional compressed representation of the input distribution
  * It's expected that the output recovered by the decoder can only approximate the input
  * The goal is to make `x_approx` as similar as `x`
    * So the loss function is trying to maximize the similarity between the distrobutions of `x` and `x_approx`
    * If the decoder's output is assumed to be Gaussian, then the loss function boils down to `MSE`
    * The autoencoder can be trained to minimize the loss function through backpropagation. Keras does backpropagation automatically for you
      * Just make sure the loss function is differentiable, which is a requirement of backproporgation 
* Key Components
  * Encoder
    * It outputs the "latent vector" which has a lower dimension of the input
      * The low dimension forces the encoder to capture the most important features of the input
      * The dimension of latent vector `z` indicates the amount of salient (important) features it can represent
        * In the application examples below, denoising autoencoder only has 16 latent-dims but in colorization example the latent-dims is 256. This is because the denoise one was using black-sand-white data while the colorization was using RGB data which needs to capture more complex features
   * Decoder
     * It reads the latent vector as the input and output the compressed result with the same size as the original input 
   * Both encoder and decoder are non-linear functions, and therefore they are be implemented in NN
* Visualized Autoencoder's Structure
  * The example here was used for MNIST dataset (0 ~ 9 digits in grey scale). I found it's very helpful in showing the number of filters and the dimensions of feature maps' changes throughout the network
  * [The implementation of this basic Autoencoder][101]
    * For more complex dataset, we can create deeper encoder & decoder, as well as using more epoches in training 
    * In the encoder part, several `Conv2D()` were used with increased number of filters
      * The reason when network goes deeper and the filters can increase is, in shallower layers, we want to exclude the "noise" from the raw pixel as much as possible, and in deeper layers, we can extra more detailed features from previous denoised feature maps
    * In the decoder part, the same amount of `Conv2DTransposed()` were used with reversed filter amount as what's in encoder
  * [We can plot the latent vector as 2-dim plot][58], in order to understand the latent vector output better
<p align="center">
<img src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/images/encoder.PNG" width="600" height="350" />
<img src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/images/decoder.PNG" width="600" height="450" />
 <img src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/images/autoencoder.PNG" width="600" height="250" />
</p>

### Autoencoder Applications
* [Autoencoder classification][61]
  * It does both denoising and digit classification
  * Within the autoencoder here, instead of using `Conv2D()` for encoder and `Conv2DTransposed()` for decoder, it's using `BN-ReLU-Conv2D-MaxPolling2D()` for encoder and `BN-ReLU-Conv2DTransposed-UpSampling2D` for decoder
  * For classifier, it needs labels y_train to train the model
  * [Denoising without classification][59]
    * It doesn't have classification part, therefore labels are not needed 
* [Autoencoder colorization][60]
  * The input is gray impage and the output is color image 
  * Its structure is basic, just uses `Conv2D()` for encoder and `Conv2DTransposed()` for decoder
    * Filters for each layer increased to capture more feature details
    * Latent-dims increased to increase the capacity fo getting the most important features

## Generative Adversarial Networks (GAN)
* GANs belong to the family of generative models. Different from autoencoder, generative models can create new and meaningful output given arbitrary encodings
* GANs train 2 competing components
  * Discriminator: it learns to discriminate the real and the fake data 
  * Generator: it's trying to fool the discriminator by generating fake data and pretends it's real
  * When it reaches to the point that the Discriminator can no longer tell the difference between real and fake data, it will be discarded and the model will use Generator to create new "realistic" data
  * If the data input is images, both generator and discriminator will be using CNN; if the input is single-dimensional sequence (sucb as audio), both generator and discriminator will be using RNN/LSTM/GRU
* Since the labeling process is automated, GANs are still considered as unsupervised learning
  * The labeling here means "fake" (generated by generator) or "real" (real data) labels, not other types of "labels" used in GANs 
* Comparing with other types of NN, GANs are notoriously hard to train, and prone to mode collapse
  * Mode Collapse: the generator collapses which produces limited varieties of samples. For example, 0 ~ 9 digits, it only provides a few modes such as 6,7,8. The main cause is, in each iteration of generator over-optimizes for a particular discriminator, and the discriminator never manages to learn its way out of the trap. As a result the generators rotate through a small set of output types. 

### DCGAN vs CGAN
#### DCGAN
* Core Architecture
  * Left is discrinminator training process
    * The generator is to generate the fake data to serve as the discriminator's fake data input (`label=0`) 
    * The discrimonator is trained with both real and fake data, in order to distinguish real or fake data better
  * Right is generator training (adversarial training) process  
    * The generator is to generate the fake data to serve as the discriminator's fake data input (`label=1`) 
    * The discriminator weights are frozen, its only input will be fake data generated by the generator with label as 1
    * This step is trying to train the generator to generate fake data that looks more real
<p align="center">
<img src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/images/DCGAN_discriminator_training.PNG" width="400" height="300" />
<img src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/images/DCGAN_adversarial_training.PNG" width="400" height="300" />
</p>

* Loss function
  * The purpose of loss function of DCGAN is to maximize the chance of making the discriminator believes the generated fake data is real during the generator training
* [The Implementation of DCGAN][64]  
  * Opposite to `Conv2D()`, `Conv2DTransposed()` can create an image given feature maps, this is also why it's used in autoencoder too 
  * In the discriminator here, it didn't use `BN` or `ReLU`. It's known that DCGAN became unstable when you insert `BN` before `ReLU` in the discriminator
  * Due to custom training, `train_on_batch()` is used instead of using `fit()`
  * The process of `discrimintor training then generator training (adversarial training)` will be repeated in multiple train steps
  * Both generator and discriminator were using `RMSprop` optimizer
  * Settings the learning rate in adversarial training as half as the learning rate in discrimonator training will result in more stable training
    * If the discriminator learns faster, the generator's params may fail to optimize
    * If the discriminator learns slower, the gradients may vanish before reaching to the generator
  * Detailed model structure:
    * The 100-dim z-vector is a latent vector (similar to the latent vector in autoencoder), using to generate noise in [-1, 1] range using uniform distriution 
<p align="center">
<img src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/images/DCGAN_MNIST.PNG" width="500" height="350" />
</p>

#### CGAN
* It's quite similar to DCGAN, except the additional conditions applied, which was used to specify which class to generate
* Core Architecture
  * Left is discrinminator training process. Comparing with DCGAN, 2 one-hot labels have been added into the input of both generator and discriminator:
    * A fake one-hot vector has been added as part of the generator input
    * This same fake one-hot vector later has been concatenated with a real one-hot vector and get into the discriminator
      * Since the fake data came from random noise, the fake class here is also random
    * Another one-hot labels got into the discriminator's input too, along with the real data
      * It specifies which class the real data belongs to 
  * Right is generator training (adversarial training) process, comparing with DCGAN, a newly generated one-hot laabels has been added as the input of generator and discriminator:
    * It's a fake one-hot vector
<p align="center">
<img src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/images/CGAN_distriminator_training.PNG" width="400" height="300" />
<img src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/images/CGAN_adversarial_training.PNG" width="400" height="300" />
</p>

* [The Implementation of CGAN][65] 
  * Comparing with DCGAN:
    * For generator, the added one-hot label is concatenated with the images and get into the same generator structure used in DCGAN
    * For discriminator, a `Dense()` layer and `Reshape()` have been applied to the one-hot label, then concatenated one-hot label and the images has been sent to the same discrimonator structure used in DCGAN
    * `y` is the binary label to indicate whether the image is real (y=1) or fake (y=0)
  * The fake one-hot label was also randomly generated as the noise
<p align="center">
<img src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/images/CGAN_MNIST.PNG" width="500" height="350" />
</p>
 
### Improved GANs
* WGAN (Wasserstein GAN) is trying to improve model stability and avoid mode collapse by replacing the loss function based on Earth Mover's Distance (EMD), also known as Wasserstein 1, which is a smooth disfferentiable function even when there is no overlap between the 2 probaiblity distributions.
* LSGAN (Least Square GAN) is trying to improve both model stability and generated images' perceptive quality on DCGAN, by replacing sigmoid cross-entropy loss with least squares loss
* ACGAN (Auxiliary GAN) is trying to improve both model stability and generated images' perceptive quality on CGAN, by requiring its discrimiator to perform an additional classification task
* [Reusable discriminator, generator and `train()` for multiple GANs][67]
#### WGAN
* In DCGAN and CGAN, the loss function is trying to minimizing the DISTANCE (JS distance, Jensen Shannon distance) between the 2 distributions (real data's distribution and fake datta's distribution). However, when 2 distributions have no overlap, with JS Distance, there is no smooth path to close the gap between them, and the training will fail to converge
  * The "distribution" here is probability distribution
  * JS is a divergence based on KL (Kullback-Leibler) divergence, but KL is asymmetric while JS is symmetric and finite 
* The idea of EMD is, it's a measure of how much mass should be tranported in order to match the 2 probability distributions
  * EMD can be interpreted as the least amount of work needed to move the pile of dirt p to fill holes q
  * When there is no overlap between 2 distributions, it can servee as a logical loss 
* WGAN is same as DCGAN, except:
  * The fake data has label as -1 in WGAN while in DCGAN the fake data's label is 0
    * Because of the opposite sign in real & fale labels in WGAN, in the discriminator's training process, WGAN trains a batch of real data and then trains a batch of fake data, instead of training them together as what DCGAN does
      * This prevents the gradient from vanishing 
  * WGAN trains the distriminator `n_critic` iterations before training the generator for 1 iteration; In DCGAN, `n_critic=1`
  * The loss function in WGAN is using`wasserstein_loss`, while DCGAN is using `binary_crossentropy`
    * `wasserstein_loss` is the loss function that the generator tries to minimize while the discriminator tries to maximize
    * At the end of each epoch of gradient updates (each round of "critic"), discriminator weights were clipped between lower and upper bounds (such as between (-0.01, 0.01)), in order to satisfy Lipschitz continuity (having bounded derivatives and be continuously differentiable)
      * It constraints the discriminator to a compact parameter space 
  * See [WGAN implementation and wasserstein_loss][66], [DCGAN implementation][64]
* But WGAN doesn't take care of the generated images' quality
#### LSGAN
* Fake samples on the correct side vs Fake samples' distribution is closer
  * Idealy, the fake samples' distribution should be as close as the real samples' distribution. However, in DCGAN, once the fake samples are already on the correct side of the decision boundary, the gradients vanish.
    * Fake samples that are still far away from the decision boundary will no longer move towards the real samples' distribution
    * This prevents the generator from further improving the quality of generated fake data in order to move closer to the real samples' distribution.
  * The solution is to replace the loss function with least square loss function, the gradients won't vanish as long as the fake samples' distribution is still far from the real samples' distribution. This will motivate the generator to keep improving its estimate of real density distribution, even if the fake samples are already on the correct side of the decision boundary.
* LSGAN is same as DCGAN, except:
  * The loss function for both generator and discriminator is `MSE` or `L2` in LSGAN (replaced binary cross entropy)
    * Least square error (L2) minimizes the "total" euclidean distance between a line and the data points. MSE is mean squared error, a good metric to evaluate the distance in average
  * There is linear activation or the avtivation is None in LSGAN
    * DCGAN can use 'sigmoid', 'relu', 'leakyrule'
    * LSGAN uses None or 'linear' for the activation function
  * See [LSGAN implementation][68], [DCGAN implementation][64]
#### ACGAN
* ACGAN makes the improvement upon CGAN by adding an auxiliary class decoder network. 
* ACGAN assumes forcing the network to do additional work will improve the performance of the original task.
  * The additional work is the added classification
    * This added classification also helps ACGAN decides which class to generate, same as what CGAN does 
  * The original task is fake data generation
* ACGAN vs CGAN
  * The data input doesn't have to be images, but here let's use images as an example 
  * `discriminator.train_on_batch()`
    * In CGAN, it takes the real+fake images and real+fake image labels as input, the binary labels as output
    * In ACGAN, it takes the real+fake images as the input, both binary labels and the real+fake image labels are the output
  * `adversarial.train_on_batch()`
    * In CGAN, it takes the noise and fake labels as input, the binary labels as output
    * In ACGAN, it takes the noise and fake labels as the input, both binary labels and the fake labels are the output
  * ACGAN added a loss function for classification
    * The loss function to predict fake or real is 'binary_corssentropy', both CGAN and ACGAN have it
    * The loss function to predict for classification is 'categorical_crossentropy'
      * In discriminator training stage, it's asking the discriminator to correctly classify both real and fake data
      * In the adversarial training stage, it's asking the discriminator to correctly classify the fake data 
  * [ACGAN implementation][69], [ACGAN discriminaton & generator implementation][67], [CGAN implementation][65]
    * `softmax` was used as the activation in ACGAN. `Softmax` is often used for multi-class probability output. This is used for the added auxiliary classification
    * The loss function later for classification is `categorical_crossentropy` 
* ACGAN architecture
  * Comparing with CGAN, there is an Auxiliary network being added after the `Flatten()` function of the discrimonator, which do the the extra classification work
  * Same as CGAN, "y label" here is one-hot label, but in the code, `y` is the binary label
<p align="center">
<img src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/images/ACGAN_DG.PNG" width="450" height="450" />
<img src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/images/ACGAN.PNG" width="500" height="450" />
</p>

### Disentangled Representation GANs
* GANs can also learn disentangled latent codes or representations that can vary attributes of the generator outputs
* A disentangled code or representation is a tensor that can change a specific feature or attribute of the output data while not affecting other attributes
* Comparing with the original GANs, if we're able to seperate the representation into entangled and disentangled interpretable latent code vectors, we will be able to tell the generator what to synthesize
  * For example, in an image, with disentangled codes, we can indicate thickness, gender, hair style, etc. 

#### InfoGAN
* InfoGAN learns the disentangled representation in an unsupervised way, by maximizing the mutual info between the input codes and output observation
  * In order to maximize this mutual info, InfoGAN forces the generator to consider the latent codes when it synthesizes the fake data
  * The latent codes can be discrete or continuous
* InfoGAN vs ACGAN
  * In InfoGAN, the binary labels and multi-class (one-hot) labels used in ACGAN are considered as "discrete codes", besides InfoGAN added extra continuous codes
    * Same as ACGAN, the binary discrete code helps the discriminator predict real of fake image; the multi-class discrete code helps image classification and decide which class to generate from the generator
    * Each continuous code controls the disentangled attributes of the image that the generator is going to generate. InfoGAN uses `mi_loss` for EACH continuous code
      * The goal is to minimize mi_loss which will maximuze the mutual info between the codes and the generator output 
* InfoGAN Architecture, comparing with ACGAN:
  * The disentangled codes have been added as the input of both discriminator and generator
    * `z` indicates both z-vector, th eentangled codes (noise code)
    * `c` are the codes: both discrete and continuous codes
    * `x` here include both data input (real images/noise, discrete codes, continuous codes)
  * The auxiliary decoder here looks the same as the one in ACGAN, it's used to deal with discrete code, which is the same one-hot labels used in ACGAN
  * MI (mutual info) loss for continuous codes
<p align="center">
 <img src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/images/InfoGAN_DG.PNG" width="400" height="600" />
<img src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/images/infoGAN.PNG" width="500" height="600" />
</p>

* [InfoGAN implementation][70], [InfoGAN generator and discriminator implementation][67], [ACGAN implementation][69]
  * The disentangle codes in InfoGAN
    * In this example, each continuous code is drawn from a normal distribution with 0.5 std and 0 mean
      * `codes` in InfoGAN's implemenattion here means the continuous codes 
      * `sigmoid` activation function was used for continuous code
      * Same as CGAN and ACGAN, `sigmoid` was used for the binary discrete code too
      * Same as ACGAN, `softmax` was used for multi-class discrete code 
  * Loss functions 
    * `λ` is the weight of loss functions, a small positive constant
      * For continuous code, recommends to have `λ<1`, and in the code here, it's using 0.5
      * For discrete code, recommends to have `λ=1`
        * As you can see both binary discrete code and multi-class discrete code get weight as 1 
  * n attributes
    * You can add whatever number of disentangled codes (n=2 in this example)

#### StackedGAN
* StackedGAN uses a pretrained encoder or classifer to help disentangle the latent codes.
* It breaks a GAN into a stack of GANs, that each is trained independently in discriminator-adversarial manner with its own latent code:
  * Each `encoder_i` extracts certain features from the image, then each `GAN_i` learns to invert the process of its conrresponding `encoder_i` to generate fake images from the extracted real features
  * Each `GAN_i` uses latent code `z_i` that conditions its generator output, which means an `z_i` can alter specific attributes of the generated image
  * Simplified overall view:
<p align="center">
<img src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/images/stackGAN.PNG" width="500" height="350" />
</p>

##### StackedGAN Core Architecture
* An overall view of a StackedGAN with 2 encoders
  * Different from other GANS, StackedGAN has added "Conditional Loss", "Entropy Loss"
<p align="center">
<img src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/images/stackGAN_DG.PNG" width="400" height="450" />
</p>

* Conditional Loss is measured by `L2` or `MSE` (mean squared error), it's the difference between the generator input and the encoder recovered input
  * As we can see in the flow chart, when the generator synthesizes the output `f_i'` from noise code `z_i`, with conditional loss, it forces the generator to consider the input `f_i+1` at the same time 
<p align="center">
<img src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/images/stackGAN_LC.PNG" width="300" height="250" />
</p>

* Entropy Loss is measured by `L2` or `MSE` (mean squared error), it's the difference between the recovered noise and the input noise
  * Entropy loss forces the generator to consider the noise code `z_i` at the same time 
  * Q network recovers the noise from generator output
<p align="center">
<img src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/images/stackGAN_LE.PNG" width="300" height="250" />
</p>

* Similar to multiple other GANs, it has `binary_crossentropy` to discriminate between real and fake images; it has `categorical_crossentropy` to classify images
  * This binary discrimination is the "Adversarial Loss" in StackedGAN
  * The multi-class classification is considered as a type of conditional loss here, even though it's using neither L2 nor MSE 
<p align="center">
<img src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/images/stackGAN_LA.PNG" width="300" height="250" />
</p>

##### [StackedGAN implementation][71]
* General Process:
  * Discrimonators are trained with real & fake images, one-hot labels and latent codes
  * Adversarial is trained next with fake images pretending to be real, and corresponding one-hot labels as well as latent codes
* Except entropy loss has weights as 10, other loss functions have weights 1
* 2 Encoders
  * Encoder0 reads real images as the input and output feature1 (intermediate latent features)
  * Encoder1 reads feature1 as input and outputs the predicted labels (predicted image classes)
<p align="center">
<img src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/images/stack_GAN_encoders.PNG" width="300" height="350" />
</p>

* 2 Generators
  * Generator1 reads noise code z1 and one-hot labels to generate fake_feature1
    * This generator is StackedGAN specific
  * Generator0 reads fake_feature1 and noise code z0 to generate the fake image
    * This generator [shares common structure as other GANs][67] 
<p align="center">
<img src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/images/stackGAN_generator.PNG" width="350" height="350" />
</p>

* 2 Discrimonators
  * Both Discrimonator_i outputs recovered latent code z_i, and the probability of real
  * The input of Discrimonator0 is image, while the input of Discrimonator1 is feature_i
<p align="center">
<img src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/images/stackGAN_discriminator.PNG" width="400" height="350" />
</p>

* 2 Adversarials models
  * Both encoder and discrimonator's weights are frozen for each adversarial model
  * Adversarial0 is formed by generator0, discrimonator0 and encoder0
    * The loss functions here are adversarial loss (the probability of being real), entropy loss and conditional loss 
  * Adversarial1 is formed by generator1, discrimonator1 and encoder1
    * The loss functions here are adversarial loss (the probability of being real), entropy loss and conditional loss (the image classification error)
##### StackedGAN vs InfoGAN
* InfoGAN has simpler structure and faster to train 
* The latent codes between GAN, InfoGAN and StackedGAN
  * Maybe this is also why 2 encoders in StackedGAN alters 2+ attributes while 2 codes in InfoGAN alters 2 attributes. But the overall cost of InfoGAN looks much smaller 
<p align="center">
<img src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/images/infoGAN_vs_stackGAN_latent_vector.PNG" width="500" height="150" />
</p>

### Cross Domain GANs
* This technique can be used in computer vision when an image in the source domain is transferred to the target domain.

#### CycleGAN
* This method doesn't need aligned source and traget images in the training.
  * Most often, the aligned image pairs are not available or expensive to generate. Better to have a model that doesn't need aligned image pairs. By contrast, methods like pix2pix which required aligned pairs have limited capability.
* It's symmetric, which means the translation between the target and the source images can be reversed
##### Core Architecture
* The main structure of CycleGAN
  * In the forward cycle, the input data is real source data, while in the backward cycle, the input is real target data.
  * In both forward and backward cycles, both have cycle consistency check, to minimize the differences between the input data and reconstructed data. Also because of the cycle consistency check, we don't need paired source and target data.
    * The cycle consistency check implies that, although we are transfering domain x to domain y, the original features in x should remain intact in y and be recoverable
    * Cycle consistency check uses L1 loss (MAE, mean absolute error) so that the reconstructed images can be less bluring than using L2 loss (MSE, mean squared error)
  * CycleGAN is symmetric. Forward cycle GAN is identical to the backward cycle GAN, but have the roles of the source data x and target data y reversed
    * Generator F is just another generator borrowed from the backward cycle GAN
  * <b>The objective of CycleGAN is to have generator G learn to synthesize fake target data y that can fool discriminator Dy in forward cycle, and have generator F learn to synthesize fake source data x that can fool discrinminator Dx in backward cycle</b> 
<p align="center">
<img src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/images/cycleGAN.PNG" width="500" height="500" />
</p>

* Identity Regularizer
  * The color composition may not be successfully transfered from the source image to the fake target image. To deal with this issue, identity regularizer is added to CycleGAN's network. It adds cycle identity loss function to both forward and backward cycles
  * But this piece is not needed when the source and target images are having different number of channels, such as the translation between grey and colored images
<p align="center">
<img src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/images/cycleGAN_identity_regularizer.PNG" width="550" height="550" />
</p>

* U-network
  * The generator of CycleGAN learns the latent represnetation of the input data distribution and translates to the target data distribution. This is exactly what does an autoencoder do. However, the low level features are shared between encoder and decoder layers in autoencoder, which is not suitable for image translation. U-network is used to solve this problem.
  * From the example U-net below we can see, when there are `n` encoder & decoder layers, the `encoder_i` will be connected to `decoder_n-i`. This structure allows specific feature-level info flows between corresponding encoder & decoder layer
  * IN (Instance Normalization) is BN (Batch Normalization) per image/sample or per feature
    * <b>In style transfer, it's important to normalize the contrast per sample, not per batch</b>. IN is equivalent to contrast normalization while BN breaks contrast normlaization.
  * The encoder is using `IN-LeakyReLU-Conv2D` while the decoder is using `IN-ReLU-Conv2DTransposed`. But in the code, `IN` is optional, `LeakyReLU` and `ReLU` can replace each other in both encoder and decoder
<p align="center">
<img src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/images/U_network.PNG" width="600" height="400" />
</p>

* PatchGAN
  * Left is the discriminator without PatchGAN, right has PatchGAN. 
  * The difference is the last layer. Without PatchGAN, `Dense(1)` is used at the end to predict the probability of whether an image is real. However in large images, computing this probability using 1 number is param-inefficient and the image quality from the generator later can be poor.
  * With PatchGAN, it divides each image into a grid of patches, and the discriminator will make prediction for each patch. Meanwhile, the whole image looks more real if its patches look more real
    * Patches can overlap 
<p align="center">
<img src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/images/patchGAN.PNG" width="600" height="400" />
</p>

##### [CycleGAN implementation][72]
* Training Process, repeat in each training step:
  * Train the forward discriminator with a batch of real target data (label=1), and a batch of fake target data (label=0), minimize the MSE loss for this discriminator
  * Train the backward discriminator with a batch of real source data (label=1), and a batch of fake source data (label=0), minimize the MSE loss for this discriminator
  * Train the forward and backward generators in the adversarial network, minimizing the MSE loss for generator and MAE loss for the cycle consistency checks
    * The input of forward generator is a batch of fake target data with label=1 
    * The input of backward generator is a batch of fake source data with label=1 
    * The weights of discriminators are frozen
* Loss functions
  * MAE (L1 loss) is used for cycle consistency check
    * CycleGAN suggests to give its weight as `λ=1` 
  * MSE (L2 loss) is used for generator and discriminator losses. This is inspired by LSGAN, that by replacing binary_crossentropy with MSE improves the perceptual quality
    * CycleGAN suggests to give its weight as `λ=10` to give more importance to the cycle consistency check
  * MAE is also used for cycle identity loss
    *  `λ=0.5`
    *  It's optimized during the adversarial training step
* The network is symmetric, therefore it can do grey <--> color 2 directions image translation
  * In fact, in this code, trained `g_target`, the target generator (backward generator) can do the reversed work by converting color image to gery 
* Tips
  * When the 2 domains of the source and the target are drastically different, suggest to have a larger kernel_size. 
    * For example, in [CycleGAN implementation][72], `mnist_cross_svhn` is trying to translate images between MNIST digits and SVHM street view numbers, comparing with the color tranlation which uses `kernel_size=3`, this case is using `kernel_size=5`
  * Also as `mnist_cross_svhn` shown, the results of CycleGAN may not be sematic consistent. To address this issue, we can try `CyCADA (Cycle-Consistent Adversarial Domain Adaptation)` which adds a semantic loss to improve the semantic consistency
    * "Semantic Inconsistent" here means, the image translation tranlated the source digit to the wrong target digit, even though the style has been translated right between the 2 domains. This is also called as "label flipping" 

### [ProGAN][78]
* It involves training by starting with a very small image and then the blocks of layers added incrementally so that the output size of the generator model increases and increases the input size of the discriminator model until the desired image size is obtained. 
* During the training process, it systematically adds new blocks of convolutional layers to both the generator model and the discriminator model. 
  * This incremental addition of the convolutional layers allows the models to learn coarse-level detail effectively at the beginning and later learn even finer detail, both on the generator and discriminator side.
* Batch Normalization is not used here, instead of that two other techniques are introduced here, including pixel-wise normalization and minibatch standard deviation.

### Other
* [A trick when tunning GAN][39]

## Variational Autoencoders (VAE)
* Same as GANs, VAEs also belong to generative models family
* Both GANs and VAEs are attempting to create synthetic output from latent space, VAEs are simpler and easier to train, and GANs are able to generate more realistic signals (sharper images)
  * GANs focus on how to get a model that approximate the input distribution
  * VAEs focus on modeling the input distribution from a decodable continuous latent space
* Within VAEs:
  * They focus on the variational inferance of latent codes, and therefore it provides a framework for both learning and efficient bayesian inference with latent variable
  * Meanwhile, VAEs have an intrinsic mechanism to disentangle the latent vectors.
* VAE vs Autoencoder
  * Both VAEs and Autoencoders attempt to reconstruct the input data while learning the latent vector
  * In VAEs, a latent vector is sampled from a distribution. This is a "latent" distribution because this distribution outputs a compact (and hidden) representation of the inputs 
  * In Autoencoder, the latent vector is an internal (hidden) layer that describes a code used to represent the input
  * Different from Autoencoders, the latent space of VAEs is continuous, and the decoder is used as a generative model
### VAE Design Principles
* The goal of VAEs is to find a tractable distribution that can closely estimate the conditional distribution of the latent attributes `z`, given input `x`.
  * To make the distribution tractable, VAEs introduced "Variational Inference Model" (an encoder) `Q(z|x)`, a modle that's often chosen to be a multivariate gaussian, whose mean and std are computed in the encoder network using the input
    * The elements of `z` (latent attributes) are independent
  * Since the inference model is an estimate, we use `KL (Kullbeck-Leibler) divergence` to determine the distance between the inference model `Q(z|x)` and the conditional distribution `P(z|x)`, in the encoder
  * In the decoder, the reconstruction loss:
    * If the image distribution is assumed to be Gaussian, `MSE` is used
    * If every pixel is considered to be Bernoulli distribution, then `binary cross entropy` is used
  * `VAE loss = Reconstruction Loss + KL Loss`
  * Reparameterization Trick
    * Decoder takes samples from the latent vector `z` to reconstruct the input. But the gradients cannot pass through the stochastic sampling layer during backpropagation. To solve the problem, reparametrrization trick is brought in to move the sampling block outside of the encoder-decoder network and make it look like the sampling came from the latent space
    * And now the network can be trained with familiar optimization methods such as Adam, RMSProp, SGD, etc.
<p align="center">
<img src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/images/VAE_reparameterization_trick.PNG" width="650" height="400" />
</p>

* Afrer training VAE model, we can discard the inference model as well as the addition and multiplication operators. To generate new meaningful outputs, samples are taken from the distribution block (often normal distribution) which generates `ε`
<p align="center">
<img src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/images/VAE_decoder_test.PNG" width="300" height="300" />
</p>

* Implementations
  * [VAE MLP][81], [VAE CNN][82] 
    * VAE CNN has a significant improvement in perceptive quality and a great reduction in the number of params
    * The `lambda()` function implements the reparameterization trick to push the sampling block out of VAE network
    * VAEs are intrinsically disentangle the latent vector dimensions to a certain extent, therefore when you set `latent_dim=2`, you will see the output images might be affected by 2 attributes
### Conditional VAE (CVAE)
* It imposes the condition (such as a one-hot label) on both encoder and decoder inputs, so that we can control which class to generate
* Comparing with VAE, the loss function is similar but added the given condition
  * KL Loss: meansures the distance between the encoder given the latent vector and the condition `Q(z|x, c)`, and the priori distribution given the condition `P(z|c)`
  * Reconstruction Loss: measures the loss of the decoder, given both latent vector and condition
* [CVAE implementation][83], comparing with VAE CNN:
  * The encoder input is the concatenation of the added condition and the original image input  
  * The decoder input is the concatenation of the added codition and the latent space sampling
<p align="center">
<img src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/images/VAE_encoder.PNG" width="500" height="600" />
 <img src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/images/VAE_decoder.PNG" width="500" height="600" />
</p>

### β-VAE
* It's the VAE with disentangled representations 
  * Although VAEs are intrinsically disentangle the latent vector dimensions to a certain extent
  * Disentangle representation has each single latent units sensitive to the changes in a single generative factor, but stay inrelevant to changes in other factors
* VAE, CVAE, β-VAE
  * β-VAE only changes the loss funcion of VAE, by adding `β` to KL loss
    * β > 1, it serves as a regularizer, which forces the latent codes disentangle further
      * The implicit effect of β is to set tighter standard deviation, which forces the latent codes in the posterior distribution `Q(z|x)` to be independent
    * When β=1, it's CVAE, that's why CVAE is a special case of β-VAE
    * To determine the value of β needs some trial and error, there must be a careful balance between reconstruction error and regularization of latent code independence
* [β-VAE, CVAE implementation][83]
  * IN this example, the disentanglement maximized at around β=9. Cuz when β>9, there is only 1 latent code got learned and the output is only affected by only 1 latent code

## Object Detection
* Bounding Box: It's used to localize an object in the image. Often uses upper left corner pixel and lower right corner pixel coordinates to describe a bounding box.
  * The coordinates system has the origin (0,0) at the upper left corner pixel of the entire image. 
* Object Detection needs to identify a bounding box belongs to a known object or background. It predicts 2 things:
  * The category (class) of an object
  * Bounding box pixel coordinates `y_box = ((x_min, y_min), (x_max, y_max))`
* Anchor Box: An image is divided into regions, each region is an anchor box.
  * The network will estimate the offsets with respect to each anchor box, in order to make the prediction closer to the ground truth
    * Offsets: They are the pixel error values that the network is trying to minimize between the ground truth bounding box and the predicted bounding box coordinates. It's the distance between the upper left corner of the anchor box and the upper left corner of the bounding box; it's also the distance between the lower right corner of the anchor box and the lower right corner of the bounding box
<p align="center">
<img src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/images/anchor_box.PNG" width="500" height="600" />
</p>

* Multi-scale object detection: the use of different scales of anchor boxes to detect object
  * Scaling Factors
    * Example: `s = [(1/40, 1/30), (1/20, 1/15), (1/10, 1/8), (1/5, 1/4), (1/3, 1/2), (1/2, 1)]`, this list contains all the scaling factors
    * `(1/40, 1/30)` means to divide the image's width into 40 pieces and divide its height into 30 pieces  
    * In each scaling factor, pixels covered by the smallest anchor box is known as the "Receptive Field"
* The offsets might be be reduced if we allow an anchor box to have different aspect ratios (different dimensions with the same centroid) 
  * The resized anchor box has the same centroid as the original anchor box
  * For each aspect ratio, the dimensions of the anchor box is
    * `(w_i, h_i) = (w*sx_j*sqrt(a_i), h*sy_j/sqrt(a_i))`
      * `a_i` is the aspect ratio of the anchor box
      * `(w, h)` is the original image's dimension
      * `sx_j`, `sy_j` is the jth scaling factor
  * For `aspect_ratio=1`, SSD recommends an additional anchor box
    * Its dimension is `(w_i, h_i) = (w*sqrt(sx_j * sx_j+1), h*sqrt(sy_j * sy_j+1))` 
  * The image below is showing aspect ratios as `[2, 1, 1/2]`
<p align="center">
<img src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/images/aspect_ratios.PNG" width="400" height="300" />
</p>

* IoU (Intersection over Union) 
  * It's using Jaccard Index, `IoU = (A and B) / (A union B)`, A is an anchor box, B is the ground truth bounding box
<p align="center">
<img src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/images/IoU.PNG" width="500" height="400" />
</p>

* Ground Truth Anchor Box (positive anchor box): it's the anchor box that has the largest IoU with the ground truth bounding box
* Extra Positive Anchor Box: besides the ground truth anchor box, if the IoU of an anchor box is above a certain threshold, this anchor box will also be added into the positive anchor box list
  * The remained anchor boxes are the negative anchor boxes and they do not contribute to the offset loss function
* Multi-scale object detection
  * During the process of creating anchor boxes of various dimensions, an optimal anchor box size that nearest to the ground truth bounding box will emerge
  * Multi-scale object detection uses multi-scale anchor boxes to effectively detect objects of different sizes
    
### SSD (Single-Shot Detection)
* It's a supervised object detection method
#### Loss Functions
* `L = L_cls + L_off`, predict both object category and the offsets of each anchor box
  * L_cls is the loss for object class prediction
    * By default, it's categorical cross-entropy loss 
    * We can replace categorical cross-entropy with Focal Loss to deal with class imbalance issue
      * The majority of anchor boxes are classified as negative anchor boxes (including background) while the anchor boxes that represent the target object is the minority, this leads to the class imbalance issue. Categorical cross entropy can be overpowered by the contribution of negative anchor boxes.
    * `L_cls_categorical_cross_entropy = -sum(y_true_i * log(y_pred_i))`
    * `L_cls_focal_loss = -α * sum(y_true_i * log(y_pred_i) * pow(1 - y_pred_i, γ))`
      * `pow(1 - y_pred_i, γ)` helps reduce the contribution of negative anchor boxes, since negative anchor box has label as 1 while positive anchor box has label as 0. For negative anchor boxes, `1 - y_pred_i` is close to 0 which will increase the loss of negative anchor box without sacrificing the contributions from the positive anchor box
      * This was inspired by RetinaNet, which works best with `γ=2, α=0.25` 
  * L_off is the offsets loss 
    * By default, it's using L1 loss (mean absolute error), L2 loss (mean square error) 
    * SSD can also use Smooth L1, which is more robust than L1 and less sensitive to outliers
      * `L_off = L1_smooth = pow(std * u, 2)/2 if |u| < 1/pow(std, 2) else |u| - 1/(2*pow(std, 2))`
        * `u = y_true - y_pred` 
        * In SSD, `std=1`, therefore L1_smooth is the same as [Huber Loss][85], which is less likely to be dominated by outliers
        * When `std --> inf`, `L1_smooth = L1`
      * Smooth L1 is quadratic for small values of `u`, and linear for large values, you can think it's a combo of L1, L2
* Labels
  * y_label vs y_cls are the labels of the object class and the predicted class
  * `y_gt = (x_gmin, x_gmax, y_gmin, y_gmax)` is the ground truth offsets, `y_off = ((x_omin, y_omin),(x_omax, y_omax))` is the predicted offsets in the form of pixel coordinates
    * `y_gt = (x_bmin - x_amin, x_bmax - x_bmax, y_bmin - y_amin, y_bmax - y_bmax)` 
  * <b>However, SSD doesn't recommend to predict raw pixel error values</b>, because raw pixel values tend to have high variance. Therefore, <b>here comes normalized offset values</b>:
    * `y_bounding_box = ((x_bmin, y_bmin), (x_bmax, y_bmax))` to centroid dimension format is `(c_bx, c_by, w_b, h_b)`
      * `(c_bx, c_by) = (x_min + (x_max - x_min)/2, y_min + (y_max - y_min)/2)`  
      * `(w_b, h_b) = (x_max - x_min, y_max - y_min)`
    * `y_anchor_box = ((x_amin, y_amin), (x_amax, y_amax))` to centroid dimension format is `(c_ax, c_ay, w_a, h_a)`
    * `y_gt_normalized = ((c_bx - c_ax)/w_a, (c_by - c_ay)/h_a, log(w_b/w_a), log(h_b/h_a))`, but when y_gt_normalized are small, `||y_gt_normalized||<<1`, small gradients can make it more difficult for the network converge. To alleviate the problem, each element is divided by its estimated standard deviation:
      * `y_gt_normalized = ((c_bx - c_ax)/w_a/std_x, (c_by - c_ay)/h_a/std_y, log(w_b/w_a)/std_w, log(h_b/h_a)/std_h)`
        * Recommend to use `std_x = std_y = 0.1`, meaning the expected range of pixel error along x, y axes is (-10%, +10%)  
        * Recommend to use `std_w = std_h = 0.2`, meaning the expected range of width and height is (-20%, +20%) 
#### SSD Architecture
* Backbone Network: It extracts features of the downstream of classification and offset prediction
  * It can be a pre-trained model with frozen weights 
  * If a backbone implements `k` rounds of downsampling, the image size is `m*n`, and there are `p` aspect ratios, then the total number of anchor boxes in the first set is `(m/(k*k)) * (n/(k*k)) * (1+p)`
* SSD Head: It performs the object detection task after getting the features from the backbone
* As we can see here, after the backbone, it starts with lower scaling factor
<p align="center">
<img src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/images/SSD_head_layers_order.PNG" width="600" height="500" />
</p>

##### [SSD Implementation Code][87]
* SSD Object
<p align="center">
<img src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/images/SSD_object.PNG" width="500" height="400" />
</p>

* Multi-thread data generator is used since the images are in high resolution
* NMS (Non-Maximum Suppression)
  * After model training, when the model predicts the bounding box offsets, there can be 2+ bounding box refer to the same object, causing redundant predictions. NMS is used to remove redundant predictions.
  * Among all the deplicated bounding boxes, the one with maximum confidence score (or probability) is used as reference `b_m`
    * For the remaining boxes, if the IoU of a bounding box with `b_m` >= threshold `N_t`, the bounding box is removed. The process repeated until there is no remaining box.
      * The problem here is, a bounding box contains another object but might be removed too
    * SoftNMS proposes that, instead of removing the box from the list, the score of the overlapping bounding box is decreased at a negative exponential rate in proportion to the square of its IoU with `b_m`. By doing this, the bounding box with smaller IoU has a higher chance to survive for later iterations
      * SoftNMS appears to have higher average precision than classic NMS 
* SSD Model Validation
  * Mean IoU (mIoU) between ground truth bounding boxes and predicted bounding boxes
    * Suggest to make comparisons between the same object class
    * In this code, the comparisons are NOT between the same object class...
  * Precision, Recall between the ground truth class and the predicted class
    * Precision measures how good SSD is at correctly identifying an object in the image
    * Recall measures how good SSD is at not misclassifying an object in the image
  * In object detection, the precision and recall curves over different mIoUs are used to measure the performance

## Semantic Segmentation
* The goal of semantic segmentation is to classify each pixel according to its object class.
  * All pixels of the same object have the same color and they all belong to the same class.
* Segmentation algorithms partition an image into different regions (set of pixels) in order to better understand what does the image represent
* "Thing": countable object (such as vehicle, traffic sign, etc.) in an image
* "Stuff": uncountable object (such as sky, water, grass, etc.) or collectively countable objects
* Instance Segmentation: identify thing
* Semantic Segmentation: identify stuff
  * Sometimes a collection of countable things are consider a stuff as a whole 
* Panoptic Segmentation: identify both thing and stuff

### FCN (Fully Convolutional Networks) 
* The architecture of FCN
  * The backbone network (ResNetv2) serves as the feature extractor
  * Parallel classifiers running simultaneously
    * The number of classifiers is the number of stuff need to identify
    * Each classifier has a one-hot vector with the domension of image_width * image_height (the number of pixels in the image) 
  * In this example, `1/4, 1/8, 1/16, 1/32` are the feature map dimensions
    * The 1st set of feature maps came from the backbone
    * The additional sets of feature maps are generated by using successive `Conv2D(strides=2)-BN-ReLU` layers
  * PSPNet (Pyramid Scene Network) adds further improvement to the network by further processing each feature map from the `Conv2D(strides=2)-BN-ReLU` layer by another convolutional layer
  * Upsampling is used to make sure all the features pyramid output has the same size as the first set of feature map before concatenation
  * [The code][89]
    * I didn't find the PSPNet part in this code... 
<p align="center">
<img src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/images/FCN.PNG" width="700" height="500" />
</p>

* Validation
  * `mIoU`: IoU is computed between the ground truth segmentation mask and the predicted segmentation mask for each stuff category
    * Including the background
  * `Average Pixel Accuracy`
    * The number of predictions equals to the number of pixels in the image
    * For each test image, an average pixel accuracy is calculated, then the mean of all the test images is computed
* Further Exploration Tips
  * Image Augumentation is always an option
  * The number of filters in the features' pyramid can be reduced to reduce the # of params
  * Increasing the number of levels in the features' pyramid can also be explored

## Unsupervised Learning using Mutual Information (MI)
* One of a successful unsupervised learning in deep learning is to maximize the MI betwrrn 2 random variables
  * MI helps cluster latent vectors
  * If we can cluster all the training data's latent vectors, then a linear seperation algorithm or linear classifier can be used to classify all the testing data's latent vectors
    * Similar data will have their latent vectors be clustered together
    * Regions far apart can be seperated by a linear seperation algorithm or linear classifier
* Mutual Information (MI) is a measure of dependency between 2 random variables M, N
  * It's also known as "Information Gain" or the reduction of uncertainty of M upon observing N
  * In contrast with correlation, MI can measure non-linear statistical dependence between M and N
  * MI `I(M;N) = D_KL(P(M, N) || P(M)P(N))` is the KL divergence between joint distribution and the product of marginal distribution
    * P(M, N) is the joint distribution
    * P(M), P(N) are the marginal distribution
  * MI is a measure of how far M, N are independent from each other. Higher MI, higher dependency. When MI=0, M, N are independent
  * MI vs Entropy
    * `I(M;N) = H(M) + H(N) - H(M, N)`
      * `H(M)` is the entropy (the measure of undertainty) of variable M
      * MI increases with marginal entropy but decreases with joint entropy
     * `I(M;N) = H(P(M)) - H(P(M|N)) = H(M) - H(M|N) = H(P(N)) - H(P(N|M)) = H(N) - H(N|M)`
       * ML is how much uncertainty decreases in one variable while given the other variable
       * So when we are more certain about one variable given the other variable, we can maximize MI
     * `I(M;N) = H(M,N) - H(N|M) - H(M|N)`
     * `I(M;N) = I(N;M)`, MI is symmetric 
### Unsupervised Learning by Maximizing MI
* "Discrete" below means discrete joint and marginal distributions of latent vectors Z, Z_bar; "Continuous" means continuous joint and marginal distributions
#### Model IIC (Invariant Information Clustering) for Discrete Random Variables
##### Loss function 
  * `L(Z, Z_bar) = -I(Z;Z_bar) = P(Z, Z_bar) * (log(P(Z)) + log(P(Z_bar)) - log(P(Z, Z_bar)))` 
    * Minimize the loss is to minimize the negative MI (maximize MI) 
    * X is the input image 
    * `X_bar` is transformed image
      * The transformation can be small rotation, random cropping, brightness adjustment, etc., as long as the meaning of X stays the same
    * `Z` and `Z_bar` are the encoded X and X_bar, namely, the latent vectors of X and X_bar
      * The foundation of the whole unsupervised labling idea is base on that, X and X_bar share the same info as their latent vectors 
    * ICC assumes Z and Z_bar are independent such that the joint distribution can be estimated as `P(Z, Z_bar) = P(Z) * transpose(P(Z_bar))`
      * `P(Z, Z_bar)` is an N*N matrix where each element Z_ij corresponds to the probability of simultaneously observing 2 random variables (Z_i, Z_bar_j)
    * Since for each X sample, we calculate its latent vector, so the joint distribution is `P(Z, Z_bar) = sum(P(Z_i) * transpose(P(Z_bar_i))) / M`
      * `M` is the batch size; `i` indicates the ith batch
      * To further enforce symmetry, `P(Z, Z_bar) = (P(Z, Z_bar) + transpose(P(Z, Z_bar))) / 2`
    * The marginal distributions are:
      * `P(Z) = sum(P(Z_i, Z_bar_j))`, `j >= 1 and j<= N`, sum up ROW-WISE
      * `P(Z_bar) = sum(P(Z_i, Z_bar_j))`, `i >= 1 and i<= N`, sum up COLUMN-WISE
##### [IIC implementation][90]
* The encoder network for unsupervised clustering is formed by [a VGG backbone][91], and a `Dense()` layer with `softmax` output
* [Paired training input data][102] (Siamese input image) made of the input image X and transformed image X_bar
  * ‼ NOTE: In [IIC implementation code][90], `fit()` should be `fit_generator()`, since there is a data generator to generate the paired images 
* Overclustering is used to improve IIC performance, overclustering here is an encoder with 2 or more heads. The encoder network looks like below:
    * Each head contributes equally to the total loss, so the final negative MI is scaled by the number of heads 
    * All the heads may not get same level of performance though
<p align="center">
<img src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/images/IIC_multi_heads.PNG" width="450" height="250" />
</p>

* [Unsupervised Labeling use Hungarian algorithm to assign a label to a cluster with the min cost][104]
  * An example of how does Hungarian algorithm work
  * Matrix X is a binary matrix that each row assiged to only 1 column
  * To evaluate the clustering results, it's comparing matrix X output and the ground truth to calculate the accuracy
<p align="center">
<img src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/images/hungarian_alg.PNG" width="550" height="250" />
</p>

#### Model MINE (Mutual Information Network Estimator) for Continuous Random Variables
* The idea is similar to IIC for discrere random variables, but it's an approximation here with the input follows a certain distributions
  * The approximation is done by taking huge amount of samples and creating a histogram with large number of bins 
* Because MINE is an approximation, it's not expected to perform better than IIC
##### [MINE implementation][92]
* Assume the X and X_bar forms a binary Gussian distribution
* Similar to IIC:
  * The input is images and their corresponding transformed images, the goal of MINE clustering is to maximize the MI between the latent vectors of the image pairs
* Different from IIC:
  * The output of MINE clustering is in continuous format instead of one-hot vector format, therefore we need to use linear classifier instead of linear assignment algorithm
    * A linear classifier is a MLP without a non-linear activation function
    * ‼ In the code, it's making 2 mistakes:
      * The activation function in class MINE's `build_model()` should be `softmax`
      * The activation function in class LinearClassifier's `build_model()` should be `linear`
   
## [Deep Reinforcement Learning][84]

## Meta Learning
* Different from traditional supervised learning where the model learned the ground truth from the training labels, meta learning doesn't provide the ground truth but let the model to learn how to learn
### [Few-Shot Learning][62]
* It's a type of meta learning. The model is provided with a "support set" where doesn't have the labels but contains the multiples samples of the ground truth. The model will compare the similarities between the "query" and the supporting set, to find the most similar one and predict that as the right one
* "K-ways": means k classes. More classes, more challenging to improve the accuracy
* "N-shot": means n samples in a class. Higher n, easier to improve the accuracy

## Well Known Datasets
* [Keras packaged datasets][3]
* [Cats and Dogs image set][15]

## Reference
* [Deep Learning Python Notebooks][1]
* [Deep Learning with Python][2]
* [Advanced Deep Learning with Keras and Tensorflow2][45]

## Interview Questions
I just found some companies like to ask you to implement methods used in deep learning without using numpy, sklearn, no matter whether they need deep learning... This is why I put these here, even though there is no time to check these during the interviews. Personally I don't like this type of interview at all, they gave impossible timeline (unless you have remembered each answer), no one talk with you to help you understand what do the questions really mean, and the solution used here has a big difference with the practical solutions. This type of interviews could be used at most for entry level data scientists but shuld not be used for senior level. Unfortunately, these companies do not have a qualified hiring manager. Sometimes, this world is funny and ridicuclous in a way I can never understand.
* [Implement one-hot encoding][26]
  * This one uses numpy to create the matrix, but you can do it with simple python lists
  * I don't like its second method because of the hashing collision. Instead, I think there are some preprocessing can be done before the encoding here, such as tokenize all the words, removing stop words and other less important tokens (such as those got low tf-idf score)
* [Implement k-fold CV][5]
* [Implements machine learning algorithms from scratch][42]

[1]:https://github.com/fchollet/deep-learning-with-python-notebooks
[2]:https://www.manning.com/books/deep-learning-with-python?a_aid=keras&a_bid=76564dff
[3]:https://keras.io/api/datasets/
[4]:https://keras.io/api/metrics/
[5]:https://github.com/hanhanwu/deep-learning-with-python-notebooks-on-polyaxon/blob/master/3.7-predicting-house-prices.ipynb
[6]:https://www.machinecurve.com/index.php/2020/02/18/how-to-use-k-fold-cross-validation-with-keras/
[7]:https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
[8]:https://github.com/hanhanwu/deep-learning-with-python-notebooks-on-polyaxon/blob/master/3.6-classifying-newswires.ipynb
[9]:https://github.com/hanhanwu/deep-learning-with-python-notebooks-on-polyaxon/blob/master/4.4-overfitting-and-underfitting.ipynb#Adding-weight-regularization
[10]:https://github.com/hanhanwu/deep-learning-with-python-notebooks-on-polyaxon/blob/master/4.4-overfitting-and-underfitting.ipynb#Adding-dropout
[11]:https://keras.io/api/layers/convolution_layers/
[12]:https://keras.io/api/layers/convolution_layers/convolution2d/
[13]:https://stackoverflow.com/questions/19062875/how-to-get-the-number-of-channels-from-an-image-in-opencv-2
[14]:https://www.quora.com/What-does-stride-mean-in-the-context-of-convolutional-neural-networks
[15]:https://www.kaggle.com/c/dogs-vs-cats/data
[16]:https://keras.io/api/layers/pooling_layers/max_pooling2d/
[17]:https://github.com/hanhanwu/deep-learning-with-python-notebooks-on-polyaxon/blob/master/5.2-using-convnets-with-small-datasets.ipynb#Data-preprocessing
[18]:https://github.com/hanhanwu/deep-learning-with-python-notebooks-on-polyaxon/blob/master/5.2-using-convnets-with-small-datasets.ipynb#Using-data-augmentation
[19]:https://github.com/keras-team/keras-applications
[20]:https://keras.io/api/applications/
[21]:https://github.com/hanhanwu/deep-learning-with-python-notebooks-on-polyaxon/blob/master/5.3-using-a-pretrained-convnet.ipynb
[22]:https://github.com/hanhanwu/deep-learning-with-python-notebooks-on-polyaxon/blob/master/5.4-visualizing-what-convnets-learn.ipynb#Visualizing-intermediate-activations
[23]:https://github.com/hanhanwu/deep-learning-with-python-notebooks-on-polyaxon/blob/master/5.4-visualizing-what-convnets-learn.ipynb
[24]:https://www.quora.com/What-does-the-terms-Top-1-and-Top-5-mean-in-the-context-of-Machine-Learning-research-papers-when-report-empirical-results#:~:text=The%20Top%2D1%20error%20is,among%20its%20top%205%20guesses.
[25]:https://www.analyticsvidhya.com/blog/2020/08/top-4-pre-trained-models-for-image-classification-with-python-code/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
[26]:https://github.com/hanhanwu/deep-learning-with-python-notebooks-on-polyaxon/blob/master/6.1-one-hot-encoding-of-words-or-characters.ipynb
[27]:https://nlp.stanford.edu/projects/glove/
[28]:https://github.com/hanhanwu/deep-learning-with-python-notebooks-on-polyaxon/blob/master/6.1-using-word-embeddings.ipynb
[29]:https://www.analyticsvidhya.com/blog/2020/08/top-4-sentence-embedding-techniques-using-python/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
[30]:https://radimrehurek.com/gensim/models/doc2vec.html#gensim.models.doc2vec.Doc2Vec
[31]:https://github.com/UKPLab/sentence-transformers
[32]:https://docs.google.com/spreadsheets/d/14QplCdTCDwEmTqrn1LH4yrbKvdogK4oQvYO1K1aPR5M/edit#gid=0
[33]:https://github.com/facebookresearch/InferSent
[34]:https://github.com/tensorflow/tfjs-models/tree/master/universal-sentence-encoder
[35]:https://github.com/hanhanwu/deep-learning-with-python-notebooks-on-polyaxon/blob/master/6.3-advanced-usage-of-recurrent-neural-networks.ipynb
[36]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/sequencial_analysis/time_series_forecasting.ipynb
[37]:https://www.analyticsvidhya.com/blog/2020/08/a-beginners-guide-to-focal-loss-in-object-detection/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
[38]:https://github.com/hanhanwu/deep-learning-with-python-notebooks-on-polyaxon/blob/master/6.4-sequence-processing-with-convnets.ipynb
[39]:https://github.com/hanhanwu/deep-learning-with-python-notebooks-on-polyaxon/blob/master/8.5-introduction-to-gans.ipynb#A-bag-of-tricks
[40]:https://www.analyticsvidhya.com/blog/2020/09/overfitting-in-cnn-show-to-treat-overfitting-in-convolutional-neural-networks/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
[41]:https://www.tensorflow.org/api_docs/python/tf/keras/initializers
[42]:https://github.com/hanhanwu/ML-From-Scratch
[43]:https://github.com/tensorflow/federated/blob/master/docs/tutorials/custom_federated_algorithms_2.ipynb
[44]:https://www.analyticsvidhya.com/blog/2021/03/grayscale-and-rgb-format-for-storing-images/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
[45]:https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras
[46]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/sequencial_analysis/learning_notes.md
[47]:https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter1-keras-quick-tour/mlp-mnist-1.3.2.py
[48]:https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter1-keras-quick-tour/cnn-model-1.3.2.py
[49]:https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter1-keras-quick-tour/rnn-mnist-1.5.1.py
[50]:https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#activation-functions
[51]:https://play.google.com/books/reader?id=68rTDwAAQBAJ&hl=en_CA&pg=GBS.PA14.w.1.0.83
[52]:https://play.google.com/books/reader?id=68rTDwAAQBAJ&hl=en_CA&pg=GBS.PA30.w.3.0.53
[53]:https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter1-keras-quick-tour/rnn-mnist-1.5.1.py
[54]:https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter2-deep-networks/cnn-y-network-2.1.2.py
[55]:https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d#:~:text=Dilated%20convolutions%20introduce%20another%20parameter,while%20only%20using%209%20parameters.
[56]:https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter2-deep-networks/resnet-cifar10-2.2.1.py
[57]:https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter2-deep-networks/densenet-cifar10-2.4.1.py
[58]:https://github.com/hanhanwu/Hanhan_COLAB_Experiemnts/blob/master/autoencoder_latent_vector_plot.ipynb
[59]:https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter3-autoencoders/denoising-autoencoder-mnist-3.3.1.py
[60]:https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter3-autoencoders/colorization-autoencoder-cifar10-3.4.1.py
[61]:https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter3-autoencoders/classifier-autoencoder-mnist-3.3.1.py
[62]:https://www.analyticsvidhya.com/blog/2021/05/an-introduction-to-few-shot-learning/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
[63]:https://play.google.com/books/reader?id=68rTDwAAQBAJ&hl=en_CA&pg=GBS.PA111.w.5.0.3
[64]:https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter4-gan/dcgan-mnist-4.2.1.py
[65]:https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter4-gan/cgan-mnist-4.3.1.py
[66]:https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter5-improved-gan/wgan-mnist-5.1.2.py
[67]:https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/lib/gan.py
[68]:https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter5-improved-gan/lsgan-mnist-5.2.1.py
[69]:https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter5-improved-gan/acgan-mnist-5.3.1.py
[70]:https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter6-disentangled-gan/infogan-mnist-6.1.1.py
[71]:https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter6-disentangled-gan/stackedgan-mnist-6.2.1.py
[72]:https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter7-cross-domain-gan/cyclegan-7.1.1.py
[73]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/images/U_Net.PNG
[74]:https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter7-cross-domain-gan/other_utils.py
[75]:https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter7-cross-domain-gan/cifar10_utils.py
[76]:https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter7-cross-domain-gan/mnist_svhn_utils.py
[77]:https://www.analyticsvidhya.com/blog/2021/05/image-processing-using-numpy-with-practical-implementation-and-code/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
[78]:https://www.analyticsvidhya.com/blog/2021/05/progressive-growing-gan-progan/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
[79]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/images/infoGAN.PNG
[80]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/images/VAE_reparameterization_trick.PNG
[81]:https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter8-vae/vae-mlp-mnist-8.1.1.py
[82]:https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter8-vae/vae-cnn-mnist-8.1.2.py
[83]:https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter8-vae/cvae-cnn-mnist-8.2.1.py
[84]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/deep_reinforcement_learning.md
[85]:https://en.wikipedia.org/wiki/Huber_loss
[86]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/images/ssd_imp_stru.PNG
[87]:https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/tree/master/chapter11-detection
[88]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/images/FCN.PNG
[89]:https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/tree/master/chapter12-segmentation
[90]:https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter13-mi-unsupervised/iic-13.5.1.py
[91]:https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter13-mi-unsupervised/vgg.py
[92]:https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter13-mi-unsupervised/mine-13.8.1.py
[93]:https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter11-detection/data_generator.py
[94]:https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter1-keras-quick-tour/cnn-model-1.3.2.py
[95]:https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter1-keras-quick-tour/cnn-mnist-1.4.1.py
[96]:https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter2-deep-networks/cnn-functional-2.1.1.py
[97]:https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter2-deep-networks/resnet-cifar10-2.2.1.py#L392
[98]:https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter2-deep-networks/resnet-cifar10-2.2.1.py#L98
[99]:https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter2-deep-networks/resnet-cifar10-2.2.1.py#L377
[100]:https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter2-deep-networks/resnet-cifar10-2.2.1.py#L396
[101]:https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter3-autoencoders/autoencoder-mnist-3.2.1.py
[102]:https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter13-mi-unsupervised/data_generator.py
[103]:https://www.pyimagesearch.com/2018/12/24/how-to-use-keras-fit-and-fit_generator-a-hands-on-tutorial/
[104]:https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter13-mi-unsupervised/utils.py#L9
