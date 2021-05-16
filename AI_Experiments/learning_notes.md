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


## Data Preprocessing Methods
### Collected Preprocessing Methods
#### Used in Computer Vision
* [Convert color images to gray][74] and [reshape the images][75]
* [Reshape MNIST digits data to SVHN data format][76]
* [Image Processing using numpy][77]
  * It includes: Opening an Image, Details of an image, Saving ndarray as Image, Rotating an Image, Negative of an Image, Padding Black Spaces, Visualizing RGB Channels, Colour Reduction, Trim Image, Pasting With Slice, Binarize Image, Flip Image, An alternate way to Flip an Image, Blending Two Images, Masking Images, Histogram For Pixel Intensity 
  * Looks like the author really likes Emma Stone :D

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

### How to Preprocessing Testing Data ❣❣
* The parameters used for preprocessing the training data should be generated from the training data only, and be applied to the testing data too.
* For example, when you are using `mean` and `std` to normalize the data, these params should be generated from the training data, and be applied to the testing data.
  * It seems that `(value - mean)/std` is a common way used to normalize features when they are in different scales. This method doesn't guarantee they are on the same scale, but could make them on similar scales

### Multi-class lables (Python)
* Mehtod 1 - One-hot encoding
  * The labels have N columns, each column represents a class
  * Mark the labeled class as 1
* Method 2 - Integer labels
  * Each class is represented as an integer
  * 1 numpy array is enough for the labels
* [Example for both methods][8]

### Preprocess Text data
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
  * "Batch size" is the number of samples in each training step
    * Because you can't pass the entire dataset into NN all at once, need multiple batches
    * `training_batchs = total training sample size / training batch size`
    * `validation_batchs = total validation sample size / validation batch size`
  * 1 "Epoch" will process the entire dataset to update weights
  * "Number of iterations (or number of steps) = total dataset size/batch size", it's the number of steps needed to complete 1 epoch
  * Seperate data and labels
    * Labels are created based on directories, different classes of images are put in different directories
* [Image batch preprocess for federated learning][43]
  * For each 28*28 image, it flatten to 784 one dimensional matrix, then divided by 255 to convert the values into [0,1] range because the pixel values are between 0 and 255
  * The preprocessed image can also be reshaped baçk to 28*28
#### Convert an image to image tensor
* [Example][22]
* The image tensor can be used to understand the output of each activation layer
  

## Layers & Dimensions 😱😱😱
### Multiple Inputs
* We often see one input in NN, but in fact it allows multiple inputs. 
  * For example, such as `Concatenate()` which concatenate multiple inputs of the same shape at the concatenation axis to form 1 tensor. [Check the code example here][54]
    * The 2 branches in this code are using different "dilation rate". The Dilation rate decides the kernel's receptive field's size. Comparing with `dilation_rate=1`, larger rate will fill more 0 around the kernel of dilation_rate as 1, which is a computationally effecient method to increase the kernel's receptive field's size. [See example here][55].
    * Meanwhile, using different receptive field sizes for kernels here allows each branch to generate different feature maps.
  * Besides concatenation, we can also do other operations to put multiple inputs together, such as `add`, `dot`, `multiple`, etc.
* Note! The input here doesn't have to bethe first layer of NN. Each input can be a sequence of layers, and finally all these inputs merged at a certain layer.
  * It's like an NN has multiple branches, and each branch do different types of work with the same original input data 
* Multiple inputs also has a cost in model complexity and the increasing in parameters

### Hidden Layers
* Having more units in a layer, also means having higher dimensional space representaton, will allow you to learn more complex representation.
  * Imagine this allows you to cut an image into smaller pieces for learning.
  * The drawback is more computationally expensive, and you might be learning unnecessary patterns which could even lead to overfitting.
* When there are N classes to predict, the dimensions of all the hidden layers better > N, otherwise the information loss will lead to overfitting
  
### Activation Functions
* In deep learning, layers like `Dense` only does linear operation, and a sequence of `Dense` only approximate a linear function. 
* Inserting the activation function enables the nonlinear mappings.
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
* `rmsprop` is often good enough, `adam` is better
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
  * We often saw "top-1 error", "top-5" error appeared in papers, here's more explaination

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
  * `sparse_categorical_` is similar to the use case of `categorical_`, but it's used when the labels are in integer format
* `focal loss` vs `cross entropy` vs `balanced cross entropy`
  * Focal loss works better in dealing with the data imbalance issue in object detection of one stage detection
    * Data imbalance in object detection is often caused by much larger amount of background object and only a few objects (foreground) locations
  * Cross-Entropy loss is to penalize the wrong predictions more than to reward the right predictions. But because the final loss is summed over small losses from the entire image, when the data is imbalanced, the loss value won't be accurate to reflect the detected objects.
  * Balanced Cross-Entropy Loss is to add weights to both positive and negative classes. It does help differentiate the positive & negative, but cannot distinguish easy & hard examples
  * Focal loss down-weights easy examples and focus training on hard negatives.
    * After a lot of trials and experiments, researchers have found `∝=0.25 & γ=2` to work best
  * [reference][37]
* Some loss functions's formula
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
  * Training data only, no need to be used in the last layer
  * It helps reduce overfirtting and makes NN more robust to unseen data input
  * In Keras, bias, weights and activation functions can be regularized in each layer
  * L1 and L2
    * L1 regularization, where the cost added is proportional to the absolute value of the weights coefficients
    * L2 regularization, where the cost added is proportional to the square of the value of the weights coefficients
    * Therefore, both L1 and L2 favor smaller param values. However, NN with small params are more insensitive to the noise in the input datasof
  * [Dropout][10]
    * It's one of the most effective and most commonly used regularization method
    * It randomly drops (i.e. setting to zero) a number of output features of a layer during training
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
  * [See examples here][40]
  * [Tensorflow initializer][41]
  
## Convolutional Networks (Convnet)
* Convnets can learn <b>local, translation-invariant features</b>, so they are very data-efficient on perceptual problems. Therefore, even when the dataset is small (such as hundreds of images), you might still get a reasonable results.
* In CNN, a <b>kernel (filter)</b> can be visualized as a window that slides through the whole image from left to right, from top to bottom. 
  * This operation is called as "Convolution", which transforms the input image as a feature map
* [Keras Convnet][11]
* [Conv2D][12]
  * "Input shape" is `(batch_size, image_height, image_width, image_channels)`, batch_size is optional
    * The width and height tend to shrink when we go deeper in convnet
    * However, in Keras `reshape()`, the order is `(sample_size, channels, width, height)`...
  * When there are multiple layers of Conv2D in a neural net, deeper layer gets larger number of batches, notmrally we choose batch size in a sequence of 32, 64, 128, ...
  * `image_channels` is also the image depth. [How to use opencv2 to find image channels][13]
    * If returns 2 dimensions, the image channel is 1, otherwise it's the third number returned in the output
  * `padding=same` will pad the borders with 0 to keep the original image size
  * `kernel` can be non-square, such as `kernel=(3,5)`
* [MaxPooling2D][16]
  * Pooling Layer - pooling is a down-sampling operation that reduces the feature map size
  * It performs down-sampling operations to reduce the dimensionality and creates a pooled feature map by sliding a filter matrix over the input matrix. See [example here][52], every patch of size pool_size * pool_size is reduced to 1 feature map.
  * `MaxPooling2D` chooses the max value from each patch, `AveragePooling2D` chooses the average value from each patch
  * `pool_size` can be non-square, such as `pool_size=(1,2)`
* [What is `strides`][14]
  * The size (height, width) of the moving window
* Why `Flatten()`
  * `Dense` layer is 1D, so if the output from Conv2D is more than 1D, it needs to be flatterned into 1D before entering into the Dense layer.
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

## ResNet
* ResNet introduces residual learning, which allows to build a very deep network while addresing the vanishing gradient problem.
  * Gradient Vanishing: Backpropagation follows the chain rule, there is a tendency for the gradient to diminish as it reaches to the shallow layers, due to the multiplication of small numbers (small loss functions and parameter values). However, if the gradients decrease and the parameters cannot update appropriately, then the network will fail to improve its performance.
  * ResNet allows info flows through shortcuts to the shallow layers, in order to relieve the gradient vanishing problem
* [The implementation of ResNet][56]
  * A transition layer is used when joining 2 residual blocks in different sizes
  * `kernel_initializer='he_normal'` to help the convergence when back propagation is taking the place
  * ResNet is easier to converge with Adam
    * In Adam, `lr_reducer()` is to reduce the learning rate by a certain factor if the validation rate has not been improved after `patience=5` epochs
  * In Keras, we can use `load_model()` to load the saved model from `checkpoint`
  * It has v1 and v2. v2 improves the performance by making some chanages in the layers arrangement in residual block design. It moves Conv2D layer ahead of BN-ReLU layers in each residual block, and the kernel sizes of Conv2D are a bit different.
<p align="center">
<img src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/images/resnet_v1_v2_diff.PNG" width="400" height="250" />
</p>

## DenseNet
* DenseNet improves ResNet further by allowing the next layer to get access to all the previous feature maps (Dense), while keeping the number of params low in deep networks by using both "bottleneck" and "transition layer". [See DenseNet implementation here][57]
* The number of feature maps generated per layer is called the growth rate `k`, normally `k=12`
* With the Bottleneck layer, each Conv2D(3) only need to process `4k` feature maps instead of `(l-1) * k + 2*k` for layer `l`
* Transition layer is used to transform a feature map size to a smaller one between 2 `Dense` layers
  * The reduction rate is usually half
  * Within each Dense layerb, the feature map size remains constant
  * Using multiple Dense layers joined by transition layers is a solution to solve feature maps sizes mismatch 
  * When compression and dimensionality reduction being put together, the transition layer is BN-Conv2D(1)-AvergingPooling2D
* `RMSprop` is used as the optimizer for DenseNet, since it converges better

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
* Autoencoder is a NN architecture that attempts to find the compressed representation of the given input data. It can learn the code alone without human label and therefore is considered as "unsupervised learning". However, when you are using autoencoder in some applications (such as denoising, colorization, etc.), labels are still needed in the model training part.
### Main Architecture
* Input (`x`) --> Encoder (`z = f(x)`) --> Latent Vector (`z`) --> Decoder (`x_approx = g(z)`)
  * Latent Vector is a low-dimensional compressed representation of the input distribution
  * It's expected that the output recovered by the decoder can only approximate the input
  * The goal is to make `x_approx` as similar as `x`, so there's a loss function trying to minimize the dissimilarity between `x` and `x_approx`
    * If the decoder's output is assumed to be Gaussian, then the loss function boils down to `MSE`
    * The autoencoder can be trained to minimize the loss function through backpropagation. Keras does backpropagation automatically for you.
* Visualized Common Autoencoder's Structure
  * The example here was used for MNIST dataset (0 ~ 9 digits in grey scale) 
  * For more complex dataset, we can create deeper encoder & decoder, as well as using more epoches in training
<p align="center">
<img src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/images/encoder.PNG" width="600" height="350" />
<img src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/images/decoder.PNG" width="600" height="450" />
 <img src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/images/autoencoder.PNG" width="600" height="250" />
</p>

* Using MNIST data as an example, we can plot the latent vector as 2-dim plot, [check the code here][58]
  * Each digit is clustered in a region of the space in the latent vector plot
  * In the decoder plot, it should reflect the same regions for each digit
  * This type of plot can help gain more understanding of the NN

### Autoencoder Applications
* All need labels
* [Autoencoder denoising][59]
  * Removing noise from the images
* [Autoencoder classification][61]
  * Here provides a function to convert colored images to grey images
  *  The architecture here is more complex 🌺:
    * Added more blocks of convolution and transposed convolution (see `layer_filters` has more values)
    * Increased the number of filters at each CNN block (see each value in `layer_filters`)
    * Latent Vactor has increased  the dimension, in order to increase the number of salient properties it can present
    * Training increased epoches
      * Learning rate reducer is used to scale down the learning rate when validation loss is not improving 
* [Autoencoder colorization][60]
  * Colorize the images 

## Generative Adversarial Networks (GAN)
* GANs belong to the family of generative models. Different from autoencoder, generative models can create new and meaningful output given arbitary encodings
* GANs train 2 competing components
  * Discriminator: it learns to discriminate the real and the fake data 
  * Generator: it's trying to fool the discriminator by generating fake data that pretend to be real
  * When it reaches to the point that the Discriminator can no longer tell the difference between real and fake data, it will be discarded and the model will use Generator to create new "realistic" data
  * If the data input is images, both generator and discriminator will be using CNN; if the input is single-dimensional sequence (sucb as audio), both generator and discriminator will be using RNN/LSTM/GRU
* Comparing with other types of NN, GANs are notoriously hard to train, and prone to mode collapse
  * Mode Collapse: the generator collapses which produces limited varieties of samples. For example, 0 ~9 digits, it only provides a few modes such as 6,7,8. The main cause is, in aach iteration of generator over-optimizes for a particular discriminator, and the discriminator never manages to learn its way out of the trap. As a result the generators rotate through a small set of output types. 

### DCGAN vs CGAN
* [DCGAN Design Principles][63]
  * The use of BN (Batch Normalization) can help stablize learning by normalizing the input to each layer to have 0 mean and unit variance
  * [The Code][64]  
    * Opposite to CNNs, transposed CNNs can create an image given feature maps, this is also why it's used in autoencoder too 
    * Due to custom training, `train_on_batch()` is used instead of using `fit()`
    * train the discriminator --> train the generator in the adversarial model will be repeated in multiple train steps
    * When the training converges, the discriminator loss is around 0.5 while the generator loss is around 1
* CGAN
  * It's quite similar to DCGAN, except the additional conditions applied to generator, discriminator and the loss function
  * With the given condition, we can use CGAN to create a specified fake output 
  * [The Code][65] 
    * The condition here is the additional one hot vector which indicates which digit do we want to create the fake data for 
* Loss functions
  * The loss function for discriminator is to minimize the error when identifying both real and fake data (given the condition)
    * real data with label 1, fake data with label 0 
    * The discriminator is trained again both real and fake data, the fake data is generated by the generator
  * The loss function for generator is to maximize the chance of making discriminator believing the fake data is real (conditioned on the specified conditions)
    * When training the generator, it's in the adversarial step of GAN model when discriminator weightes are frozen but gradients propagate back to the generator
      * The Generator is only trained with fake data & label 1 (pretending to be real)
    * The reason why the loss function of generator is not simply opposite the discriminator's loss function:
      * The gradient updates are small and have diminished significantly when propagate to the generator layers, and will make the generator fail to converge
  * As we note that, 
    * During the discriminator training, there's no need to frozen generator's weights, since it's only used to create fake data
    * But during generator training, we need discriminator's participation, since the loss function needs its output
* Why easy to have model instability
  * If the discriminator learns faster than the generator, then the generator's params will fail to optimize
  * But if the discriminator learns slower, then the gradients may vanish before reaching the generator
    * Worst case, if the discriminator cannot converge, the generator cannot get useful feedback 
 
### Improved GANs
* WGAN (Wasserstein GAN) is trying to improve model stability and avoid mode collapse by replacing the loss function based on Earth Mover's Distance (EMD), also known as Wasserstein 1, which is a smooth disfferentiable function even when there is no overlap between the 2 probaiblity distributions.
* LSGAN (Least Square GAN) is trying to improve both model stability and generated images' perceptive quality on DCGAN, by replacing sigmoid cross-entropy loss with least squares loss, since it doesn't bring in vanishing gradients during training
* ACGAN (Auxiliary GAN) is trying to improve both model stability and generated images' perceptive quality on CGAN, by requiring its discrimiator to perform an additional classification task
* [Reusable discriminator, generator and train() by multiple GANs][67]
#### WGAN
* In DCGAN and CGAN, the loss function is trying to minimizing the DISTANCE (JS distance, Jensen Shannon distance) between the target distribution and its estimate. However, for some pairs of distributions, there is no smooth path to minimize this JS distance, and the training will fail to converge by gradient descent.
  * The "distribution" here is probability distribution
  * When there is no overlap between the 2 distributions, there is no smooth function can close the gap between them
* The idea of EMD is, it's a measure of how much mass should be tranported in order to match the 2 probability distributions
  * EMD can be interpreted as the least amount of work needed to move the pile of dirt p to fill holes q
  * When there is no overlap between 2 distributions, it can provide a smooth function to match the 2 probability distributions
* WGAN is same as DCGAN, except:
  * WGAN is using `-1` as the label for fake data while DCGAN is using 0
    * In order to prevent the gradient from vanishing because of the opposite sign in real & fake lables, and the small magnitude of weights due to weights clipping, a tweak has been done in WGAN. Instead of training the weights in a single combined batch of real and fake data, it trains 1 batch of real data and then 1 batch of fake data
  * WGAN trains the distriminator `n_critic` iterations before training the generator for 1 iteration; In DCGAN, there is no `n_critic`, an in each iteration, both discriminator and generator will be trained once
    * In WGAN, after `n_critic` iterations of training discriminator, the training of generator will freeze the discriminator, then the discriminator will be unfrozen and start another `n_critic` iterations of distriminator training
  * The loss function in WGAN is using`wasserstein_loss`, while DCGAN is using `binary_crossentropy`
  * See [WGAN implementation and wasserstein_loss][66], [DCGAN implementation][64]
* But WGAN doesn't take care of the generated images' quality
#### LSGAN
* Fake samples on the correct side vs Fake samples' distribution is closer
  * Idealy, the fake samples' distribution should be as close as the real samples' distribution. However, in original GANs, once the fake samples are already on the correct side of the decision boundary, the gradients vanish. --> This prevents the generator from further improving the quality of generated fake data in order to move closer to the real samples' distribution.
  * The solution is to replace the loss function with least square loss function, the gradients won't vanish as long as the fake samples' distribution is still far from the real samples' distribution. This will motivate the generator to keep improving its estimate of real dentisy distribution, even if the fake samples are already on the correct side of the decision boundaty.
* LSGAN is same as DCGAN, except:
  * The loss function for both generator and discriminator is `MSE` in LSGAN
    * Least square error minimizes the "total" euclidean distance between a line and the data points. MSE is mean squared error, a good metric to evaluate the distance in average
  * There is linear avtivation or the avtivation is None in LSGAN
    * DCGAN can use 'sigmoid', 'relu', 'leakyrule'
    * LSGAN uses None or 'linear' for the activation function
  * See [LSGAN implementation][68], [DCGAN implementation][64]
#### ACGAN
* ACGAN makes the improvement upon CGAN by adding an auxiliary class decoder network. 
* ACGAN assumes forcing the network to do additional work will improve the performance of the original task.
  * The additional work is added classification
  * The original task is fake data generation
* ACGAN vs CGAN
  * The data input doesn't have to be images, but here let's use images as an example 
  * The input of CGAN discriminator is the image (fake or real) and the label of fake or real, output is the probability of fake or real. The input of ACGAN is the image (fake or real), and the input is the probability of fake or real, as well predicted class of the image
    * The loss function to predict fake or real is 'binary_corssentropy'
    * The loss function to predict image class is 'categorical_crossentropy'
    * The auxiliary decoder network is used to perform image classification. In the code, it's the extra steps added in CGAN's discriminator
  * [ACGAN implementation][69], [ACGAN discriminaton][67], [CGAN implementation][65]
    * Discriminator differences
    * Loss functions differences 

### Disentangled Representation GANs
* GANs can also learn disentangled latent codes or representations that can vary attributes of the generator outputs
* A disentangled code or representation is a tensor that can change a specific feature or attribute of the output data while not affecting other attributes
* * Comparing with the original GANs, if we're able to seperate the representation into entangled and disentangled interpretable latent code vectors, we will be able to tell the generator what to synthesize
  * For example, in an image, with disentangled codes,we can indicate thickness, gender, hair style, etc. 

#### InfoGAN
* InfoGAN learns the disentangled representation in an unsupervised way, by maximizing the mutual info between the input codes and output observation
  * In order to maximize this mutual info, InfoGAN proposes a regularizer that forces the generator to consider the latent codes when it synthesizes the fake data
* InfoGAN vs ACGAN
  * InfoGAN expands ACGAN by adding disentangled codes so that the generated output will be able to control specific attributes 
  * For the input of generator of both ACGAN and InfoGAN, both have fake labels. 
    * Besides InfoGAN has n codes for n disentangeled attributes
  * For discriminator, both have loss function to predict fake or real (binary_crossentropy)
    * ACGAN has categorical_corssentropy to predict input class, while InfoGAN has categorical_corssentropy for discrete code
    * Besides InfoGAN has `mi_loss` for EACH continuous code
  * [InfoGAN implementation][70], [ACGAN discriminaton][67]
    * The disentangle codes in InfoGAN
      * In this example , these are continuous codes, each code is drawn from a normal distribution with 0.5 std and 0 mean
    * Loss functions 
      * `λ` is the weight of loss functions
      * For continuous code, recommends to have `λ<1`, and in the code, it's using 0.5
      * For discrete code, recommends to have `λ=1`
      * The weight for binary_crossentropy is also 1
    * n attributes
      * You can add whatever number of disentangled code, just need to specific the input in generator and discriminator and loss functions for each of these code 

#### StackedGAN
* StackedGAN uses a pretrained encoder or classifer to help disentangle the latent codes.
* It breaks a GAN into a stack of GANs, that each is trained independently in discriminator-adversarial manner with its own latent code:
  * The encoder is first trained to provide features
  * Then the encoder-GANs are trained jointly to learn how to use the noise code to control the specific attributes of the generator output
<p align="center">
<img src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/images/stackGAN.PNG" width="600" height="400" />
</p>

* [StackGAN implementation][71]
  * binary_crossentropy is used to predict real or fake. MSE is used for each encoder. 
* StackGAN vs InfoGAN
  * InfoGAN has simpler structure and faster to train 

### Cross Domain GANs
* This technique can be used in computer vision when an image in the source domain is transferred to the target domain.

#### CycleGAN
* This method doesn't need aligned source and traget images in the training.
  * Most often, the aligned image pairs are not available or expensive to generate. Better to have a model that doesn't need aligned image pairs. Therefore, methods like pix2pix which required aligned pairs have limited capability.
* The structure of CycleGAN
  * In the forward cycle, the input data is real source data, while in the backward cycle, the input is real target data.
  * In both forward and backward cycles, both have cycle consistency check, to minimize the input data and reconstructed data
    * With cycle consistency check, even though we are transfering domain x to domain y, the original features in x should be intact in y and be recoverable
    * Cycle consistency check uses L1 loss (MAE, mean absolute error) so that the reconstructed images can be less bluring, comparing with L2 loss (MSE, mean squared error)
  * CycleGAN is symmetric. Forward cycle GAN is identical to the backward cycle GAN, but have the roles of the source data x and target data y reversed
    * Generator F is just another generator borrowed from the backward cycle GAN
  * <b>The objective of CycleGAN is to have generator G learn to synthesize fake target data y that can fool discriminator Dy in forward cycle, and have generator F learn to synthesize fake source daata x that can fool discrinminator Dx in backward cycle</b> 
<p align="center">
<img src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/images/cycleGAN.PNG" width="500" height="500" />
</p>

* [CycleGAN implementation][72]
  * Generator is implemented with [U-Net][73]
    * The problem of using autoencoder is, the lower level features are shared between both encoder and decoder layers, which is not suitable for image translation. 
    * U-Net is used to deal with this issue, it enables the free flow of feature-level info between paired encoder and decoder
    * IN (Instance Normalization) is BN (Batch Normalization) per image/sample or per feature
      * <b>In style transfer, it's important to normalize the contrast per sample, not per batch</b>. IN is equivalent to contrast normalization while BN breaks contrast normlaization.
    * An encoder layer is made of `IN-LeakyReLU-Conv2D`, a decoder layer is made of `IN-ReLU-Conv2D`
  * PatchGAN is also an option in the discriminator. By choosing this option, you can divide an image into patches and predict real/fake for each patch, predict the probability
    * Patches can overlap
    * This can help improving the generated images' quality since an image will look more real if each of its sub-images looks more real
  * Loss functions
    * MAE is used for cycle consistency check
    * MSE is used for generator and discriminator losses. This is inspired by LSGAN, that by replacing binary_crossentropy with MSE is improve the perceptual quality
    * `Total Loss = λ1 * Loss_GAN + λ2 * Loss_cycle_consistency + (λ3 * Loss_identity)`
      * `Total Loss` will be optimized in the adversarial training 
      * Loss_identity is optitional
        * Color composition may not be transferred from source image to the target image, that why the identity regularizers are added
        * The auxiliary identity regularizer will be added in the cycleGAN network, it also uses MAE as the loss function
<p align="center">
<img src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/images/cycleGAN_identity_regularizer.PNG" width="550" height="550" />
</p>

* Tips
  * When the 2 domains are drastically different, suggest to have smaller kernel_size. 
    * For example, in [CycleGAN implementation][72], `mnist_cross_svhn` is trying to translate images between MNIST digits and SVHM street view numbers...
  * Also as `mnist_cross_svhn` shown, the results of CycleGAN may not be sematic consistent. To address this issue, we can try `CyCADA (Cycle-Consistent Adversarial Domain Adaptation)` which adds a semantic loss to ensure the semantic consistency


### Other
* [A big of trick when tunning GAN][39]

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
[35]:https://nbviewer.jupyter.org/github/fchollet/deep-learning-with-python-notebooks/blob/master/6.3-advanced-usage-of-recurrent-neural-networks.ipynb
[36]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/sequencial_analysis/time_series_forecasting.ipynb
[37]:https://www.analyticsvidhya.com/blog/2020/08/a-beginners-guide-to-focal-loss-in-object-detection/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
[38]:https://nbviewer.jupyter.org/github/fchollet/deep-learning-with-python-notebooks/blob/master/6.4-sequence-processing-with-convnets.ipynb
[39]:https://nbviewer.jupyter.org/github/fchollet/deep-learning-with-python-notebooks/blob/master/8.5-introduction-to-gans.ipynb#A-bag-of-tricks
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
[55]:https://towardsdatascience.com/understanding-2d-dilated-convolution-operation-with-examples-in-numpy-and-tensorflow-with-d376b3972b25
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
[69]:https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter4-gan/cgan-mnist-4.3.1.py
[70]:https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter6-disentangled-gan/infogan-mnist-6.1.1.py
[71]:https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter6-disentangled-gan/stackedgan-mnist-6.2.1.py
[72]:https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter7-cross-domain-gan/cyclegan-7.1.1.py
[73]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/images/U_Net.PNG
[74]:https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter7-cross-domain-gan/other_utils.py
[75]:https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter7-cross-domain-gan/cifar10_utils.py
[76]:https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter7-cross-domain-gan/mnist_svhn_utils.py
[77]:https://www.analyticsvidhya.com/blog/2021/05/image-processing-using-numpy-with-practical-implementation-and-code/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
