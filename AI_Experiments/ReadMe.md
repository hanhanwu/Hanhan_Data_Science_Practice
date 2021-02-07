Time to try experiments in Neural Network, Deep Learning and any other AI methods :)
Neural Network is a universal approximator, which means you can use it to implment other machine learning algorithms

## Pretrained Models
### Pretrained Models in Computer Vision
* https://www.analyticsvidhya.com/blog/2018/07/top-10-pretrained-models-get-started-deep-learning-part-1-computer-vision/
### Fun Pretrained
* I found some pretrained deep learning models are doing well in a fun way!
* [Google DeepDream][30]
  * [An example to dreamify an image][31], simple words contains something to learn
    * "The idea in DeepDream is to choose a layer (or layers) and maximize the "loss" in a way that the image increasingly 'excites' the layers. The complexity of the features incorporated depends on layers chosen by you, i.e, lower layers produce strokes or simple patterns, while deeper layers give sophisticated features in images, or even whole objects."
      * There are layer0 to layer10, 11 layers to choose
    * In order to make the processed image looks more like a dream, better to generate patterns from different scales. So you can apply gradient ascent at different scales, in order to allow patterns generated from small scale to incorporate with patterns generated from the large scale.
      * To do this, you can apply gradient ascent approach while repeatedly resizing the image (octave)
    * To save the time and memory when doing the gradient calculation, especially for large images, you can split the image into tiles and compute the gradient for each tile.
    * Example output [image 1][32] & [dream 1][34], [image 2][33] & [dream 2][35]
* [Style Transformer][36]
  * The main idea is to minimize this loss 
    * `loss = distance(style(reference_image) - style(generated_image)) + distance(content(original_image) - content(generated_image))`
* [GPT-3][37]
  * It's a poswerful GPT series model that's trained with almost all the data on the Intenet, and therefore it can do many things such as writing essays, poems, writing code, writing songs, etc.. Also super expensive to train.


## RESOURCES
* Database
  * ImageNet: http://www.image-net.org
    * Here, you can download images in different format
  * 25 Open Datasets: https://www.analyticsvidhya.com/blog/2018/03/comprehensive-collection-deep-learning-datasets/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
* Standford CNN for visual Recognition lectures: http://cs231n.stanford.edu/syllabus.html
  * I think, once you started to learn deep learning, you will really feel it's so deep to learn....
  * My notes: https://github.com/hanhanwu/Hanhan_Data_Science_Resources2/blob/master/Standford_CNN_Notes1.pdf
* Tab-delimited Bilingual Sentence Pairs: http://www.manythings.org/anki/
  * Used for language translation, translate to English

* Here is a bunch of libraries, tutorials you can try: https://www.analyticsvidhya.com/blog/2016/08/deep-learning-path/

* 10 Deep Learning Architectures: https://www.analyticsvidhya.com/blog/2017/08/10-advanced-deep-learning-architectures-data-scientists/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29

* Fundamentals of neural network: https://www.analyticsvidhya.com/blog/2016/03/introduction-deep-learning-fundamentals-neural-networks/
  * It's such a great feeling to review and to understand deeper
  * After learning NN a while, are you going to forget about the relationship between weights, activation function and EACH neuron like I did? This article is a great source to review. So! Each neuron has an activation function, the input for this single neuron can come from multiple sources, each input and bias could has weight.
  * I think, the activation function can be whatever you defined. The purpose for defining an activation is to guarantee it will give the output you want. For example, the input is x1, x2 and the output should be `x1 AND x2`, so your activation function should use mathematical method to guarantee the output will be `x1 AND x2` when the input is x1, x2.
  * The power of multiple layers of NN - you can break a complex problem into small problems, and each NN layer deals with one of the small problem, finally, you get what you finally want
  * The number of neurons in the output layer will depend on the type of problem. It can be 1 for regression or binary classification problem or multiple for multi-class classification problems.
  * <b>The ultimate objective is to update the weights of the model in order to minimize the loss function</b>. The weights are updated using a back-propogation algorithm.
  * `A XNOR B = (A' AND B') OR (A AND B) = NOT ((A OR B) AND (A' OR B'))`, `A' = NOT A`
  * Each optimization includes:
    * Select a network architecture, i.e. number of hidden layers,  number of neurons in each layer and activation function
    * Initialize weights randomly
    * Use forward propagation to determine the output node
    * Find the error of the model using the known labels
    * <b>Back-propogate the error</b> into the network and determine the error for each node
    * Update the weights to minimize the loss function
  * During back-propagation:
    * <b>The error at a node is based on weighted sum of errors on all the nodes of the next layer which take output of this node as input.</b>
  * Important params for optimizing neural network
    * Type of architecture
    * Number of Layers
    * Number of Neurons in a layer
    * Regularization parameters
    * Learning Rate
    * Type of optimization / backpropagation technique to use
    * Dropout rate
    * Weight sharing
  * Think about the activation function, the architecture, do you think Deep Learning can really generate infinite number of solutions?! So powerful isn't it? The only problem is, you still have to have ground truth first, and for Deep Learning, the ground truth has to be more accurate and informative
    
* Core concepts of neural network
  * Perceptron
    * In biology, a neuron gets multiple inputs, sum them up and pass to the next neuron. A perceptron models how neuron works.
    * Multiple weighted inputs --> weighted sum --> bias --> activation function --> output.
      * A perceptron is a linear model that provides a binary output.
        * This is why a single layer perceptron is a linear model.
      * Weights determine the slope of classifier line, and bias helps shift the line towards left or right.
      * Activation function is not included in the perceptron.
    * The learning process of perceptron
      * Initialize the weights and thresholds
      * Get input and provide output
      * Update weights
      * Repeat 2,3 steps
  * Gradient vs Gradient Descent
    * Gradient is a numeric calculation allowing us to know how to adjust the parameters of a network in such a way that its output deviation is minimized. 
    * It's the multi-variable derivation of the loss function with respect to all the network parameters. Graphically it would be the slope of the tangent line to the loss function at the current point when evaluating the current parameter values. 
    * Mathematically it’s a vector that gives us the direction in which the loss function increases faster, so we should move in the opposite direction if we try to minimize it.
    * Gradient Descent can be thought of climbing down to the bottom of a valley by moving in the direction of steepest descent. It's an optimization algorithm that minimizes the loss function.
  * Batch Gradient Descent vs Stochastic Gradient Descent vs Mini Batch Gradient Descent
    * Batch gradient descent computes the gradient using the entire dataset, and performs just one update at each iteration.
    * Stochastic gradient descent computes the gradient using a single sample and updates the parameters.
      * Batch gradient descent updates weights slower and converge slower because of the data size, Stochastic gradient descent updates weights more frequent and therefore converge faster.
    * Mini Batch Gradient Descent is similar to Stochastic gradient descent but instead of using single training sample, min-batch of N samples is used. It's one of the most popular optimization methods.
      * It's more efficient than Stochastic gradient descent.
      * It's approximate the the gradient of the entire training set and helps avoid local minima.
    * With stochastic version of gradient descent, it's easier to escape from "saddle point".
      * "Saddle point" is the local minimum in one dimension but the local maximum in another dimension.
  * General steps of using gradient descent
    * Initialize random weights and bias
    * Get input and provides output
    * Calculate the error between actual and predicted values
    * Go back to each neuron that contributes to errors, changing its weights to reduce the error
    * Repeat until finding the best weights of the network
  * Use "momentum" in gradient descent
    * Here's the implementation example: https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/deep_learning/optimizers.py
    * The reason it works is because, with exponentially weighted average, it can average out the oscillations on its way finding the local optimal. So that the algorithm takes more straightforward path towards local optimum, which also means it takes less iteration.
      * I like the description here: https://engmrk.com/gradient-descent-with-momentum/
      * Bias can be 0 initially, weight can be close to 0 but not too small initially
    * With momentum, it's trying to help converge faster.
    * A typical choice of momentum is 0.5 ~ 0.9
    * Why momentum, more details
      * Gradient descent is a first-order optimization method, since it takes the first derivatives of the loss function. This gives us information on the slope of the function, but not on its curvature, so we lack part of the context. Using a second derivates can lead to high computational cost.
      * The key to this optimization lies in updating the network parameters by adding an extra term that considers the value of the last iteration update, so previous gradients will be taken into account in addition to the current one.
      * The value of the previous update by a constant known as the "momentum coefficient".
  * The meaning of "sample", "epoch", "batch"
    * <b>Sample</b>: One element of a dataset. Such as, 1 image, 1 audio file
    * <b>Batch</b>: An epoch is a full cycle of the algorithm in which the network sees all available samples once. The samples in a batch are processed independently, in parallel. If training, a batch results in only one update to the model. 
      * A batch generally approximates the distribution of the input data better than a single input. <b>The larger the batch, the better the approximation</b>, but also takes longer time.
      * A good defualt batch size is 32, we can try 32, 64, 128, 256, etc.
    * <b>Epoch</b>: 1 round of training on the entire dataset.
      * When using Keras `evaluation_data` or `evaluation_split` with the `fit` method of Keras models, <b>evaluation will be run at the end of every epoch</b>.
      * We can keep increasing epoch until validation accuracy starts to drop, even when training accuracy keeps increasing.
    * <b>Iteration</b>: N/batch_size, N is sample size
      * An epoch should run N/batch_size iterations
  * References
    * https://www.analyticsvidhya.com/blog/2016/08/evolution-core-concepts-deep-learning-neural-networks/
    * https://keras.io/getting-started/faq/#what-does-sample-batch-epoch-mean
    
* Activation functions and when to ues them: https://www.analyticsvidhya.com/blog/2017/10/fundamentals-deep-learning-activation-functions-when-to-use-them/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
  * "Activation functions are mathematical equations that determine the output of a neural network. The function is attached to each neuron in the network, and determines whether it should be activated (“fired”) or not, based on whether each neuron’s input is relevant for the model’s prediction. Activation functions also help normalize the output of each neuron to a range between 1 and 0 or between -1 and 1."
  * Without activation function, the weights, bias will simply do a linear transformation, and the neural network will work as a linear regression problem
  * <b>Binary Step Function</b>
    * Threshold based classifier, decide whether or not to activate the neuron
    * Only serves for binary classification, cannot work for multi-class classification
    * The gradient is 0, and cannot help back-propagation. Because back-propagation expects to use gradients to update errors and improve the model, but this function will set gradients to 0 and cannot really make any improvement
  * <b>Linear Function</b>
    * When you have multiple classes, choose the one with max value
    * But for linear function, the derivate is constant, which means gradient will be the same each time, still cannot help back-propagation
  * <b>Sigmoid Function</b>
    * `f(x)=1/(1+e^-x)`
    * non-linear
    * y ranges between [0,1], x ranges between [-infinite, infinite]. But because of the y range, sigmoid function is not symmetric around the origin and the values received are all positive
    * When gradient is approaching 0 (sigmoid curve is flat), the neuron is not really learning
  * <b>Tanh</b>
    * Scaled version of sigmoid function
    * `tanh(x)=2sigmoid(2x)-1`
    * It works similar to the sigmoid function but is symmetric over the origin. it ranges from -1 to 1
    * Your choice of using sigmoid or tanh would basically depend on the requirement of gradient in the problem statement
    * But similar to the sigmoid function we still have the vanishing gradient problem. When the gradient approaches to 0, the neuron is not really learn
  * <b>ReLU</b>
    * `f(x)=max(0,x)`
      * f(x)= x if x>= 0
      * f(x)=0 if x<0
    * non-linear
    * Main benefits
      * Simple math, cheap to compute and fast to train.
      * It converges faster. Linearity means that the slope doesn’t plateau, or “saturate,” when x gets large. It doesn’t have the vanishing gradient (very small update) problem suffered by other activation functions like sigmoid or tanh.
      * It’s sparsely activated. Since ReLU is zero for all negative inputs, it’s likely for any given unit to NOT activate at all. Sparsity results in concise models can lead to better predicitive power and less overfitting.
        * For example, a neron that can identify ears should not be fired when the image is a building. 
      * More computationally efficient to compute than Sigmoid like functions since Relu doesn't perform expensive exponential operations as in Sigmoids.
    * Drawback - Dying ReLu
      * A ReLU neuron is “dead” if it’s stuck in the negative side and always outputs 0. Once a neuron gets negative, it’s unlikely for it to recover. Such neurons are not playing any role in discriminating the input and is essentially useless.
    * ReLU function should <b>only be used in the hidden layers</b>
  * <b>Improvements on ReLU</b>
    * Reference: https://medium.com/@danqing/a-practical-guide-to-relu-b83ca804f1f7
    * Improve by making y value at x<0 non-constant
      * Leaky ReLU & Parametric ReLU (PReLU)
      * Exponential Linear (ELU, SELU)
      * Concatenated ReLU (CReLU)
    * Improve by learning sparse features earlier (when y=x=6)
      * ReLU-6
  * <b>Softmax</b>
    * Idealy used we you want the output is showing probability, because all the output are in range between [0,1]
  * <b>Summarized Suggestions from the author</b>
    * Sigmoid functions and their combinations generally work better in the case of classifiers
    * Sigmoids and tanh functions are sometimes avoided due to the vanishing gradient problem
    * ReLU function is a general activation function and is used in most cases these days
    * If we encounter a case of dead neurons in our networks the leaky ReLU function is the best choice
    * Always keep in mind that ReLU function should only be used in the hidden layers
    * As <b>a rule of thumb</b>, you can begin with using ReLU function and then move over to other activation functions in case ReLU doesn’t provide with optimum results
    
* How Regularization work in deep learning
  * In linear regression, we know regularization is used to penalize coefficients; Similarily, in deep learning, regularization is used to penalise the weight matrices of the nodes. 
    * Imagine when the weights are too low, close to 0 that the nodes won't contribute in the prediction and the model is near a linear model, this will underfit the data; when the weights are too high will overfit the data
    * So we need regularization to help optimize the weights
  * Commonly used regularization methods
    * Such as L1, L2, the combination of L1&L2
    * Check keras regularizer: https://keras.io/regularizers/
      * The parameter mean alpha
  * Dropout
    * It randomly selects some nodes and removes them along with all of their incoming and outgoing connections
    * It is the most frequently used regularization method in deep learning
    * Can be applied to both the hidden layer and the input layer
    * Dropout is usually preferred in a large neural network structure in order to introduce more randomness.
    * check keras dropout: https://keras.io/layers/core/#dropout
      * The parameter means the probability of being dropped out
      * Normally we choose 20% ~ 50% dropout. 20% is a good starting point.
  * Data Augmentation
    * Increase the training data size, to reduce overfitting
      * This can be very costly when data labeling is time consuming or has other challenges
      * But imagine your data is images, you can try:
        * Rotating
        * Flipping
        * Shiffting 
        * Scaling
        * Combined methods
        * Other
      * This method can provide a big leap in the improvements
    * Check keras image preprocessing methods: https://keras.io/preprocessing/image/
  * Early Stopping
    * it borrows the idea of cross validation, with both training data and validation data. When the validation erros starts to go up, the training stops immediately.
    * Check keras early_stopping: https://keras.io/callbacks/#earlystopping
      * `val_loss`, `val_error` all indicates the loss/error on validation data
      * `patience` indicates the number of epochs need to wait after the point where validation error starts to go up
        * <b>NOTE: it can happen that after x apoches, the validation error can increase again</b>, so it needs to pay more attention when tuning this parameter
   * My practice code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/deep_learning_regularization.ipynb
    
* Hierarchical Temporal Memory (HTM) - Real Time Unsupervised Learning
  * HTM replicates the functioning of the <b>Neocortex</b>, the component of real huan intelligence.
    * Our brain majorly has 3 parts:
      * Neocortex - major intelligence
      * Limbic System - supports emotions
      * Reptilian Complex - survival instincts, each as eating, sleeping
    * Neocortex has many different regions, such as processing visual and audio, etc. But different region in the brain have similar cellar structure, indicating that our brain is trying to solve similar problems while processing all types of sensary data.
      * Different regions are logically related to each other in a hierarchical structure.
  * HTM vs Deel Learning
  ![HTM vs Deep Learning](https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/HTM_vs_DeepLearning.png)
    * layer 1 in the picture is like a getting simple neural network functionality with HTM (obviously with the added benefits of HTM)
    * layer 2 in the picture is like a getting convolution neural network functionality with HTM
    * layer 3 in the picture is like a getting reinforcement learning functionality with HTM
    * layer 4 in the picture is like getting multiple CNNs to work with reinforcement learning and HTM
  * When HTM outperforms other learning methods (all have to be satisfied)
    * The input data is temporal
      * A simple way to check whether the data is temporal, is to randomly shuffle the data and check whether the semantics has changed
      * NOTE: checking temproal is not checking stationary, I think they are the opposite concepts
    * Huge amount of data that needs online learning
    * The data sources have different structures, such as images, audios
    * Need the model to learn continuously
    * Unsupervised Learning
  * Applicatons
    * Grok for anomalies detection: https://grokstream.com/product/
    * Numenta: https://numenta.com/
      * Stock volume anomalies
      * Rogue human behavior: https://numenta.com/assets/pdf/whitepapers/Rogue%20Behavior%20Detection%20White%20Paper.pdf
        * I think this one can also expose other info about each employee and looks scary to me
      * NLP prediction
      * Geospatial tracking
  * How HTM works: https://www.youtube.com/watch?v=XMB0ri4qgwc
    * Bottom up processing
    * Sparse distributed representation (SDR) - Input temporal data generated from various data sources is semantically encoded as a sparse array called SDR
      * Extremely noise resistant
      * The semantic encoding makes sure similar objects get similar SDR
    * Spatial pooling - it is the process of converting the encoded SDR into a sparse array
      * Make sure the sparsity of the output array is constant at all times, no matter how sparse the input is
      * Make sure the overlap or semantic nature of the input is maintained
        * Similar objects should have high overlap
    * Hebbian Learning - used for learning patterns
    * Boosting - it makes sure that we are using a high capacity of the spatially pooled output
    * Temporal Memory
      * The concept of temporal memory is based on the fact that each neuron not only gets information from lower level neurons, but also gets contextual information from neurons at the same level.
  * Reference: https://www.analyticsvidhya.com/blog/2018/05/alternative-deep-learning-hierarchical-temporal-memory-htm-unsupervised-learning/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
    * Python implementation: http://nbviewer.jupyter.org/github/numenta/nupic/blob/master/examples/NuPIC%20Walkthrough.ipynb
    * API to try: http://api.cortical.io/
    
* Deep Leaning for Computer Vision: https://www.analyticsvidhya.com/blog/2016/04/deep-learning-computer-vision-introduction-convolution-neural-networks/?lipi=urn%3Ali%3Apage%3Ad_flagship3_feed%3BxPn6EhynRquw3Evzrg79RA%3D%3D
  * Detailed analysis of activation functions
    * Sigmoid (not recommended for CNN)
      * Kill the gradient
      * Outputs are not zero-centered since it's between [0,1]
      * Taking the exp() is computationally expensive
    * Tanh (not recommended for CNN)
      * Still kill the gradient
      * But output can be zero-centered
    * ReLU (Rectified Linear Unit)
      * <b>most commonly used for CNN</b>
      * Gradient won’t saturate in the positive region
      * Computationally very efficient as simple thresholding is required
      * Empirically found to converge faster than sigmoid or tanh.
      * Drawbacks - Output is not zero-centered and always positive
      * Drawbacks - Gradient is killed for x<0. Few techniques like <b>leaky ReLU</b> and <b>parametric ReLU</b> are used to overcome this
      * Drawbacks - Gradient is not defined at x=0. But this can be easily catered using <b>sub-gradients</b> and posts less practical challenges as x=0 is generally a rare case
    * Data Preprocessing
      * Same image size
      * Mean centering: subtract mean value from each pixel
    * Weights Initialization
      * All zeros - a bad idea
      * Gaussian Random Variables: you need to play with the standard deviation of the gaussian distribution which works well for your network....
      * Xavier Initialization: It suggests that variance of the gaussian distribution of weights for each neuron should depend on the number of inputs to the layer. A recent research suggested that for ReLU neurons, the recommended update is, `np.random.randn(n_in, n_out)*sqrt(2/n_in)`
      * You might be surprised to know that 10-20% of the ReLUs might be dead at a particular time while training and even in the end.
    * CNN important layers
      * Convolution Layer - the layer that does the convolutional operation by creating smaller picture windows to go over the data.
        * The input of CNN is not vectors as neural network, but it's a multi-channeled image.
        * Using padding in convolution layer, the image size remains same.
        * Padding works by extending the area of which a convolutional neural network processes an image. The kernel scans each pixel and convert the data into smaller (sometimes larger) format. By adding the frame of the image through padding, it allows more space for the kernel to cover the image.
      * ReLU layer - it brings non-linearity to the network and converts all the negative pixels to zero. The output is a rectified feature map.
      * Pooling Layer - pooling is a down-sampling operation that reduces the dimensionality of the feature map.
        * It performs down-sampling operations to reduce the dimensionality and creates a pooled feature map by sliding a filter matrix over the input matrix.
      * Fully Connected Layer - Classifiy objects.
        * Each pixel is considered as a separate neuron just like a regular neural network. The last fully-connected layer will contain as many neurons as the number of classes to be predicted.
    * In this article, I think the correct formula for calculating output size should be: `(W-F+2P)/S + 1`
    * To calculate zero-padding size: `(F-1)/2`
    * Filters might be called kernels sometimes
    * Alexnet
    ![AlexNet](https://www.analyticsvidhya.com/wp-content/uploads/2016/03/fig-8.png)
    * It has code example using GraphLab with pre-trained model. Pre-trained model may increase the accuracy but difficult to find and if you use CPU, may take very long time to run
    

* Digits recognition with <b>TensorFlow</b> [lower level library]: https://www.analyticsvidhya.com/blog/2016/10/an-introduction-to-implementing-neural-networks-using-tensorflow/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
  * <b>Tensorflow Resources</b>: https://github.com/jtoy/awesome-tensorflow
  * Tensorflow Models: https://github.com/tensorflow/models
  * Google Object Detection API: https://github.com/tensorflow/models/tree/master/object_detection

* Digits recognition, using NN with <b>Keras</b> [higher level library]: https://www.analyticsvidhya.com/blog/2016/10/tutorial-optimizing-neural-networks-using-keras-with-image-recognition-case-study/
  * Keras has 3 backend you can use: https://keras.io/backend/
    * Tensorflow (from Google)
    * CNTK (from Microsoft)
    * Theano (from University of Montreal)
    * You can change between these backend
  * <b>KERAS BLOG</b>: https://blog.keras.io/category/tutorials.html
  * Install Keras: https://keras.io/#installation
  * <b>Keras Resources</b>: https://github.com/fchollet/keras-resources
  * <b>Keras Examples</b>: https://github.com/fchollet/keras/tree/master/examples
  * For installing TensorFlow, strongly recommend to use `virtualenv` (it's fast, simply and won't influence other installed python libraries): https://www.tensorflow.org/install/install_mac
    * `sudo pip install --upgrade virtualenv`
      * This may no longer works, try commands below:
        * `pip uninstall virtualenv`
        * `conda install virtualenv`
    * Then you create a new folder as virtual environment, such as folder myvirtual_env, then type `virtualenv --system-site-packages venv`, of course, it should be thr path of your folder
    * Now you can activate it, in this case, type command `source ~/venv/bin/activate `
  * After `virtualenv` installation and validaton, Commands to turn on and turn off virtual environment:
    * To activate the virtual environment, `$ source ~/venv/bin/activate      # If using bash, sh, ksh, or zsh`, change "venv" to your own virtual environment folder path
    * To activate the virtual environment, `$ source ~/venv/bin/activate.csh  # If using csh or tcsh`, change "venv" to your own virtual environment folder name
    * Then in your terminal, you will see `(venv)$`
    * To deactivate your virtual envvironment, `(venv)$ deactivate`
  * Install Tensorflow & Keras:
    * `sudo pip install tensorflow`
    * `sudo pip install keras`
    * These 2 commands works even when you are in conda virtual environment. I tried to use `conda` install but didn't works well.
  * Install Jupyter Notebook in your virtual environment
    * `(venv)$ pip install jupyter`, install jupyter within the active virtualenv
    * `(venv)$ pip install ipykernel`, install reference ipykernel package
    * `(venv)$ python -m ipykernel install --user --name conda_virtualenv --display-name "Python2 (venv)"`, set up the kernel. Here, if you will install multiple kernel, `testenv` name should be changed to other names
    * `(venv)$ jupyter notebook`
    * After jupyter notebook has been turned on, when you are creating a new notebook, choose "Python 2 (venv)"
    * NOTE: If you are using Python3, for example, python3.5, then in the above commands, change `pip` to `pip3`; change `python` to `python3.5`
  * Pros and Cons of Keras
    * Simple and no detailed implemention of NN like lower level libraries (e.g. Tensorflow) required, but also because of this, it can be less flexible
    * Only support GPU Nvidia
  * Keras Sequential models: https://keras.io/getting-started/sequential-model-guide/
    
* CNN for visual recognition: http://cs231n.github.io/neural-networks-3/
* Comparison between CNN, RCNN, Fast RCNN, Faster RCNN in object detection: https://www.analyticsvidhya.com/blog/2018/10/a-step-by-step-introduction-to-the-basic-object-detection-algorithms-part-1/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
    
* Image recognition with Keras: https://www.analyticsvidhya.com/blog/2017/06/architecture-of-convolutional-neural-networks-simplified-demystified/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
  * Stride & Padding
    * Same Padding (zero padding) - remain image size
    * Valid Padding - reduce features
  * Activation Map & Number of Filters
  * The convolution and pooling layers will only extract features and reduce the number of parameters from the  original images. Pooling is done for the sole purpose of reducing the spatial size of the image, but the depth of the image remains unchanged
  * Calculate output volume: `([W-F+2P]/S)+1`, W is the input volume size, F is the size of the filter, P is the number of padding applied and S is the number of strides.
  * Output layer loss function to compute the error in prediction, then backpropagation
  * <b>Entire CNN Network</b>
  
  ![entire network](https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/06/28132045/cnnimage.png)
    * Activation Map on convolution layer extracts features. 
    * Pooling is to further reduce features
    * When CNN goes further, the extracted features become more specific
    * Output layer makes prediction and has the loss function to check the prediction error
    * Backpropagation
    * Multiple rounds of forwardpropagation and backpropagation
    
    
* Use pre-trained model for digits recognition: https://www.analyticsvidhya.com/blog/2017/06/transfer-learning-the-art-of-fine-tuning-a-pre-trained-model/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29

* NN Implementation
  * Implement in Python and R: https://www.analyticsvidhya.com/blog/2017/05/neural-network-from-scratch-in-python-and-r/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
  * My NN code in python: https://github.com/hanhanwu/Hanhan-Machine-Learning-Model-Implementation/blob/master/neural_network.py
  * My NN code in real time short context search: https://github.com/hanhanwu/Hanhan_NLP/blob/master/short_context_search.py

* GPU for Deep Learning: https://www.analyticsvidhya.com/blog/2017/05/gpus-necessary-for-deep-learning/
  * Why GPU: https://www.slideshare.net/AlessioVillardita/ca-1st-presentation-final-published
  
* Deep Learning Book: http://www.deeplearningbook.org/
* Deep Learning Path: https://www.analyticsvidhya.com/blog/2016/08/deep-learning-path/
* Some Deep Learning tutorials: http://machinelearningmastery.com/start-here/

* Object Detection using YOLO [No Code Example]
  * Article: https://www.analyticsvidhya.com/blog/2017/08/finding-chairs-deep-learning-part-i/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
  * I really like the author's passion in deep learning!
  * Inspiration - what to check when something is wrong with the deep learning model
    * Step 1: Check the architecture
    * Step 2: Check the hyper-parameters of neural network
    * Step 3: Check the Complexity of network
    * Step 4: Check the Structure of Input data
    * Step 5: Check the Distribution of data
  * It cotains how to install YOLO for object detection, how to resize training images to improve accuracy
  
* Deep Learning Courses
  * [Course 1][11]
  * [Understanding Inception Network][12]

* 2017 Projects/Development in AI/Machine Learning
  * https://www.analyticsvidhya.com/blog/2017/12/reminiscing-2017-defining-moments-and-future-of-data-science/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
  * Most of them are Deep Learning, some are really good to read, such as PassGAN
* Sequence Model Use Cases: https://www.analyticsvidhya.com/blog/2018/04/sequence-modelling-an-introduction-with-practical-use-cases/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29

### Reinforcement Learning
#### Using Dynamic Programming in Reinforcement Learning
* Dynamic programming algorithms solve planning problems. Given the complete model and specifications of the environment (MDP), we can successfully find an optimal policy for the agent to follow. It contains two main steps:
  * Break the problem into subproblems and solve it
  * Solutions to subproblems are cached or stored for reuse to find overall optimal solution to the problem at hand
* DP can only be used if the model of the environment is known.
* DP Has a very high computational expense. It does not scale well as the number of states increase to a large number. An alternative called asynchronous dynamic programming helps to resolve this issue to some extent. 
* Reference: https://www.analyticsvidhya.com/blog/2018/09/reinforcement-learning-model-based-planning-dynamic-programming/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
  * The author provided detailed mathematics methods and his implementation in an example.
  * Step 1 - Policy Evaluation. The objective is to converge to the true value function for a given policy π.
  * Step 2 - Policy Improvement. We need a helper function that does one step lookahead to calculate the state-value function. This will return an array of length nA containing expected value of each action.
  * Step 3.1 - Policy Iteration. it will return the optimal policy matrix and value function for each state.
  * Step 3.2 - Value Iteration. Instead of waiting for the policy evaluation step to converge exactly to the value function vπ, ee can get the optimal policy with just 1 step of policy evaluation followed by updating the value function repeatedly.
  * Final observation is, value iteration has a better average reward and higher number of wins when it is run for 10,000 episodes.
  
### Attention Mechanism Deep Learning
* It's the idea that not only can all the input words be taken into account in the context vector, but relative importance should also be given to each one of them.  Whenever the proposed model generates a sentence, it searches for a set of positions in the encoder hidden states where the most relevant information is available. This idea is called ‘Attention’.
  * The most basic mechanism - each input has been given some importance, "global" attention. But this will increase computation when there is more input.
  * "Local" attention, instead of considering all the encoded inputs, only a part is considered for the context vector generation. This avoids epensive computation and it's also easier to train.
  
#### References
* [A comprehensive guide of attention mechanism deep learning][28]
  
## Tips
### Keras LSTM Changing Batch Size
* When using built-in method of keras, the batch size limits the number of samples to be shown to the network before a weight update can be performed. Specifically, the batch size used when fitting your model controls how many predictions you must make at a time.
* This will become an error when the number of predictions is lower than the batch size. For example, you may get the best results with a large batch size, but are required to make predictions for one observation at a time on something like a time series or sequence problem.
* So, it will be better to have different batch size for training and testing.
* My code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/LSTM_changing_batch_size.ipynb
  * <b>Just keep scrolling down... I know...</b>
  
### Keras ModelCheckPoint - save the best model for validation data
* Even with early stopping, there could be overfitting at the end, the best model may appear in the middle of the early stopping when both training and testing evaluation are improving. So in Keras, with `ModelCheckPoint`, it will help you save the model that got best evaluation results on validation/testing data.
* Code Snapshot
![model checkpoint code](https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/model_checkpoint.png)
* reference: https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/
  

## EXPERIMENTS

### Sequence Analysis Reltaed AI Practice:
* https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/sequencial_analysis/ReadMe.md#sequence-prediction

### Conda vs Non-Conda Environments
  * After you have installed so many python libraries, you may have already experienced 2 types of virtual environment and non-virtual environment. The installing mathods cannot apply to all of them, sometimes, 1 method just apply to 1 environment, but that's good enough if finally you can install all libraries in a certain environment and do your experiment. Let me write down my brief summary about using <b>non-virtual environment, python virtualenv and conda virtual environemnt</b>
  * Non-Virtual Environemnt - Your local Python site-packages
    * For new versions of Mac, better to use `sudo easy_install [package_name]`
    * But for some machines, pip also works. `pip install [package_name]` for Python 2.*
    * `pip3 install [package_name]` for python 3.*
  * Python `virtualenv` - It creates a virtual environment very fast
    * Insall virtualenv, `sudo easy_install virtualenv`, for more details, check https://www.tensorflow.org/install/install_mac
    * Also check the above commands I used for `virtualenv`
    * In this environment, you can just use `pip install [package_name]`, or `pip3 install [package_name]`
    * But! Sometimes there are libraries you just cannot intsall here. This maybe because the python package does not match the requirements for this specific library... Then, maybe get some help from Conda Virtual Environment
  * Conda Virtual Environemnt
    * For detailed commands, check my posts here: https://stackoverflow.com/questions/45707010/ipython-importerror-cannot-import-name-layout/45727917#45727917
    * Sometime! You can use `conda install [package_name]`. When this command does not work, try `pip install [package_name]` or `pip3 install [package_name]`. By changing between these 2 types of commands, finally, I got all the libraries I want in Conda Environemnt


### Digit Recognition with Keras
  * Adam Optimizer: https://arxiv.org/abs/1412.6980
    * In addition to storing an exponentially decaying average of past squared gradients like Adadelta and RMSprop, Adam also keeps an exponentially decaying average of past gradients, similar to momentum. Adam is usually the fastest one of these optimization techniques.
  * Tensorflow optimizers: https://www.tensorflow.org/addons/api_docs/python/tfa/optimizers
    * Gradient descent is the most basic optimizer: https://algorithmia.com/blog/introduction-to-optimizers
    * Improvement on Adam: RAdam, LookAhead: https://medium.com/@lessw/new-deep-learning-optimizer-ranger-synergistic-combination-of-radam-lookahead-for-the-best-of-2dc83f79a48d
  * Supported Optimizers in Keras: https://keras.io/optimizers/
    * RMSProp (Root Mean Square Propagation), in this method the learning rate is adapted for each parameter, as in other method known as Adagrad. RMSProp improves the latter by including the exponential moving average of the squared gradient.
    * `AMSgrad` is a recent proposed improvement to Adam. It has been observed that for certain datasets, Adam fails to converge to the globally optimal solution, whereas simpler algorithms like SGD do.
      * `keras.optimizers.adam(amsgrad=True)`
    * Detailed comparison of keras optimizers: https://www.kaggle.com/residentmario/keras-optimizers
  * Supported loss functions in Keras: https://keras.io/losses/
  * NN used in this practive
    * Multi-Layer Perceptrons (MLP)
    * Convolutional neural network (CNN)
    * In this experiment, CNN is slower, but got better validation results
  * Methods used in improving Multi-Layer Perceptrons (MLP)
    * Add hidden layers
    * Use dropout to avoid overfitting. Dropout is essentially randomly turning off parts of the model so that it does not “overlearn” a concept
    * Increase epochs
  * My code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/digital_recognition_Keras.ipynb
  * Download the data: https://datahack.analyticsvidhya.com/contest/practice-problem-identify-the-digits/
    * You need to register
    * In fact the data came from MINIST data (a dataset used for digital recognition), downloading from above link you just need to download 2 files instead of 4
  * Reference: https://www.analyticsvidhya.com/blog/2016/10/tutorial-optimizing-neural-networks-using-keras-with-image-recognition-case-study/
  
  
### Digit Recognition with Unsupervised Deep Learning
  * My code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/unsupervised_deep_learning.ipynb
    * The evaluation method is NMI (Normalized Mutual Information), which calculates the purity of each cluster in order to measure the clustering quality. Higher NMI, the higher purity the clusters have.
    * This code used 3 methods:
      * Method 1 Simple kmeans to clustering the digits
      * Autoencoder
        * It has encoder & decoder. Encoder will convert the input to a lower dimensional representation, while decoder will recreate the input from this lower dimensional representation.
        * Detailed description about autoencoder from Keras creator's blog: https://blog.keras.io/building-autoencoders-in-keras.html
        * Method 2 - DIY Autoencoder, it reduce the dimensions of the data and extract the useful info, then pass to kmean [very slow]
        * Method 3 - Using DEC (Deep Embedding Clustering): https://github.com/XifengGuo/DEC-keras
          * You need large epoch and clustering iteration, so that the final NMI can be higher. But this will also be computational costly. In my code, I just used at most 7 rounds iteration for clustering, otherwise I need to wait for very long time
          * That DEC code, you need to copy the author's all the code from https://github.com/XifengGuo/DEC-keras/blob/master/DEC.py
          
### More About Autoencoder
* `input_layer -> hidden_layer` is called "encoding", and `hidden_layer -> output_layer` is called "decoding".
* Properties
  * Autoencoders are data-specific, which means that they will only be able to compress data similar to what they have been trained on.
  * Autoencoders are lossy, which means that the decompressed outputs will be degraded compared to the original inputs (similar to MP3 or JPEG compression).
  * Autoencoders are learned automatically from data examples, which is a useful property: it means that it is easy to train specialized instances of the algorithm that will perform well on a specific type of input. It doesn't require any new engineering, just appropriate training data.
* Practical Use
  * Help dimensional reduction. 
    * The input and output has same number of dimensions, and the hidden layer has less dimensions, thus it contains compressed informations of input layer, which is why it acts as a dimension reduction for the original input. "Decoding" is the lossy reconstruction of the input.
    * For example, t-SNE can plot data into 2D or 3D, but it doesn't work well when the original dimension is large. So a good practice is to have autoencoder help reduce the dimensions first, then use t-SNE for 2D, 3D plot.
  * Autoencoder vs PCA
    * Autoencoder is more time consuming, but when the data is too large that the memory is not enough to store, PCA cannot handle but autoencoder can handle the memory limitation.
  * [Python Example - Autoencoder for Dimensional Reduction][22]
    * Dimensional reduced data came from `encoder.predict()`, `encoder` here is the intermediate result of `input -> hidden layer`
    * With dimensional reduced data, it's later using lighGMB for model training
  * [R Example - Autoencoder for Dimension Reduction vs PCA][25]
    * [Reference][26]
    * [Generated HTML R Notebook][24]
      * The drawback of R Notebook is, its output cannot be visualizable in GitHub. But when using `plotly` in R, no need credentials as python does.
      * When using PCA first 3 principle components to see how they seperate gender
      * "k" means the number of selected principle components, when k is smaller, PCA has larger reconstruction error than autoencoder, but when k became larger, the reconstruction error became the same
      <p align="left">
<img width="400" height="300" src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/images/PCA_3d.png">
 </p>
 
  * It doesn't work well in data compression
    * Such as image compression, JPEG can do better. Plus autoencoder can only work for a specific set of images, more limitation.
  
    
#### Variational autoencoder (VAE)
* VAE is a type of autoencoder with added constraints on the encoded representations being learned. More precisely, it is an autoencoder that learns a latent variable model for its input data. So instead of letting your neural network learn an arbitrary function, you are learning the parameters of a probability distribution modeling your data. If you sample points from this distribution, you can generate new input data samples: a VAE is a "generative model".

* References
  * [Build Autoencoders in Keras][21]
    * Different types of autoencoder for image recognition on digits.
    * Image denoising.
    * VAE
  * [Python Example - Autoencoder for Dimensional Reduction][23]
    * [Its Code][22]
    
    
### Digital Recognition with Tensorflow
  * “TensorFlow is an open source software library for numerical computation using dataflow graphs. Nodes in the graph represents mathematical operations, while graph edges represent multi-dimensional data arrays (aka tensors) communicated between them. The flexible architecture allows you to deploy computation to one or more CPUs or GPUs in a desktop, server, or mobile device with a single API.”
  * Tensorflow Workflow
    * Build a computational graph, this can be any mathematical operation TensorFlow supports.
    * Initialize variables
    * Create session
    * Run graph in session, the compiled graph is passed to the session, which starts its execution. 
    * Close session
  * Terminology in Tensorflow
    * <b>placeholder</b>: A way to feed data into the graphs
    * <b>feed_dict</b>: A dictionary to pass numeric values to computational graph
  * Process for building neural network
    * Define Neural Network architecture to be compiled
    * Transfer data to your model
    * Under the hood, the data is first divided into batches, so that it can be ingested. The batches are first preprocessed, augmented and then fed into Neural Network for training
    * The model then gets trained incrementally
    * Display the accuracy for a specific number of timesteps
    * After training save the model for future use
    * Test the model on a new data and check how it performs
  * Tensorflow Resources
    * Optimizers can be found here: https://www.tensorflow.org/api_docs/python/tf/train
  * Reference: https://www.analyticsvidhya.com/blog/2016/10/an-introduction-to-implementing-neural-networks-using-tensorflow/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
  * My code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/digital_recognition_TensorFlow.ipynb
  
  
### Image Recognition with Keras
  * <b>Better to use Conda virtual environment</b>
  * Commands to install required libraries in conda virtual environment
    * `conda install -c menpo opencv`
    * `pip install tensorflow`
    * `pip install keras`
  * Download images: https://github.com/hanhanwu/Basic_But_Useful/blob/master/python_download_images.ipynb
  * <b>Basic Version</b>
    * reference: https://www.analyticsvidhya.com/blog/2017/06/architecture-of-convolutional-neural-networks-simplified-demystified/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
    * My code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/Image_Recognition_Keras_simple_example.ipynb
    * rose images I'm using: http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n04971313
    * sunflower images I'm using: http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n11978713
  * <b>NOTE</b>: As you can see in my code above, I'm reading local image file instead of reading image urls directly. This is because, [when I tried to read image urls directly, no matter how did I change the code, there is dimension issue][1], using `scipy.misc.imread()` to read local file can get the right dimension of images, and it only supports local files 
    * This is the image data I got after reading from url directly 
    ![wrong dimwnsion](https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/wrong_dimensions.png)
    * This is the image data I got after reading from local file
    ![right dimwnsion](https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/right_dimensions.png)
  * <b>Level 2 Version</b>
    * My code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/image_recognition_Keras_level2.ipynb
    * In this code, I added more layers in CNN and did some random prediction
    * The most important thing is, I have learned much more about how to adjust the param values in CNN.... This is challenging for beginners, since there will be color and gray images, and the settings will be different. A wrong value could lead to long time debudding without really sure where is the problem. Let me write down a few summary here:
      * In [my digital recognition code][6], you will see `train_x_temp = train_x.reshape(-1, 28, 28, 1)`, this is because the image is gray, so the depth is 1 and the image size is 28*28*1, -1 here is to make it 4 dimensions as `model.fit()` requires. But in [image recognition level 2 code][7], you are seeing `training_image_temp = training_images.reshape(-1, 100, 100, 3)`, this is because the images are color images (RGB images), the depth should be 3, and size is 100*100*3
      * When you are using `pylab.imshow(testing_img)` to show images, whether you could show color image or 1-color image depends on this line of code `testing_img = scipy.misc.imread(testing_image_path)`, if you set `flatten=Ture` in `imread`, will be 1-color, otherwise can be RGB image
  
  
### GraphLab for Image Recognition
  * You need to register GraphLab email first (it's free): https://turi.com/download/academic.html
  * Right after the email registration, an install page will appear with your email and GraphLab key filled in pip command line: https://turi.com/download/install-graphlab-create-command-line.html
    * Type `sudo pip install --upgrade --no-cache-dir https://get.graphlab.com/GraphLab-Create/2.1/[your registered email address here]/[your product key here]/GraphLab-Create-License.tar.gz`, the email and product key here should be yours. <b>sudo</b> here is very important to me during the installation
  * The side effects of the command line is, it will uninstall something related to `shutil`, then when you want to run you ipython kernel, it will fail. by showing errors such as "No module named shutil_get_terminal_size". To solve this problem, run commands after the above command
    * `pip install --upgrade setuptools pip`
    * `pip uninstall ipython`
    * `pip install ipython`
    * `pip install ipykernel`, this maybe optional
    * `python -m ipykernel install --user --name testenv --display-name "Python2 (yourenvname)"`, here need to change `testenv` and `(yourenvname)`
    * `conda install ipywidgets --no-deps`
    * Then you can open ipython and `import graphlab`
  * [Get started with GrapgLab][2]
  * [GraphLab Create API][4]
  * [Image dataset - CIFAR-10][3]
  * My code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/try_GraphLab.ipynb
  * [reference][5]
  
  
### Audio Classification with Deep Learning
  * Dataset: https://drive.google.com/drive/folders/0By0bAi7hOBAFUHVXd1JCN3MwTEU
  * My code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/try_audio_deep_learning.ipynb
  * Audio Feature Extraction
    * Major Feature Types
      * Time Domain Features
      * Frequency Domain Features
      * Perceptual Features
      * Windowing Features
    * `librosa` provided methods for audio feature extraction: https://librosa.github.io/librosa/feature.html
   * reference: https://www.analyticsvidhya.com/blog/2017/08/audio-voice-processing-deep-learning/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
   
   
### Neural Network Word Embedding
  * Through word embedding, each word has a numerical value in the vector. Trough learning, words with similar meaning will stay close in the vetor space, which also means they have similar representation in the vector space
  * Keras has its own word embedding layer. You can also replace this embedding layer with a general pre-trained embedding methods (such as word2vec). You can also use specific trained embedding model, for the movie review data here, you can try Stanford GloVe.
  * Download Stanford GloVe pretrained embedding model here: https://github.com/stanfordnlp/GloVe
  * But when you are NOT using neural network its own embedding layer, it can be slow in Keras, and may also be less accurate. Since training data influence a lot in final results for text data, you can rarely find pre-trained model just used your dataset
  * My code 1: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/text_embedding_NN.ipynb
    * Step 1 - Generate vocabulary (word, count) first
    * Step 2 - CNN to predict sentiment (positive or negative) with word embedding
    * NOTE 1 - In this code, `texts_to_sequences()` apply to a list of tokens joined through ' ', it won't work on a list
    * NOTE 2 - Also in `texts_to_sequences()`, only exactly the same word will be encoded with the same word index 
  * My code 2: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/stemming_text_embedding_NN.ipynb
    * Compared with code 1, the code here just added 1 line of code - "stemming". Because in code 1, each unique word will have a numerical value, so I was wondering whether stemmed result can be better, since it converts words with similar meaning to the same format before word embedding. The fially result improved a little bit.
    * But looking back, similar words with different format can still be good, since they can reflect the word position and meaning better.
    
    
### [Python] Sequence to Sequence with Attention
* Beam Search
  * When generating words as a sequences, at each time step, the decoder has to make a decision as to what the next word would be in the sequence. One way to make a decision would be to greedily find out the most probable word at each time step.
  * Beam search takes into account the probability of the next k words in the sequence, and then chooses the proposal with the max combined probability.
* Attention Mechanism
  * It takes input from each time step of the encoder – but give weights to the timesteps. The weights depend on the importance of that time step for the decoder to optimally generate the next word in the sequence
* reference: https://www.analyticsvidhya.com/blog/2018/03/essentials-of-deep-learning-sequence-to-sequence-modelling-with-attention-part-i/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
  * it simply copied part of the Keras sample code.... That's sneaky, I don't like it.
* One of the practical application is language translation. Here's the sample code of using Keras, LSTM to translate the language: https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py
* My Language Translation practice code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/deep_learning_sequence2sequence.ipynb
  * I just wanted to see how it works, so copied the whole code and changed the language & parameter. Got the cold recetly, have to go to sleep early...
* Other Sequence2Sequence NLP
  * https://www.analyticsvidhya.com/blog/2019/01/neural-machine-translation-keras/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
    * It has more description in data preprocessing. The issue for this tutorial and my code above is the same, you need to use large amout of computing resource in order to get a satisfied result.
* NLP translation applications - use directly
  * TextBlob translation
   

### [R] Try Neural Network in R
  * Scale the data before using NN
    * min-max normalization - <b>remains original distribution</b>
    * Z-score normalization
    * median & MAD method
    * tan-h estimators
  * download the data: https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/09/07122416/cereals.csv
  * about the data: http://lib.stat.cmu.edu/DASL/Datafiles/Cereals.html
  * My code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/R_neural_network_basics.R
    * The plots here are very useful
      * plot trained NN
      * plot grouod truth and predicted results (using reversed scale)
      * boxplot to show RMSE at a certain length of training data
      * plot RMSE changes along the length of training data, while you are implementing k-fold cross validation on your own
  * reference: https://www.analyticsvidhya.com/blog/2017/09/creating-visualizing-neural-network-in-r/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
  
  
### Recurrent Neural Network (RNN)
* <b>The difference between RNN and basic neural network:</b> As we all know, a basic neural network has 1 input layer, 1 output layer plus 1 or more hidden layer. Also a neural network only stores the input and output for a layer each step. The difference in RNN is, it only has a recurrent layer between the input layer and the output layer, this recurrent layer is similar to multiple hidden layers in a basic neural network. But it can store the state of a previous input and combines with the current input, so it keeps some relationship between previous and current input
  * It can be one time step or multiple time steps
  * It also has Forward Propagation & Backward Propagation
  * How to to backward propagation in RNN
    * The <b>cross entropy error</b> is first computed using the current output and the actual output
    * Remember that the network is unrolled for all the time steps
    * For the <b>unrolled network</b>, the gradient is calculated for each time step with respect to the weight parameter
    * Now that the weight is the same for all the time steps the gradients can be combined together for all time steps
    * The weights are then updated for both recurrent neuron and the dense layers
  * Here, "unrolled network" looks just like basics neural network, and the backpropagation here is similar to basic neural network but it combines the gradients of error for all time steps
* <b>Vanishing Gradient</b>
  * When we do Back-propagation, the gradients tend to get smaller and smaller as we keep on moving backward in the Network. This means that the neurons in the Earlier layers learn very slowly as compared to the neurons in the later layers in the Hierarchy.
  * Examples: "Baby Emmanuel, who loves eating vegetables and cares about animals, ______ (don't or doesn't) like pork"
  * In the example, whether the blank should be filled with "don't" or "doesn't" depends on subject "Baby Emmanuel", however it's far away from the the blank. RNN suffers from Vanishing Gradient tend to miss out relations between words which are far away from each other
* <b>Exploding Gradient Descent</b>
  * RNN could also suffer from Exploding Gradient Descent.
  * Exploding gradients are a problem when large error gradients accumulate and result in very large updates to neural network model weights during training. This could lead to poor performance results.
* LSTM, GRU are the architectures of RNN that can solve the issue of vanishing gradient.
* GRU (Gated Recurrent Unit)
  * [GRU Mathematical Concept][8]
  * Hidden state for example "who loves eating vegetables and cares about animals"
  * RNN modifies hidden states, GRU simply give a bypass option to hidden state. So in the above "Baby Emmanuel" example, it can ignore the hidden state, and correctly fill in "doesn't"
* Because most of RNN practice are sequential analysis, I put the code here: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/tree/master/sequencial_analysis
  * Check those sections start with "RNN - "
* LSTM
  * LSTM is supposed to perform better than GRU in sentences with long dependencies. It has more params and therefore requires longer time to train.
  * LSTM has forget gate to control how much previous info will be sent to the next cell, whereas GRU exposes its entire memory to the next cell
  * Controlling the hidden state that is moving forward to the next cell can give us complete control of all gates in the next cell.
  * [LSTM Mathematical Concept Difference with GRU][8]
* Both GRU and LSTM understand a text from left to right, sometimes you need to read to the right side and go back to the left side, this requires to add "Bidirectional" into RNN
  
### Encoder-Decoder RNN Architecture
  * Sequence to Sequence problem: <b>It takes a sequence as input and requires a sequence prediction as output</b>. Input length can be different from the output length.
  * [Patterns for the Encoder-Decoder RNN Architecture][9]
  * [Encoder-Decoder LSTM Implementation through Keras][10]
    * "One or more LSTM layers can be used to implement the encoder model. The output of this model is a fixed-size vector that represents the internal representation of the input sequence. The number of memory cells in this layer defines the length of this fixed-sized vector."
    * "One or more LSTM layers can also be used to implement the decoder model. This model reads from the fixed sized output from the encoder model."
    * "The <b>RepeatVector</b> is used as an adapter to fit the fixed-sized 2D output of the encoder to the differing length and 3D input expected by the decoder."
    * "The TimeDistributed wrapper allows the same output layer to be reused for each element in the output sequence."
    
* Resources I didn't work on
  * CoreML for Spam classification on iPhone app development: https://www.analyticsvidhya.com/blog/2017/09/build-machine-learning-iphone-apple-coreml/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
  
### Pytorch
* How to install: `conda install pytorch torchvision cuda91 -c pytorch`
* Pytorch has 200+ mathematical operations: https://pytorch.org/docs/stable/torch.html#math-operations
* [My code - Pytorch for Digit Recognition][13]
  * Comparing with [My code - Keras for Digit Recognition][14], I still think Keras is much more convenient to use. It took me so much time to debug here because here you need to create batches yourself (Keras will do that for you). And because of the frequent package updates, missing 1 param will cause so much troubles later.
  
### Adversarial Machine Learning
* [Adversarial Machine Learning with foolbox][27]


## RELAVANT PAPERS & NEWS

* Google Deep Learning Products
  * Google TensorFlow Object Detection API: https://research.googleblog.com/2017/06/supercharge-your-computer-vision-models.html
  * MobileNet: https://research.googleblog.com/2017/06/mobilenets-open-source-models-for.html
  * Street View & Captcha
    * https://research.googleblog.com/2017/05/updating-google-maps-with-deep-learning.html
    * https://security.googleblog.com/2014/04/street-view-and-recaptcha-technology.html

* Adam Optimizer: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/Adam_optimizer.pdf
  * Don't quite understand, but I really admire those PhDs who design and compare algorithms...
  
* [RAdam][19]
  * It suggests that the convergence issue we face in deep learning techniques is due to the undesirably big variance of the adaptive learning rate in the early stages of model training.
  * RAdam is a new variant of Adam, that rectifies the variance of the adaptive learning rate. This release brings a solid improvement over the vanilla Adam optimizer which does suffer from the issue of variance.
  
* Practical Recommendations for Gradient-Based Training
  * https://github.com/hanhanwu/readings/blob/master/Practical%20Recommendations%20for%20Gradient-Based%20Training.pdf
  * Detailed suggestions in almost all the process in Gradient-Based training, very practical

## Deep Learning in Real World
* [How to use forward feed network to detect PowerShell Obfuscation][15]
  * Obfuscation is typically used to evade detection
  * Microsoft PowerShell is the ideal attacker’s tool in a Windows operating system. Because it is installed by default in Windows, and attackers are better off using existing tools that allow them to blend well and possibly evade Antivirus (AV) software.
  * They used character-level representation for all PowerShell scripts, instead of word-based, since obfuscated scripts are different from normal scripts
* [AI Projects for Creating Arts, Music - Including Source Code][16]
* [GPT-2 - Text Generator][17]
* [word2vec in product recommendation][18]
* [DeepPrivacy - Face Detection & Switch Face][20]
* Model Deployment
  * [Tensorflow Serving][29]
    * Version control: auto update, rollback
    * Configuration update
    * Deploying multiple models


[1]:https://stackoverflow.com/questions/45912124/python-keras-how-to-generate-right-image-dimension
[2]:https://www.analyticsvidhya.com/blog/2015/12/started-graphlab-python/
[3]:https://www.cs.toronto.edu/~kriz/cifar.html
[4]:https://turi.com/products/create/docs/index.html
[5]:https://www.analyticsvidhya.com/blog/2016/04/deep-learning-computer-vision-introduction-convolution-neural-networks/?lipi=urn%3Ali%3Apage%3Ad_flagship3_feed%3BxPn6EhynRquw3Evzrg79RA%3D%3D
[6]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/digital_recognition_Keras.ipynb
[7]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/image_recognition_Keras_level2.ipynb
[8]:https://www.analyticsvidhya.com/blog/2018/04/replicating-human-memory-structures-in-neural-networks-to-create-precise-nlu-algorithms/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
[9]:https://machinelearningmastery.com/implementation-patterns-encoder-decoder-rnn-architecture-attention/
[10]:https://machinelearningmastery.com/encoder-decoder-long-short-term-memory-networks/
[11]:https://www.analyticsvidhya.com/blog/2018/10/introduction-neural-networks-deep-learning/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
[12]:https://www.analyticsvidhya.com/blog/2018/10/understanding-inception-network-from-scratch/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
[13]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/digit_recognition_Pytorch.ipynb
[14]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/digital_recognition_Keras.ipynb
[15]:https://www.analyticsvidhya.com/blog/2019/05/using-power-deep-learning-cyber-security-2/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
[16]:https://www.analyticsvidhya.com/blog/2019/07/8-impressive-data-science-projects-create-art-music-debates/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
[17]:https://www.analyticsvidhya.com/blog/2019/07/openai-gpt2-text-generator-python/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
[18]:https://www.analyticsvidhya.com/blog/2019/07/how-to-build-recommendation-system-word2vec-python/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
[19]:https://github.com/LiyuanLucasLiu/RAdam
[20]:https://github.com/hukkelas/DeepPrivacy
[21]:https://blog.keras.io/building-autoencoders-in-keras.html
[22]:https://github.com/MJeremy2017/Machine-Learning-Models/blob/master/AutoEncoder/autoencoder.ipynb
[23]:https://towardsdatascience.com/autoencoder-on-dimension-reduction-100f2c98608c
[24]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/autoencoder_dimensional_reduction.nb.html
[25]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/autoencoder_dimensional_reduction.Rmd
[26]:https://www.r-bloggers.com/pca-vs-autoencoders-for-dimensionality-reduction/
[27]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/Henso_Jutsu_101.ipynb
[28]:https://www.analyticsvidhya.com/blog/2019/11/comprehensive-guide-attention-mechanism-deep-learning/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
[29]:https://www.analyticsvidhya.com/blog/2020/03/tensorflow-serving-deploy-deep-learning-models/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
[30]:https://github.com/google/deepdream
[31]:https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/generative/deepdream.ipynb
[32]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/images/test1.jpg
[33]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/images/test2.jpg
[34]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/images/dream1.png
[35]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/images/dream2.png
[36]:https://nbviewer.jupyter.org/github/fchollet/deep-learning-with-python-notebooks/blob/master/8.3-neural-style-transfer.ipynb
[37]:https://www.analyticsvidhya.com/blog/2021/01/gpt-3-the-next-big-thing-foundation-of-future/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
