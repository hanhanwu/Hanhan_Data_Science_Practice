
Time to try experiments in Neural Network, Deep Learning and any other AI methods :)


*****************************************************************

RESOURCES

* Fundamentals of neural network: https://www.analyticsvidhya.com/blog/2016/03/introduction-deep-learning-fundamentals-neural-networks/
  * Important params for optimizing neural network
    * Type of architecture
    * Number of Layers
    * Number of Neurons in a layer
    * Regularization parameters
    * Learning Rate
    * Type of optimization / backpropagation technique to use
    * Dropout rate
    * Weight sharing
    
* Core concepts of neural network: https://www.analyticsvidhya.com/blog/2016/08/evolution-core-concepts-deep-learning-neural-networks/
  * The meaning of "sample", "epoch", "batch": https://keras.io/getting-started/faq/#what-does-sample-batch-epoch-mean
    * <b>Sample</b>: One element of a dataset. Such as, 1 image, 1 audio file
    * <b>Batch</b>: A set of N samples. The samples in a batch are processed independently, in parallel. If training, a batch results in only one update to the model. A batch generally approximates the distribution of the input data better than a single input. <b>The larger the batch, the better the approximation</b>, but also takes longer time.
    * <b>Epoch</b>: Epoch: an arbitrary cutoff, generally defined as "one pass over the entire dataset", used to separate training into <b>distinct phases</b>, which is useful for <b>logging and periodic evaluation</b>. When using Keras `evaluation_data` or `evaluation_split` with the `fit` method of Keras models, evaluation will be run at the end of every epoch.

* Digits recognition with <b>TensorFlow</b> [lower level library]: https://www.analyticsvidhya.com/blog/2016/10/an-introduction-to-implementing-neural-networks-using-tensorflow/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29

* Digits recognition, using NN with <b>Keras</b> [higher level library]: https://www.analyticsvidhya.com/blog/2016/10/tutorial-optimizing-neural-networks-using-keras-with-image-recognition-case-study/
  * Keras has 3 backend you can use: https://keras.io/backend/
    * Tensorflow (from Google)
    * CNTK (from Microsoft)
    * Theano (from University of Montreal)
    * You can change between these backend
  * Install Keras: https://keras.io/#installation
  * <b>Keras Resources</b>: https://github.com/fchollet/keras-resources
  * For installing TensorFlow, strongly recommend to use `virtualenv` (it's fast, simply and won't influence other installed python libraries): https://www.tensorflow.org/install/install_mac
  * After `virtualenv` installation and validaton, Commands to turn on and turn off virtual environment:
    * To activate the virtual environment, `$ source ~/Documents/Virtual_Env/bin/activate      # If using bash, sh, ksh, or zsh`, change "Documents/Virtual_Env" to your own virtual environment folder name
    * To activate the virtual environment, `$ source ~/Documents/Virtual_Env/bin/activate.csh  # If using csh or tcsh`, change "Documents/Virtual_Env" to your own virtual environment folder name
    * Then in your terminal, you will see `(Virtual_Env)$`
    * To deactivate your virtual envvironment, `(Virtual_Env)$ deactivate`
  * Install Jupyter Notebook in your virtual environment
    * `(Virtual_Env)$ pip install jupyter`, install jupyter within the active virtualenv
    * `(Virtual_Env)$ pip install ipykernel`, install reference ipykernel package
    * `(Virtual_Env)$ python -m ipykernel install --user --name testenv --display-name "Python2 (Virtual_Env)"`, set up the kernel
    * `(Virtual_Env)$ jupyter notebook`
    * After jupyter notebook has been turned on, when you are creating a new notebook, choose "Python 2 (Virtual_Env)"
  * Pros and Cons of Keras
    * Simple and no detailed implemention of NN like lower level libraries (e.g. Tensorflow) required, but also because of this, it can be less flexible
    * Only support GPU Nvidia
    
* CNN for visual recognition: http://cs231n.github.io/neural-networks-3/
    
* Image recognition with Keras: https://www.analyticsvidhya.com/blog/2017/06/architecture-of-convolutional-neural-networks-simplified-demystified/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
    
* Use pre-trained model for digits recognition: https://www.analyticsvidhya.com/blog/2017/06/transfer-learning-the-art-of-fine-tuning-a-pre-trained-model/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29

* NN Implementation
  * Implement in Python and R: https://www.analyticsvidhya.com/blog/2017/05/neural-network-from-scratch-in-python-and-r/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
  * My NN code in python: https://github.com/hanhanwu/Hanhan-Machine-Learning-Model-Implementation/blob/master/neural_network.py
  * My NN code in real time short context search: https://github.com/hanhanwu/Hanhan_NLP/blob/master/short_context_search.py


*****************************************************************

EXPERIMENTS

* Digital Recognition with Keras
  * Adam Optimizer: https://arxiv.org/abs/1412.6980
  * Supported Optimizers in Keras: https://keras.io/optimizers/
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
  * Reference: https://www.analyticsvidhya.com/blog/2016/10/tutorial-optimizing-neural-networks-using-keras-with-image-recognition-case-study/
    
 
