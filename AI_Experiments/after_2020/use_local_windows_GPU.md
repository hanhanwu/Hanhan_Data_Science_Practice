# How to Use Windows Local GPU

## Tensorflow
* Whenever Tensorflow updates, there's always some old code could not use any more, and this happened to me almost everytime when I am upgrading tensorflow...😓😓 Dependent packages also updated in different rhythms... Meanwhile, many online tutorials either got out of dated quickly or being incomplete or even incorrect. So here I have create my own document to track the process and update the changes. Good luck to me and the audience of this page 🍀🍀

### Software Versions
* `tensorflow-gpu==2.4.1`
  * Make sure `tensorflow` is uninstalled and `tensorflow-gpu` is installed through `pip3 install --upgrade tensorflow-gpu`
* `cuda_11.2`
  * Download from https://developer.nvidia.com/cuda-downloads
  * Check `nvcc --version` through terminal to find the version
* Download [cudnn][1] that matches your cuda version
  * This is an accelerator which will allow you to use `cuDNN` related algorithms, people say is much faster for LSTM and GRU
  * I'm using `cudnn-11.1`, which doesn't match the latest cuda_11.2... But it seems that this package updates frequently, so I might upgrade later when it got a matched version
  * Had to rename `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin\cusolver64_11.dll` to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin\cusolver64_10.dll`
  * I had to copy all the files in "Downloads\cudnn-11.1-windows-x64-v8.0.5.39\cuda\bin" to "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin", otherwise later when you are using `cuDNN` related algorithms in tensorflow, python program will be killed
  
### Install Process
* I was mainly follow [this tutorial for Windows 10][2]
  * To check whether your Windows 10 already has satisfied NVIDIA GPU, [check this page][3]
  * Check above settings notes, since this tutorial is not enough
  * To check whether GPU is really available, [check my experiment here][4]
  
### Local GPU vs CPU Experiment
* [Check my experiment here][4]
  * GPU is not always better than CPU. In this example, it's far slower than CPU
    * Check Windows "Task Manager" --> "Performance", you will be able to monitor the utilization percentage of CPU and GPU
    * In this case, the utilization of GPU is very small even after increased batch size
    * Because the overhead of invoking GPU kernels, and copying data to and from GPU, is very high. For operations on models with very little parameters it is not worth of using GPU
  * I have also tried `CuDNNLSTM` to replace lstm for GPU, since it's faster, but it's still underdeveopped
    * "dropout" and "activation" are not supported in CuDNNLSTM for now....😔😔
    * Somehow it's compalining the data didn't have initial state. I'm wondering whether it's caused the version match between my CUDA and cudnn
    

[1]:https://developer.nvidia.com/cudnn
[2]:https://towardsdatascience.com/installing-tensorflow-with-cuda-cudnn-and-gpu-support-on-windows-10-60693e46e781
[3]:https://www.analyticsvidhya.com/blog/2020/11/how-to-download-install-and-use-nvidia-gpu-for-tensorflow-on-windows/#:~:text=To%20download%2C%20Navigate%20to%20the,will%20provide%20the%20download%20link.&text=once%20installed%20we%20should%20get,drive%20containing%20CUDA%20subfolder%20inside.
[4]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AI_Experiments/after_2020/test_local_GPU.ipynb
