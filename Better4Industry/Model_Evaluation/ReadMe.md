# Model Evaluation In Industry
Note: The methods used here are using in the industry, but it doesn't mean they are ideal.


## Partial AUC
* Sometimes, instead of simply telling customers AUC (they won't understand), you need to tell them TPR and FPR. To compre your models, you may compare the TPR at specified FPR. Meanwhile you can calculate partial AUC, which is the AUC under the vertical line of the TPR for that specified FPR.
* My sample code for partical AUC: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Better4Industry/Model_Evaluation/about_partial_AUC.ipynb
