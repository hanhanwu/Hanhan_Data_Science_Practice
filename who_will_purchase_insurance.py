__author__ = 'hanhanwu'

import sys
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithSGD, SVMWithSGD, LogisticRegressionWithLBFGS

conf = SparkConf().setAppName("who will buy insurance")
sc = SparkContext(conf=conf)
assert sc.version >= '1.5.1'
sqlContext = SQLContext(sc)

train_inputs = sys.argv[1]
test_inputs = sys.argv[2]


def to_LP_training(line):

    try:
        nums = [float(float(n)) for n in line.split(",")]
        s = len(nums)-1
        index_lst = [idx for idx in range(s) if nums[idx] != 0]
        num_lst = [n for n in nums[0:-1] if n != 0]
        return LabeledPoint(nums[-1], SparseVector(s, index_lst, num_lst))

    except:
        print "Unexpected error:", sys.exc_info()[1]
        return None


def to_LP_testing(line):

    try:
        nums = [float(float(n)) for n in line.split(",")]
        s = len(nums)
        index_lst = [idx for idx in range(s) if nums[idx] != 0]
        num_lst = [n for n in nums if n != 0]
        return SparseVector(s, index_lst, num_lst)

    except:
        print "Unexpected error:", sys.exc_info()[1]
        return None



def main():
    training_rdd = sc.textFile(train_inputs).map(to_LP_training).filter(lambda lp: lp!=None)
    testing_rdd = sc.textFile(test_inputs).map(to_LP_testing).filter(lambda lp: lp!=None).zipWithIndex()

    # Logistic Regression with SGD
    lg_model = LogisticRegressionWithSGD.train(training_rdd, step = 0.1, regType = 'l1')
    lg_prediction = testing_rdd.map(lambda (fs, idx): (lg_model.predict(fs), idx))


    # Logistic Regression with LBFGS
    lg_model2 = LogisticRegressionWithLBFGS.train(training_rdd)
    lg_prediction2 = testing_rdd.map(lambda (fs, idx): (lg_model2.predict(fs), idx))


    # SVM with SGD
    svm_model = SVMWithSGD.train(training_rdd, step = 0.01)
    svm_prediction = testing_rdd.map(lambda (fs, idx): (svm_model.predict(fs), idx))


    print 'Logistic Regression with SGD results: ', len(lg_prediction.filter(lambda (p,idx):p!=0).collect())
    print 'Logistic Regression with LBFGS results: ', len(lg_prediction2.filter(lambda (p,idx):p!=0).collect())
    print 'SVM with SGD results: ', len(svm_prediction.filter(lambda (p,idx):p!=0).collect())

if __name__ == "__main__":
    main()
