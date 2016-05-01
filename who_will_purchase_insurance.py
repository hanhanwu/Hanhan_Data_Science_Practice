__author__ = 'hanhanwu'

import sys
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithSGD, SVMWithSGD

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

    svm_model = SVMWithSGD.train(training_rdd, iterations=10)
    svm_prediction = testing_rdd.map(lambda (sv, idx): (svm_model.predict(sv), idx))
    print svm_prediction.take(50)

if __name__ == "__main__":
    main()
