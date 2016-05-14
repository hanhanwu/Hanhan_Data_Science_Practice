import sys
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithSGD, SVMWithSGD, LogisticRegressionWithLBFGS
import csv

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
        qt = int(nums[0])
        fs = nums[1:]
        s = len(fs)
        index_lst = [idx for idx in range(s) if fs[idx] != 0]
        fs_lst = [n for n in fs if n != 0]
        return qt, SparseVector(s, index_lst, fs_lst)

    except:
        print "Unexpected error:", sys.exc_info()[1]
        return None



def main():
    training_rdd = sc.textFile(train_inputs).map(to_LP_training).filter(lambda lp: lp!=None)
    testing_rdd = sc.textFile(test_inputs).map(to_LP_testing).filter(lambda lp: lp!=None)

    # # Logistic Regression with SGD
    # lg_model = LogisticRegressionWithSGD.train(training_rdd, step = 0.1, regType = 'l1')
    # lg_prediction = testing_rdd.map(lambda (qt, sv): (qt, lg_model.predict(sv)))
    #
    #
    # Logistic Regression with LBFGS
    lg_model2 = LogisticRegressionWithLBFGS.train(training_rdd)
    lg_prediction2 = testing_rdd.map(lambda (qt, sv): (qt, lg_model2.predict(sv)))
    #
    #
    # # SVM with SGD
    # svm_model = SVMWithSGD.train(training_rdd, step = 0.01)
    # svm_prediction = testing_rdd.map(lambda (qt, sv): (qt, svm_model.predict(sv)))


    # print 'Logistic Regression with SGD results: ', len(lg_prediction.filter(lambda (idx, p):p!=0).collect())
    result = lg_prediction2.collect()
    # print 'SVM with SGD', len(svm_prediction.filter(lambda (idx, p):p!=0).collect())

    with open('[your result.csv path]', 'w') as csvfile:
        fieldnames = ['QuoteNumber', 'QuoteConversion_Flag']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for l in result:
            writer.writerow({'QuoteNumber': l[0], 'QuoteConversion_Flag': l[1]})

if __name__ == "__main__":
    main()
