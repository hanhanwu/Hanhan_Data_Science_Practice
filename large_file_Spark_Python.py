from pyspark import SparkConf, SparkContext, Row
from pyspark.sql import SQLContext
import sys
import re

inputs = sys.argv[1]
# output = sys.argv[2]

conf = SparkConf().setAppName("larger data regex")
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)


def extract_pattern(l):
    ptn1 = ".*?\d+0000000000000000(.*?)\s+(\d{4})\.\s+0\..*?"
    ptn2 = ".*?\s+0000000000000000(.*?)\s+(\d{4})\.\s+0\..*?"
    ptn3 = ".*?\d+0000000000000000(.*?)0\..*?"
    ptn4 = ".*?\s+0000000000000000(.*?)0\..*?"

    m1 = re.search(ptn1, l)
    m2 = re.search(ptn2, l)
    m3 = re.search(ptn3, l)
    m4 = re.search(ptn4, l)

    if m1 != None:
        return Row(Merchant=m1.group(1), Category=m1.group(2), Label='merchant')
    elif m2 != None:
        return Row(Merchant=m2.group(1), Category=m2.group(2), Label='merchant')
    elif m3 != None:
        return Row(Merchant=m3.group(1), Category=None, Label='ept')
    elif m4 != None:
        return Row(Merchant=m4.group(1), Category=None, Label='ept')
    else:
        Row(Merchant=l, Category=None, Label='error')



def main():
    text = sc.textFile(inputs)

    extracted_pattern_df = text.map(extract_pattern).toDF().cache()
    merchant_df = extracted_pattern_df.where(extracted_pattern_df.Label == 'merchant')
    ept_df = extracted_pattern_df.where(extracted_pattern_df.Label == 'ept')
    error_df = extracted_pattern_df.where(extracted_pattern_df.Label == 'error')
    error_df.show()


if __name__ == "__main__":
    main()
