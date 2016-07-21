'''
Created on Jul 20, 2016
'''

import re


def main():
    f_path1 = "[large file path]"

    f1 = open(f_path1)


    ct = 0
    ptn1 = ".*?\d+0000000000000000(.*?)\s+(\d{4}\.)\s+0\..*?"
    ptn2 = ".*?\s+0000000000000000(.*?)\s+(\d{4}\.)\s+0\..*?"
    ptn3 = ".*?\d+0000000000000000(.*?)0\..*?"
    ptn4 = ".*?\s+0000000000000000(.*?)0\..*?"
    all_merchant = []
    epts = []


    for l in f1:
        ct += 1
        if ct == 5000: break
        m1 = re.search(ptn1, l)
        m2 = re.search(ptn2, l)
        m3 = re.search(ptn3, l)
        m4 = re.search(ptn4, l)
        
        if m1 != None:
            all_merchant.append(m1.group(1))
            print len(all_merchant)
            continue
        elif m2 != None:
            all_merchant.append(m2.group(1))
            print len(all_merchant)
            continue
        elif m3 != None:
            epts.append(m3.group(1))
            continue
        elif m4 != None:
            epts.append(m4.group(1))
            continue
        else:
            print l
            break
        
        
    print len(all_merchant)
    print
    for itm in all_merchant:
        print itm
    
    print "************epts*************"
    print len(epts)
    print
    for ept in epts:
        print ept

if __name__ == "__main__":
    main()
    
