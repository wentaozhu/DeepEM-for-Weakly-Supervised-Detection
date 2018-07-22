''' Wentao Zhu, wentaozhu1991@gmail.com
    mv negative folder into .../data/negNLST/ to reduce memory usage 
    file organization: .../data/filterctim.py; .../data/NLST/****/****; 
    .../data/package-nlst-304-2017.07.05/Spiral CT Abnormalities CSV/sctabn.csv '''
import csv 
import os 
import shutil

nlstpath = './data/NLST/'
csvpath = './package-nlst-304-2017.07.05/Spiral CT Abnormalities CSV/sctabn.csv'
qcsvpath = './package-nlst-304-2017.07.05/queryResult.csv' # query csv
savetxt = './package-nlst-304-2017.07.05/savepid.txt'
mvpath = './data/negNLST/'
f = open(csvpath)
csvf = csv.reader(f)
count = 0
poscount = 0 # number of patient having nodule
lastpid = 0 # record last pid to facilitate poscount
pospidlist = set([]) # record positive pid list
negcount = 0 # subsample 1/3 of CTs having no nodule
negpidlist = set([]) # set for fast remove repulicate items
pididx = 0 # record which column the pid in
lepidx = 0 # record which column the SCT_EPI_LOC in
for row in csvf:
    if count == 0:
        count += 1
        for col in row:
            if col == 'pid':
                break
            pididx += 1
        for col in row:
            if col == 'SCT_EPI_LOC':
                break
            lepidx += 1
        continue
    if row[pididx] != lastpid and row[lepidx] != '':
        poscount += 1
        pospidlist.add(row[pididx])
    elif row[pididx] != lastpid:
        negcount += 1
        if negcount % 6 == 0:
            negpidlist.add(row[pididx])
    lastpid = row[pididx]
f.close()
assert poscount == len(pospidlist)
# assert(negcount == len(negpidlist))
print '# pid', poscount+len(negpidlist)
print '# positive pid', poscount
print '# negative pid', len(negpidlist)

posnum = 0
negnum = 0
txtlist = set([]) # remove repulicated txt pid
txtf = open(savetxt, 'w')
f = open(qcsvpath)
qcsvf = csv.reader(f)
for row in qcsvf:
    if row[0] in pospidlist:
        if row[0] not in txtlist:
            txtf.write(row[0]+'\n')
            txtlist.add(row[0])
            posnum += 1
    elif row[0] in negpidlist:
        if row[0] not in txtlist:
            txtf.write(row[0]+'\n')
            txtlist.add(row[0])
            negnum += 1
f.close()
txtf.close()
print '# query pid', posnum+negnum
print '# positive pid', posnum
print '# negative pid', negnum

rmnum = 0
leftnum = 0
for pid in os.listdir(nlstpath):
    if pid not in negpidlist and pid not in pospidlist:
        print 'mv', pid
        shutil.move(nlstpath+pid, mvpath)
        # shutil.rmtree(nlstpath+pid+'/')
        rmnum += 1
    else:
        leftnum += 1
print 'mv', rmnum
print 'left', leftnum
