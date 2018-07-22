import csv
import os
fid = open('fnamedct.csv', 'w')
writer = csv.writer(fid)
writer.writerow(['fname', 'newname'])
dct = {}
curk = 0
path = '../luna16/subset'
for f in xrange(10):
	for fname in os.listdir(path+str(f)):
		if fname.endswith('raw'):
			dct[fname[:-4]] = '{:05d}'.format(curk)
			writer.writerow([fname[:-4], dct[fname[:-4]]])
			curk += 1
fid.close()
print('number of files', curk)
import pandas as pd 
# process seriesuid.csv
name = '1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860'
sdfrm = pd.read_csv('./evaluationScript/annotations/seriesuids.csv', \
	names=[name])
fnamelst = sdfrm[name].tolist()
print('number of series uid', len(fnamelst))
fid = open('./evaluationScript/annotations/newseriesuids.csv', 'w')
writer = csv.writer(fid)
for fname in fnamelst:
	writer.writerow([dct[fname]])
fid.close()
# process annotations_excluded.csv
name = ['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm']
antexd = pd.read_csv('./evaluationScript/annotations/annotations_excluded.csv', names=name)
srsidlst = antexd[name[0]].tolist()[1:]
crdxlst = antexd[name[1]].tolist()[1:]
crdylst = antexd[name[2]].tolist()[1:]
crdzlst = antexd[name[3]].tolist()[1:]
dimlst = antexd[name[4]].tolist()[1:]
fid = open('./evaluationScript/annotations/newannotations_excluded.csv', 'w')
writer = csv.writer(fid)
writer.writerow(name)
for i in xrange(len(srsidlst)):
	writer.writerow([dct[srsidlst[i]], crdxlst[i], crdylst[i], crdzlst[i], dimlst[i]])
fid.close()
# process annotations.csv
name = ['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm']
ant = pd.read_csv('./evaluationScript/annotations/annotations.csv', names=name)
srsidlst = ant[name[0]].tolist()[1:]
crdxlst = ant[name[1]].tolist()[1:]
crdylst = ant[name[2]].tolist()[1:]
crdzlst = ant[name[3]].tolist()[1:]
dimlst = ant[name[4]].tolist()[1:]
fid = open('./evaluationScript/annotations/newannotations.csv', 'w')
writer = csv.writer(fid)
writer.writerow(name)
for i in xrange(len(srsidlst)):
	writer.writerow([dct[srsidlst[i]], crdxlst[i], crdylst[i], crdzlst[i], dimlst[i]])
fid.close()
# process luna16 annotations.csv
name = ['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm']
ant = pd.read_csv('../luna16/CSVFILES/annotations.csv', names=name)
srsidlst = ant[name[0]].tolist()[1:]
crdxlst = ant[name[1]].tolist()[1:]
crdylst = ant[name[2]].tolist()[1:]
crdzlst = ant[name[3]].tolist()[1:]
dimlst = ant[name[4]].tolist()[1:]
fid = open('../luna16/CSVFILES/newannotations.csv', 'w')
writer = csv.writer(fid)
writer.writerow(name)
for i in xrange(len(srsidlst)):
	writer.writerow([dct[srsidlst[i]], crdxlst[i], crdylst[i], crdzlst[i], dimlst[i]])
fid.close()
# revise file name
# Do it in the prepare.py to save space.