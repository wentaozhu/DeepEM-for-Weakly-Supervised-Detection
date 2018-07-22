import numpy as np 
import shutil 
import pandas as pd 
import os 
import os.path
import zipfile
from multiprocessing import Pool
filepath = '/mnt/wentao/NLST/NLST/'
dstpath = '/mnt/media/wentao/NLST/'
sctimageinfofname = '/mnt/wentao/NLST/package-nlst-304-2017.07.05/CT Image Info CSV/sctimageinfo.csv'
sctimageinfocolname = ['softwareversion', 'visit', 'seriesdescription', 'imagetype', 'kvp_raw', \
    'mas_raw', 'effmas_raw', 'pitch_raw', 'tablerotation_raw', 'reconthickness_raw', 'reconinterval_raw', \
    'reconfilter', 'reconstruction_diameter_raw', 'manufacturer_raw', 'manufacturers_model_name', 'scannercode', \
    'seriesinstanceuids', 'studyuid', 'pid', 'study_yr', 'numberimages', 'imagesize_kilobytes', 'imageclass', \
    'reconinterval', 'reconstruction_diameter', 'reconthickness	manufacturer', 'effmas', 'kvp', 'mas', 'pitch', \
    'tablerotation', 'dataset_version']
sctiminfoframe = pd.read_csv(sctimageinfofname, names=sctimageinfocolname, dtype=str)
# print sctiminfoframe.softwareversion.tolist()[1], sctiminfoframe.visit.tolist()[1]
uiddict = {}
pidlist = sctiminfoframe.studyuid.tolist()[1:] # pid
studyidlist = sctiminfoframe.seriesinstanceuids.tolist()[1:] # studyuid
seriesinstlist = sctiminfoframe.scannercode.tolist()[1:] # seriesinstanceuids
reconfilterlist = sctiminfoframe.reconinterval_raw.tolist()[1:] # reconfilter
imagetypelist = sctiminfoframe.seriesdescription.tolist()[1:] # imagetype

for idx in xrange(len(pidlist)):
    if idx % 1000 == 0:
        print idx, imagetypelist[idx], reconfilterlist[idx], pidlist[idx], studyidlist[idx], seriesinstlist[idx]
    if 'local' in imagetypelist[idx].lower():
        continue
    if 'lung' not in  reconfilterlist[idx].lower():
        continue
    if pidlist[idx] not in uiddict:
        uiddict[pidlist[idx]] = [[studyidlist[idx], seriesinstlist[idx]]]
    else:
        uiddict[pidlist[idx]].append([studyidlist[idx], seriesinstlist[idx]])
# ncp = 0
def cpfunc(pid):
	if pid.endswith('.zip') and pid[:-4] in uiddict:
		print 'cp', pid
		# ncp += 1
		# if ncp % 100 == 0:
			# print ncp, pid
		shutil.copy(filepath+pid, dstpath+pid)
def delfunc(pid):
	if pid.endswith('.zip') and pid[:-4] not in uiddict:
		print 'del', pid
		os.remove(dstpath+pid)
		# ndel += 1
def unzipfunc(pid):
	if pid.endswith('.zip'):
		zip_ref = zipfile.ZipFile(dstpath+pid, 'r')
		zip_ref.extractall(dstpath+pid[:-4]+'/')
		zip_ref.close()
		os.remove(dstpath+pid)

p = Pool(15)
# p.map(cpfunc, os.listdir(filepath))
# p.map(delfunc, os.listdir(dstpath))
p.map(unzipfunc, os.listdir(dstpath))
p.close()

# for pid in os.listdir(filepath):
# 	if pid.endswith('.zip') and pid[:-4] in uiddict:
# 		ncp += 1
# 		if ncp % 100 == 0:
# 			print ncp, pid
# 		shutil.copy(filepath+pid, dstpath+pid)

# ndel = 0
# for pid in os.listdir(dstpath):
# 	if pid.endswith('.zip') and pid[:-4] not in uiddict:
# 		shutil.rmtree(dstpath+pid)
# 		ndel += 1
# print '#del', ndel
