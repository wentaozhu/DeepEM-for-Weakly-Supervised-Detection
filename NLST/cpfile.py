import numpy as np 
import shutil 
import pandas as pd 
import os 
import os.path
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
nCTs = 0
for pid in os.listdir(filepath):
    if nCTs % 500 == 0:
        print pid
    if pid not in uiddict:
        continue
    for studyid in os.listdir(filepath+pid+'/'):
        for seriesid in os.listdir(filepath+pid+'/'+studyid):
            hasseriesid = False
            for sidlist in uiddict[pid]:
                if studyid == sidlist[0] and seriesid == sidlist[1]:
                    hasseriesid = True
                    if not os.path.exists(dstpath+pid):
                        os.mkdir(dstpath+pid)
                    if not os.path.exists(dstpath+pid+'/'+studyid):
                        os.mkdir(dstpath+pid+'/'+studyid)
                    if not os.path.exists(dstpath+pid+'/'+studyid+'/'+seriesid):
                        os.mkdir(dstpath+pid+'/'+studyid+'/'+seriesid)
                    for fname in os.listdir(filepath+pid+'/'+studyid+'/'+seriesid):
                        shutil.copy(filepath+pid+'/'+studyid+'/'+seriesid+'/'+fname, dstpath+pid+'/'+studyid+'/'+seriesid+'/'+fname)
                    nCTs += 1
print nCTs