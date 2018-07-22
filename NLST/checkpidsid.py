''' check pid and study id is 1 to n'''
import os
import pandas as pd 
pdfrm = pd.read_csv('weaklabel.csv', names=['103303/1.2.840.113654.2.55.35659264966224867891971832130700032142', '1', '9'])
pidsid = pdfrm['103303/1.2.840.113654.2.55.35659264966224867891971832130700032142'].tolist()
position = pdfrm['1'].tolist()
centerslice = pdfrm['9'].tolist()
mydct = {}
datadct = {}
for idx, ps in enumerate(pidsid):
	pid, sid = ps.split('/')
	if sid in mydct:
		if mydct[sid] != pid:
			print(pid, mydct[sid])
		datadct[sid].append([position[idx], centerslice[idx]])
	else:
		datadct[sid] = [[position[idx], centerslice[idx]]]
		mydct[sid] = pid 
print(len(mydct.keys())) # 11006
''' we use study id as the file name'''
# save the dictionary
siddct = {}
import csv
fid = open('studyiddict.csv', 'w')
csvwriter = csv.writer(fid)
csvwriter.writerow(['studyid', 'newfname'])
for idx, sid in enumerate(mydct.keys()):
	siddct[sid] = idx
	csvwriter.writerow([sid, '{num:05d}'.format(num=idx+1)])
fid.close()
# get the numpy data
npdct = {}
maxsubsrs = 0
for pid in os.listdir('./preprocess/'):
	for studyid in os.listdir('./preprocess/'+pid+'/'):
		isnp = False
		nsubsrs = 0
		for srs in os.listdir('./preprocess/'+pid+'/'+studyid+'/'):
			if srs.endswith('_clean.npy'):
				isnp = True
				nsubsrs += 1
		if isnp:
			if studyid in npdct and npdct[studyid] != pid:
				print(pid, npdct[studyid], studyid)
			npdct[studyid] = pid 
			maxsubsrs = max(maxsubsrs, nsubsrs)
print(len(npdct.keys()), maxsubsrs)
# copy to preprocessnp and change file name 
import shutil
from shutil import copyfile
pidsidsrsdct = {}
newdatadct = {}
for studyid in npdct.keys():
	if studyid not in siddct: continue
	# fpath = siddct[studyid]+'/'+studyid+'/'
	subsrsid = 1
	for srs in os.listdir('./preprocess/'+npdct[studyid]+'/'+studyid+'/'):
		if srs.endswith('_clean.npy'):
			pidsidsrsdct['./preprocess/'+npdct[studyid]+'/'+studyid+'/'+srs[:-9]] = \
			    '{num:05d}'.format(num=siddct[studyid])+'_'+str(subsrsid)
			newdatadct['{num:05d}'.format(num=siddct[studyid])+'_'+str(subsrsid)] = datadct[studyid]
			# copyfile('./preprocess/'+npdct[studyid]+'/'+studyid+'/'+srs, \
			# 	'./preprocessnp/'+'{num:05d}'.format(num=siddct[studyid])+'_'+str(subsrsid)+'_clean.npy')
			# copyfile('./preprocess/'+npdct[studyid]+'/'+studyid+'/'+srs[:-9]+'originbox.npy', \
			# 	'./preprocessnp/'+'{num:05d}'.format(num=siddct[studyid])+'_'+str(subsrsid)+'_originbox.npy')
			# copyfile('./preprocess/'+npdct[studyid]+'/'+studyid+'/'+srs[:-9]+'spacing.npy', \
			# 	'./preprocessnp/'+'{num:05d}'.format(num=siddct[studyid])+'_'+str(subsrsid)+'_spacing.npy')
			subsrsid += 1
fid = open('pidsidsrsdct.csv', 'w')
csvwriter = csv.writer(fid)
csvwriter.writerow(['oldpath', 'newfname'])
for k, v in pidsidsrsdct.iteritems():
	csvwriter.writerow([k, v])
fid.close()
fid = open('newweaklabel.csv', 'w')
csvwriter = csv.writer(fid)
csvwriter.writerow(['fname', 'position', 'centerslice'])
for k, v in newdatadct.iteritems():
	for vv in v:
		if len(vv) != 2: print(vv, k)
		csvwriter.writerow([k, vv[0], vv[1]])
fid.close()