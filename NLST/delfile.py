import os
import shutil
import zipfile
filepath = './NLST/'
zipcount = 0
count = 0
for pid in os.listdir(filepath):
	if pid.endswith('.zip'):
		zip_ref = zipfile.ZipFile(filepath+pid, 'r')
		if not os.path.exists(filepath+pid[:-4]):
			zip_ref.extractall(filepath+pid[:-4])
		zipcount += 1
		print(filepath+pid)
		zip_ref.close()
		os.remove(filepath+pid)
print('del and unzip', zipcount)
for pid in os.listdir(filepath):
	for psid in os.listdir(os.path.join(*[filepath, pid])):
		for sid in os.listdir(os.path.join(*[filepath, pid, psid])):
			if os.path.isdir(os.path.join(*[filepath, pid, psid, sid])):
				if len(os.listdir(os.path.join(*[filepath, pid, psid, sid]))) <= 10:
					count += 1
					shutil.rmtree(os.path.join(*[filepath, pid, psid, sid]))
					print('rm '+filepath+'/'+pid+'/'+psid+'/'+sid)
					if len(os.listdir(os.path.join(*[filepath, pid, psid]))) == 0: 
						shutil.rmtree(os.path.join(*[filepath, pid, psid]))
						print('rm '+filepath+'/'+pid+'/'+psid)
						if len(os.listdir(os.path.join(*[filepath, pid]))) == 0:
							shutil.rmtree(os.path.join(*[filepath, pid]))
							print('rm '+filepath+'/'+pid)
print('del', count)
# del slice > 2.5, numberimages < 10, 'localizer', 'top' in imagetype
delfoldlst = []
delcount = 0
names = ['softwareversion', 'visit', 'seriesdescription', 'imagetype', 'kvp_raw', 'mas_raw', 'effmas_raw', \
    'pitch_raw', 'tablerotation_raw', 'reconthickness_raw','reconinterval_raw', 'reconfilter', 'reconstruction_diameter_raw', \
    'manufacturer_raw', 'manufacturers_model_name', 'scannercode', 'seriesinstanceuids', 'studyuid', 'pid', 'study_yr', \
    'numberimages', 'imagesize_kilobytes', 'imageclass', 'reconinterval', 'reconstruction_diameter', 'reconthickness', \
    'manufacturer', 'effmas', 'kvp', 'mas', 'pitch', 'tablerotation', 'dataset_version']
import pandas as pd 
pdfrm = pd.read_csv('./package-nlst-304-2017.07.05/CT Image Info CSV/sctimageinfo.csv', names=names)
pidlst = pdfrm['pid'].tolist()[1:]
stdidlst = pdfrm['studyuid'].tolist()[1:]
srsidlst = pdfrm['seriesinstanceuids'].tolist()[1:]
thclst = pdfrm['reconthickness'].tolist()[1:]
nimlst = pdfrm['numberimages'].tolist()[1:]
imgtyp = pdfrm['imagetype'].tolist()[1:]
stdyrlst0 = pdfrm['study_yr'].tolist()[1:]
pidstdyr2stdiddct = {}
for idx, thc in enumerate(thclst):
	if str(pidlst[idx])+str(stdyrlst0[idx]) not in pidstdyr2stdiddct:
		pidstdyr2stdiddct[str(pidlst[idx])+str(stdyrlst0[idx])] = stdidlst[idx]
	else:
		if pidstdyr2stdiddct[str(pidlst[idx])+str(stdyrlst0[idx])] != stdidlst[idx]:
			print(pidlst[idx], pidstdyr2stdiddct[str(pidlst[idx])+str(stdyrlst0[idx])], stdidlst[idx])
			pidstdyr2stdiddct.pop(str(pidlst[idx])+str(stdyrlst0[idx]), None)
			# assert(1==0)
	if float(thc) > 2.5 or nimlst[idx] <= 10 or 'localizer' in imgtyp[idx] or 'top' in imgtyp[idx]:
		# if type(pidlst[idx]) is int or stdidlst[idx] is int or srsidlst[idx] is int:
			# print(pidlst[idx], stdidlst[idx], srsidlst[idx])
		if os.path.isdir('./NLST/'+str(pidlst[idx])+'/'+stdidlst[idx]+'/'+srsidlst[idx]):
			print('del '+'./NLST/'+str(pidlst[idx])+'/'+stdidlst[idx]+'/'+srsidlst[idx])
			shutil.rmtree('./NLST/'+str(pidlst[idx])+'/'+stdidlst[idx]+'/'+srsidlst[idx])
			if len(os.listdir('./NLST/'+str(pidlst[idx])+'/'+stdidlst[idx])) == 0:
				shutil.rmtree('./NLST/'+str(pidlst[idx])+'/'+stdidlst[idx])
				print('rm '+'./NLST/'+str(pidlst[idx])+'/'+stdidlst[idx])
				if len(os.listdir('./NLST/'+str(pidlst[idx]))) == 0:
					shutil.rmtree('./NLST/'+str(pidlst[idx]))
					print('rm '+'./NLST/'+str(pidlst[idx]))
			delcount += 1
			# delfoldlst.append('./NLST/'+pidlst[i]+'/'+stdidlst[i]+'/'+srsidlst[i])
print('del > 2.5', delcount)
# del sct_epi_loc not in [1,2,3,4,5,6]
names = ['STUDY_YR', 'SCT_AB_DESC', 'SCT_PRE_ATT', 'SCT_EPI_LOC', 'SCT_LONG_DIA', 'SCT_PERP_DIA', 'SCT_MARGINS', \
    'sct_slice_num', 'SCT_AB_NUM', 'sct_found_after_comp', 'pid', 'dataset_version']
pdfrm = pd.read_csv('./package-nlst-304-2017.07.05/Spiral CT Abnormalities CSV/sctabn.csv', names=names)
stdyrlst = pdfrm['STUDY_YR'].tolist()[1:]
loclst = pdfrm['SCT_EPI_LOC'].tolist()[1:]
lngdimlst = pdfrm['SCT_LONG_DIA'].tolist()[1:]
pidlst1 = pdfrm['pid'].tolist()[1:]
slclst = pdfrm['sct_slice_num'].tolist()[1:]
kplst = []
kpstdidlst = set([])
kpdct = {}
for idx, loc in enumerate(loclst):
	if str(pidlst1[idx])+str(stdyrlst[idx]) not in pidstdyr2stdiddct:
		# print(pidlst1[idx], stdyrlst[idx])
		continue
	if loc in [1,2,3,4,5,6, '1', '2', '3', '4', '5', '6'] and float(lngdimlst[idx]) >= 3:
		kplst.append([pidlst1[idx], stdyrlst[idx], pidstdyr2stdiddct[str(pidlst1[idx])+str(stdyrlst[idx])], loc, \
			slclst[idx]])
		kpstdidlst.add(str(pidlst1[idx])+'/'+pidstdyr2stdiddct[str(pidlst1[idx])+str(stdyrlst[idx])])
		if str(pidlst1[idx])+'/'+pidstdyr2stdiddct[str(pidlst1[idx])+str(stdyrlst[idx])] in kpdct:
			kpdct[str(pidlst1[idx])+'/'+pidstdyr2stdiddct[str(pidlst1[idx])+str(stdyrlst[idx])]].append([loc, slclst[idx]])
		else:
			kpdct[str(pidlst1[idx])+'/'+pidstdyr2stdiddct[str(pidlst1[idx])+str(stdyrlst[idx])]] = [[loc, slclst[idx]]]
print('#kp', len(kplst), len(list(kpstdidlst)), len(kpdct.keys()))
kpstdidlst = list(kpstdidlst)
print(len(kpstdidlst))
ndel = 0
kplst = []
import csv
fid = open('weaklabel.csv', 'w')
writer = csv.writer(fid)
for pid in os.listdir('./NLST/'):
	if len(os.listdir('./NLST/'+pid)) == 0:
		print('rm '+'./NLST/'+pid)
		shutil.rmtree('./NLST/'+pid)
	for stdid in os.listdir('./NLST/'+pid):
		if str(pid) + '/' + stdid in kpstdidlst:
			kplst.append(str(pid) + '/' + stdid)
			for v in kpdct[str(pid) + '/' + stdid]:
				writer.writerow([str(pid) + '/' + stdid]+ v)
		if str(pid) + '/' + stdid not in kpstdidlst:
			print(str(pid)+ '/'+stdid)
			ndel += 1
			print('rm '+'./NLST/'+pid+'/'+stdid)
			shutil.rmtree('./NLST/'+pid+'/'+stdid)
			if len(os.listdir('./NLST/'+pid)) == 0:
				1 == 1
				print('rm '+'./NLST/'+pid)
				shutil.rmtree('./NLST/'+pid)
print('del ', ndel, kpstdidlst[0])
print('remain ', len(kplst))
nkp = 0
for pid in os.listdir('./NLST/'):
	if len(os.listdir('./NLST/'+pid)) == 0:
		print('after rm', './NLST/'+pid)
	nkp += len(os.listdir('./NLST/'+pid))
print('left', nkp)
fid.close()