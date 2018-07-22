import numpy as np
import pandas as pd
import csv
import os
prep_folder = './preprocessnp/'
pdfrm = pd.read_csv('newweaklabel.csv', names=['fname', 'position', 'centerslice'])
fnmlst = pdfrm['fname'].tolist()[1:]
poslst = pdfrm['position'].tolist()[1:]
ctrlst = pdfrm['centerslice'].tolist()[1:]
fid = open('calibweaklabel.csv', 'w')
csvwriter = csv.writer(fid)
csvwriter.writerow(['fname', 'position', 'centerslice'])
badcount = 0
keepcount = 0
for idx, fnm in enumerate(fnmlst):
	data = np.load(os.path.join(prep_folder, fnm+'_clean.npy'))
	extendbox = np.load(os.path.join(prep_folder, fnm+'_originbox.npy'))
	spacing = np.load(os.path.join(prep_folder, fnm+'_spacing.npy'))
	# print(spacing, np.expand_dims(extendbox[:,0],1), spacing.shape, np.expand_dims(extendbox[:,0],1).shape, \
		# np.expand_dims(extendbox[:,0],1)[0,0])
	label = float(ctrlst[idx])*spacing[0]
	label = label -float(np.expand_dims(extendbox[:,0],1)[0,0])
	if idx%5000 == 0:
		print('process', idx)
	if label < 0 or label > data.shape[1]:
		badcount += 1
		# print(fnm, label, data.shape)
		continue
	csvwriter.writerow([fnm, poslst[idx], label])
	keepcount += 1
fid.close()
print(badcount, keepcount)