import numpy as np
import torch
from torch.utils.data import Dataset
import os
import time
import collections
import random
from layers import iou
from scipy.ndimage import zoom
import warnings
from scipy.ndimage.interpolation import rotate
import pandas as pd
xydct = {}
pdfrm=pd.read_csv('../../NLST/calibweaklabelall.csv',names=['fname','position','centerslice','lowz','upz','dataz','datax','datay'])
fnmlst = pdfrm['fname'].tolist()[1:]
xlst = pdfrm['datax'].tolist()[1:]
ylst = pdfrm['datay'].tolist()[1:]
zlst = pdfrm['dataz'].tolist()[1:]
for idx, fnm in enumerate(fnmlst):
    if fnm not in xydct: xydct[fnm] = [float(zlst[idx]), float(xlst[idx]), float(ylst[idx])]
samplerate = 16
isprb = False #True #False
del fnmlst[:]
class DataBowl3Detector(Dataset):
    def __init__(self, data_dir, split_path, config, phase='train', split_comber=None,fnmlst=None, cdxlst=None, cdylst=None, \
        cdzlst=None, dimlst=None, prblst=None, poslst=None):
        assert(phase == 'train' or phase == 'val' or phase == 'test')
        self.phase = phase
        self.max_stride = config['max_stride']       
        self.stride = config['stride']       
        sizelim = config['sizelim']/config['reso']
        sizelim2 = config['sizelim2']/config['reso']
        sizelim3 = config['sizelim3']/config['reso']
        self.blacklist = config['blacklist']
        self.isScale = config['aug_scale']
        self.r_rand = config['r_rand_crop']
        self.augtype = config['augtype']
        self.pad_value = config['pad_value']
        self.split_comber = split_comber
        # self.blacklist += ['01624_1', '08491_1', '04847_1', '08646_1']
        self.config = config
        idcs = split_path # np.load(split_path)
        if phase!='test':
            idcs = [f for f in idcs if (f not in self.blacklist)]
            annodct = {}
            if not fnmlst:
                pdfrm = pd.read_csv(config['save_dir']+'weakinferep'+str(config['ep'])+'.csv', names=['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter', 'probability', 'position'])
                fnmlst = pdfrm['seriesuid'].tolist()[1:]
                cdxlst = pdfrm['coordX'].tolist()[1:]
                cdylst = pdfrm['coordY'].tolist()[1:]
                cdzlst = pdfrm['coordZ'].tolist()[1:]
                dimlst = pdfrm['diameter'].tolist()[1:]
                prblst = pdfrm['probability'].tolist()[1:]
                poslst = pdfrm['position'].tolist()[1:]
            for idx, fnm in enumerate(fnmlst):
            	value = [float(cdzlst[idx]),float(cdxlst[idx]),float(cdylst[idx]),float(dimlst[idx]),float(prblst[idx]),int(poslst[idx])]
                if fnm not in annodct: annodct[fnm]=[value]
                else: annodct[fnm].append(value)
        self.filenames = []
        for idx in idcs:
            if 'tianchi' in data_dir:
                self.filenames.append(os.path.join(data_dir, '%s_clean.npy' % idx.split('/')[-1]))
            else:
                self.filenames.append(os.path.join(data_dir, '%s_clean.npy' % idx))
        self.kagglenames = [f for f in self.filenames]# if len(f.split('/')[-1].split('_')[0])>20]
        labels = []
        labels2 = []
        feat = []
        # print len(idcs)
        filenamesnew = []
        for idx in idcs:
            if phase == 'test':
                l = np.array([])
                labels.append(l)
                continue
            if 'tianchi' in data_dir:
                l = np.load(data_dir+idx.split('/')[-1]+'_label.npy')
            else:
                l = np.asarray(annodct[idx])# np.load(data_dir+idx+'_label.npy')
                # print(l.shape)
            if np.all(l==0):
                l=np.array([])
            # print(type(l), l.shape)
            for lidx in range(l.shape[0]):
                if 'tianchi' in data_dir:
                    filenamesnew.append(os.path.join(data_dir, '%s_clean.npy' % idx.split('/')[-1]))
                else:
                    filenamesnew.append(os.path.join(data_dir, '%s_clean.npy' % idx))
                labels2.append(np.array(l[lidx,-2:]).reshape((1,2)))
                labels.append(np.array(l[lidx,:-2]).reshape((1,4)))
                feat.append(np.hstack([l[lidx,0]/xydct[idx.split('/')[-1]][0], l[lidx,1]/xydct[idx.split('/')[-1]][1], l[lidx,2]/xydct[idx.split('/')[-1]][2]]))
            # print(phase, labels2[-1].shape, labels[-1].shape, feat[-1].shape, l.shape)
            # labels2.append(np.array(l[:,-2:]))
            # labels.append(np.array(l[:,:-2]))
            # feat.append(np.hstack([l[:,0]/xydct[idx.split('/')[-1]][0], l[:,1]/xydct[idx.split('/')[-1]][1], l[:,2]/xydct[idx.split('/')[-1]][2]]))
            # print(np.hstack([l[:,1]/xydct[idx.split('/')[-1]][0], l[:,2]/xydct[idx.split('/')[-1]][0]]).shape)
        if phase != 'test': self.filenames = filenamesnew
        self.sample_bboxes = labels
        self.sample_prob = labels2
        self.sample_feat = feat
        if self.phase != 'test':
            self.bboxes = []
            for i, l in enumerate(labels):
                if len(l) > 0 :
                    for t in l:
                        if t[3]>sizelim:
                            self.bboxes.append([np.concatenate([[i],t])])
                        if t[3]>sizelim2:
                            self.bboxes+=[[np.concatenate([[i],t])]]*2
                        if t[3]>sizelim3:
                            self.bboxes+=[[np.concatenate([[i],t])]]*4
            self.bboxes = np.concatenate(self.bboxes,axis = 0)
        self.crop = Crop(config)
        self.label_mapping = LabelMapping(config, self.phase)
        if self.phase == 'train':
            print('#train', len(self.bboxes), len(self.filenames))#/(1-self.r_rand))
        elif self.phase =='val':
            print('#val', len(self.bboxes)/samplerate, len(self.filenames)/samplerate)#10)
        else:
            print('#test', len(self.sample_bboxes)/samplerate, len(self.filenames)/samplerate)#10)
    def __getitem__(self, idx,split=None):
        t = time.time()
        np.random.seed(int(str(t%1)[2:7]))#seed according to time
        isRandomImg  = False
        if self.phase !='test':
            if idx>=len(self.bboxes):
                isRandom = True
                idx = idx%len(self.bboxes)
                isRandomImg = np.random.randint(2)
            else:
                isRandom = False
        else:
            isRandom = False
        # if self.phase == 'train':
        #     idx = np.random.randint(10)*(len(self.bboxes)/(1-self.r_rand)/10)+idx
        if self.phase =='val':
            idx = np.random.randint(samplerate)*(len(self.bboxes)/samplerate)+idx
        elif self.phase == 'test':
            idx = np.random.randint(samplerate)*(len(self.sample_bboxes)/samplerate)+idx
        # print idx
        if self.phase != 'test':
            if not isRandomImg:
                bbox = self.bboxes[idx]
                filename = self.filenames[int(bbox[0])]
                imgs = np.load(filename)
                bboxes = self.sample_bboxes[int(bbox[0])]
                prob = self.sample_prob[int(bbox[0])]
                feat = self.sample_feat[int(bbox[0])]
                # print(bbox.shape, prob.shape, bboxes.shape, prob.shape)
                isScale = self.augtype['scale'] and (self.phase=='train')
                sample, target, bboxes, coord = self.crop(imgs, bbox[1:], bboxes, isScale,isRandom)
                if self.phase=='train' and not isRandom:
                     sample, target, bboxes, coord = augment(sample, target, bboxes, coord,
                        ifflip = self.augtype['flip'], ifrotate=self.augtype['rotate'], ifswap = self.augtype['swap'])
            else:
                randimid = np.random.randint(len(self.kagglenames))
                filename = self.kagglenames[randimid]
                imgs = np.load(filename)
                bboxes = self.sample_bboxes[randimid]
                prob = self.sample_prob[randimid]
                feat = self.sample_feat[randimid]
                isScale = self.augtype['scale'] and (self.phase=='train')
                sample, target, bboxes, coord = self.crop(imgs, [], bboxes,isScale=False,isRand=True)
                # print(bboxes.shape, prob.shape, prob.shape)
            # print sample.shape, target.shape, bboxes.shape
            # if sample.shape[1] != self.config['crop_size'][0] or sample.shape[2] != self.config['crop_size'][1] or \
            #     sample.shape[3] != self.config['crop_size'][2]:
            label = self.label_mapping(sample.shape[1:], target, bboxes, filename)
            # print(label.shape)
            sample = (sample.astype(np.float32)-128)/128
            #if filename in self.kagglenames and self.phase=='train':
            #    label[label==-1]=0
            # print(type(prob), prob.shape, prob, type(prob[0]), prob[0].shape, prob[0,0], prob[0,1])
            if torch.from_numpy(np.array([prob[0,0]])).size()[0] == 0:
                print sample.shape, label.shape, coord.shape, np.array(prob[0,0]).shape
                assert 1 == 0
            # print feat.shape, type(feat), sample.shape, label.shape, coord.shape, prob[0,0], prob[0,1]
            if feat.shape[0] == 6:
                print feat
                assert 1==0
            if not isprb:
                return torch.from_numpy(sample), torch.from_numpy(label), coord, torch.from_numpy(np.array([1])), \
                torch.from_numpy(np.array([prob[0,1]])).long(), torch.from_numpy(feat).float()
            return torch.from_numpy(sample), torch.from_numpy(label), coord, torch.from_numpy(np.array([prob[0,0]])), \
                torch.from_numpy(np.array([prob[0,1]])).long(), torch.from_numpy(feat).float() #torch.from_numpy(np.reshape(feat, (1,2))) # wrong
        else:
            imgs = np.load(self.filenames[idx])
            bboxes = self.sample_bboxes[idx]
            nz, nh, nw = imgs.shape[1:]
            pz = int(np.ceil(float(nz) / self.stride)) * self.stride
            ph = int(np.ceil(float(nh) / self.stride)) * self.stride
            pw = int(np.ceil(float(nw) / self.stride)) * self.stride
            imgs = np.pad(imgs, [[0,0],[0, pz - nz], [0, ph - nh], [0, pw - nw]], 'constant',constant_values = self.pad_value)
            
            xx,yy,zz = np.meshgrid(np.linspace(-0.5,0.5,imgs.shape[1]/self.stride),
                                   np.linspace(-0.5,0.5,imgs.shape[2]/self.stride),
                                   np.linspace(-0.5,0.5,imgs.shape[3]/self.stride),indexing ='ij')
            coord = np.concatenate([xx[np.newaxis,...], yy[np.newaxis,...],zz[np.newaxis,:]],0).astype('float32')
            imgs, nzhw = self.split_comber.split(imgs)
            coord2, nzhw2 = self.split_comber.split(coord,
                                                   side_len = self.split_comber.side_len/self.stride,
                                                   max_stride = self.split_comber.max_stride/self.stride,
                                                   margin = self.split_comber.margin/self.stride)
            assert np.all(nzhw==nzhw2)
            imgs = (imgs.astype(np.float32)-128)/128
            return torch.from_numpy(imgs), bboxes, torch.from_numpy(coord2), np.array(nzhw)
    def __len__(self):
        if self.phase == 'train':
            # print('#train', len(self.bboxes))#/(1-self.r_rand))
            return len(self.bboxes)#/(1-self.r_rand)#/10 # sample 1/6
        elif self.phase =='val':
            # print('#val', len(self.bboxes)/10)
            return len(self.bboxes)/samplerate#10
        else:
            # print('#test', len(self.sample_bboxes)/10)
            return len(self.sample_bboxes)/samplerate#10
        
def augment(sample, target, bboxes, coord, ifflip = True, ifrotate=True, ifswap = True):
    #                     angle1 = np.random.rand()*180
    if ifrotate:
        validrot = False
        counter = 0
        while not validrot:
            newtarget = np.copy(target)
            angle1 = np.random.rand()*180
            size = np.array(sample.shape[2:4]).astype('float')
            rotmat = np.array([[np.cos(angle1/180*np.pi),-np.sin(angle1/180*np.pi)],[np.sin(angle1/180*np.pi),np.cos(angle1/180*np.pi)]])
            newtarget[1:3] = np.dot(rotmat,target[1:3]-size/2)+size/2
            if np.all(newtarget[:3]>target[3]) and np.all(newtarget[:3]< np.array(sample.shape[1:4])-newtarget[3]):
                validrot = True
                target = newtarget
                sample = rotate(sample,angle1,axes=(2,3),reshape=False)
                coord = rotate(coord,angle1,axes=(2,3),reshape=False)
                for box in bboxes:
                    box[1:3] = np.dot(rotmat,box[1:3]-size/2)+size/2
            else:
                counter += 1
                if counter ==3:
                    break
    if ifswap:
        if sample.shape[1]==sample.shape[2] and sample.shape[1]==sample.shape[3]:
            axisorder = np.random.permutation(3)
            sample = np.transpose(sample,np.concatenate([[0],axisorder+1]))
            coord = np.transpose(coord,np.concatenate([[0],axisorder+1]))
            target[:3] = target[:3][axisorder]
            bboxes[:,:3] = bboxes[:,:3][:,axisorder]
            
    if ifflip:
#         flipid = np.array([np.random.randint(2),np.random.randint(2),np.random.randint(2)])*2-1
        flipid = np.array([1,np.random.randint(2),np.random.randint(2)])*2-1
        sample = np.ascontiguousarray(sample[:,::flipid[0],::flipid[1],::flipid[2]])
        coord = np.ascontiguousarray(coord[:,::flipid[0],::flipid[1],::flipid[2]])
        for ax in range(3):
            if flipid[ax]==-1:
                target[ax] = np.array(sample.shape[ax+1])-target[ax]
                bboxes[:,ax]= np.array(sample.shape[ax+1])-bboxes[:,ax]
    return sample, target, bboxes, coord 

class Crop(object):
    def __init__(self, config):
        self.crop_size = config['crop_size']
        self.bound_size = config['bound_size']
        self.stride = config['stride']
        self.pad_value = config['pad_value']
    def __call__(self, imgs, target, bboxes,isScale=False,isRand=False):
        if isScale:
            radiusLim = [8.,120.]
            scaleLim = [0.75,1.25]
            scaleRange = [np.min([np.max([(radiusLim[0]/target[3]),scaleLim[0]]),1])
                         ,np.max([np.min([(radiusLim[1]/target[3]),scaleLim[1]]),1])]
            scale = np.random.rand()*(scaleRange[1]-scaleRange[0])+scaleRange[0]
            crop_size = (np.array(self.crop_size).astype('float')/scale).astype('int')
        else:
            crop_size=self.crop_size
        bound_size = self.bound_size
        target = np.copy(target)
        bboxes = np.copy(bboxes)
        
        start = []
        for i in range(3):
            if not isRand:
                r = target[3] / 2
                s = np.floor(target[i] - r)+ 1 - bound_size
                e = np.ceil (target[i] + r)+ 1 + bound_size - crop_size[i] 
            else:
                s = np.max([imgs.shape[i+1]-crop_size[i]/2,imgs.shape[i+1]/2+bound_size])
                e = np.min([crop_size[i]/2,              imgs.shape[i+1]/2-bound_size])
                target = np.array([np.nan,np.nan,np.nan,np.nan])
            if s>e:
                start.append(np.random.randint(e,s))#!
            else:
                start.append(int(target[i])-crop_size[i]/2+np.random.randint(-bound_size/2,bound_size/2))
                
                
        normstart = np.array(start).astype('float32')/np.array(imgs.shape[1:])-0.5
        normsize = np.array(crop_size).astype('float32')/np.array(imgs.shape[1:])
        xx,yy,zz = np.meshgrid(np.linspace(normstart[0],normstart[0]+normsize[0],self.crop_size[0]/self.stride),
                           np.linspace(normstart[1],normstart[1]+normsize[1],self.crop_size[1]/self.stride),
                           np.linspace(normstart[2],normstart[2]+normsize[2],self.crop_size[2]/self.stride),indexing ='ij')
        coord = np.concatenate([xx[np.newaxis,...], yy[np.newaxis,...],zz[np.newaxis,:]],0).astype('float32')

        pad = []
        pad.append([0,0])
        for i in range(3):
            leftpad = max(0,-start[i])
            rightpad = max(0,start[i]+crop_size[i]-imgs.shape[i+1])
            pad.append([leftpad,rightpad])
        crop = imgs[:,
            max(start[0],0):min(start[0] + crop_size[0],imgs.shape[1]),
            max(start[1],0):min(start[1] + crop_size[1],imgs.shape[2]),
            max(start[2],0):min(start[2] + crop_size[2],imgs.shape[3])]
        crop = np.pad(crop,pad,'constant',constant_values =self.pad_value)
        for i in range(3):
            target[i] = target[i] - start[i] 
        for i in range(len(bboxes)):
            for j in range(3):
                bboxes[i][j] = bboxes[i][j] - start[j] 
                
        if isScale:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                crop = zoom(crop,[1,scale,scale,scale])#, mode='nearest') #order=1,
            newpad = self.crop_size[0]-crop.shape[1:][0]
            if newpad<0:
                crop = crop[:,:-newpad,:-newpad,:-newpad]
            elif newpad>0:
                pad2 = [[0,0],[0,newpad],[0,newpad],[0,newpad]]
                crop = np.pad(crop, pad2, 'constant', constant_values=self.pad_value)
            for i in range(4):
                target[i] = target[i]*scale
            for i in range(len(bboxes)):
                for j in range(4):
                    bboxes[i][j] = bboxes[i][j]*scale
        # print(coord.shape)
        if crop.shape[1] != self.crop_size[0] or crop.shape[2] != self.crop_size[1] or \
            crop.shape[3] != self.crop_size[2]:
            crop = zoom(crop,[1, float(self.crop_size[0])/crop.shape[1], float(self.crop_size[1])/crop.shape[2], \
                float(self.crop_size[2])/crop.shape[3]])#, mode='nearest') #order=1,
        return crop, target, bboxes, coord
    
class LabelMapping(object):
    def __init__(self, config, phase):
        self.stride = np.array(config['stride'])
        self.num_neg = int(config['num_neg'])
        self.th_neg = config['th_neg']
        self.anchors = np.asarray(config['anchors'])
        self.phase = phase
        if phase == 'train':
            self.th_pos = config['th_pos_train']
        elif phase == 'val':
            self.th_pos = config['th_pos_val']

            
    def __call__(self, input_size, target, bboxes, filename):
        stride = self.stride
        num_neg = self.num_neg
        th_neg = self.th_neg
        anchors = self.anchors
        th_pos = self.th_pos
        
        output_size = []
        for i in range(3):
            if input_size[i] % stride != 0:
                print filename
                print(input_size, stride)
            # assert(input_size[i] % stride == 0) 
            output_size.append(input_size[i] / stride)
        
        label = -1 * np.ones(output_size + [len(anchors), 5], np.float32)
        offset = ((stride.astype('float')) - 1) / 2
        oz = np.arange(offset, offset + stride * (output_size[0] - 1) + 1, stride)
        oh = np.arange(offset, offset + stride * (output_size[1] - 1) + 1, stride)
        ow = np.arange(offset, offset + stride * (output_size[2] - 1) + 1, stride)

        for bbox in bboxes:
            for i, anchor in enumerate(anchors):
                iz, ih, iw = select_samples(bbox, anchor, th_neg, oz, oh, ow)
                label[iz, ih, iw, i, 0] = 0

        if self.phase == 'train' and self.num_neg > 0:
            neg_z, neg_h, neg_w, neg_a = np.where(label[:, :, :, :, 0] == -1)
            neg_idcs = random.sample(range(len(neg_z)), min(num_neg, len(neg_z)))
            neg_z, neg_h, neg_w, neg_a = neg_z[neg_idcs], neg_h[neg_idcs], neg_w[neg_idcs], neg_a[neg_idcs]
            label[:, :, :, :, 0] = 0
            label[neg_z, neg_h, neg_w, neg_a, 0] = -1

        if np.isnan(target[0]):
            return label
        iz, ih, iw, ia = [], [], [], []
        for i, anchor in enumerate(anchors):
            iiz, iih, iiw = select_samples(target, anchor, th_pos, oz, oh, ow)
            iz.append(iiz)
            ih.append(iih)
            iw.append(iiw)
            ia.append(i * np.ones((len(iiz),), np.int64))
        iz = np.concatenate(iz, 0)
        ih = np.concatenate(ih, 0)
        iw = np.concatenate(iw, 0)
        ia = np.concatenate(ia, 0)
        flag = True 
        if len(iz) == 0:
            pos = []
            for i in range(3):
                pos.append(max(0, int(np.round((target[i] - offset) / stride))))
            idx = np.argmin(np.abs(np.log(target[3] / anchors)))
            pos.append(idx)
            flag = False
        else:
            idx = random.sample(range(len(iz)), 1)[0]
            pos = [iz[idx], ih[idx], iw[idx], ia[idx]]
        dz = (target[0] - oz[pos[0]]) / anchors[pos[3]]
        dh = (target[1] - oh[pos[1]]) / anchors[pos[3]]
        dw = (target[2] - ow[pos[2]]) / anchors[pos[3]]
        dd = np.log(target[3] / anchors[pos[3]])
        label[pos[0], pos[1], pos[2], pos[3], :] = [1, dz, dh, dw, dd]
        return label        

def select_samples(bbox, anchor, th, oz, oh, ow):
    z, h, w, d = bbox
    max_overlap = min(d, anchor)
    min_overlap = np.power(max(d, anchor), 3) * th / max_overlap / max_overlap
    if min_overlap > max_overlap:
        return np.zeros((0,), np.int64), np.zeros((0,), np.int64), np.zeros((0,), np.int64)
    else:
        s = z - 0.5 * np.abs(d - anchor) - (max_overlap - min_overlap)
        e = z + 0.5 * np.abs(d - anchor) + (max_overlap - min_overlap)
        mz = np.logical_and(oz >= s, oz <= e)
        iz = np.where(mz)[0]
        
        s = h - 0.5 * np.abs(d - anchor) - (max_overlap - min_overlap)
        e = h + 0.5 * np.abs(d - anchor) + (max_overlap - min_overlap)
        mh = np.logical_and(oh >= s, oh <= e)
        ih = np.where(mh)[0]
            
        s = w - 0.5 * np.abs(d - anchor) - (max_overlap - min_overlap)
        e = w + 0.5 * np.abs(d - anchor) + (max_overlap - min_overlap)
        mw = np.logical_and(ow >= s, ow <= e)
        iw = np.where(mw)[0]

        if len(iz) == 0 or len(ih) == 0 or len(iw) == 0:
            return np.zeros((0,), np.int64), np.zeros((0,), np.int64), np.zeros((0,), np.int64)
        
        lz, lh, lw = len(iz), len(ih), len(iw)
        iz = iz.reshape((-1, 1, 1))
        ih = ih.reshape((1, -1, 1))
        iw = iw.reshape((1, 1, -1))
        iz = np.tile(iz, (1, lh, lw)).reshape((-1))
        ih = np.tile(ih, (lz, 1, lw)).reshape((-1))
        iw = np.tile(iw, (lz, lh, 1)).reshape((-1))
        centers = np.concatenate([
            oz[iz].reshape((-1, 1)),
            oh[ih].reshape((-1, 1)),
            ow[iw].reshape((-1, 1))], axis = 1)
        
        r0 = anchor / 2
        s0 = centers - r0
        e0 = centers + r0
        
        r1 = d / 2
        s1 = bbox[:3] - r1
        s1 = s1.reshape((1, -1))
        e1 = bbox[:3] + r1
        e1 = e1.reshape((1, -1))
        
        overlap = np.maximum(0, np.minimum(e0, e1) - np.maximum(s0, s1))
        
        intersection = overlap[:, 0] * overlap[:, 1] * overlap[:, 2]
        union = anchor * anchor * anchor + d * d * d - intersection

        iou = intersection / union

        mask = iou >= th
        #if th > 0.4:
         #   if np.sum(mask) == 0:
          #      print(['iou not large', iou.max()])
           # else:
            #    print(['iou large', iou[mask]])
        iz = iz[mask]
        ih = ih[mask]
        iw = iw[mask]
        return iz, ih, iw

def collate(batch):
    if torch.is_tensor(batch[0]):
        return [b.unsqueeze(0) for b in batch]
    elif isinstance(batch[0], np.ndarray):
        return batch
    elif isinstance(batch[0], int):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], collections.Iterable):
        transposed = zip(*batch)
        return [collate(samples) for samples in transposed]

