import argparse
import os
import time
import numpy as np
import data
import weakdatav2
from importlib import import_module
import shutil
from utils import *
import sys
sys.path.append('../')
from split_combine import SplitComb
import csv
import torch
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable
# from config_training import config as config_training
# read filename dct to short the fname and speed up process time
import pandas
import pandas as pd
name = ['fname', 'newname']
pdfrm = pandas.read_csv('../fnamedct.csv', names=name)
fnamelst = pdfrm[name[0]].tolist()[1:]
newnamelst = pdfrm[name[1]].tolist()[1:]
fnamedct = {}
for i in xrange(len(fnamelst)):
    fnamedct[fnamelst[i]] = newnamelst[i]
from layers import acc
import pickle
parser = argparse.ArgumentParser(description='PyTorch DataBowl3 Detector')
parser.add_argument('--model', '-m', metavar='MODEL', default='base',
                    help='model')
parser.add_argument('--config', '-c', default='config_training', type=str)
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--save-freq', default='1', type=int, metavar='S',
                    help='save frequency')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--save-dir', default='', type=str, metavar='SAVE',
                    help='directory to save checkpoint (default: none)')
parser.add_argument('--test', default=0, type=int, metavar='TEST',
                    help='1 do test evaluation, 0 not')
parser.add_argument('--testthresh', default=-3, type=float,
                    help='threshod for get pbb')
parser.add_argument('--split', default=8, type=int, metavar='SPLIT',
                    help='In the test phase, split the image to 8 parts')
parser.add_argument('--gpu', default='all', type=str, metavar='N',
                    help='use gpu')
parser.add_argument('--n_test', default=4, type=int, metavar='N',
                    help='number of gpu for test')
parser.add_argument('--n_weak', default=10000, type=int, metavar='nweak',
                    help='number of weak data')
parser.add_argument('--weaknmsthresh', default=0.1, type=float, metavar='nms',
                    help='NMS threshold for weak supervision inference')
parser.add_argument('--weakdetp', default=-2, type=float, metavar='detectprobthresh',
                    help='detection probability threshold threshold for weak supervision inference') # 1 0.7311 2 0.8808 1.5 0.8176 0.1 0.5250 0.5 0.6225
parser.add_argument('--weakzp', default=0.99, type=float, metavar='slicethresh',
                    help='slice probability threshold threshold for weak supervision inference') # exp(-0.01): 0.1: 0.99
# parser.add_argument('--zp', default=0.75, type=float, metavar='detection',
                    # help='slice probability threshold threshold for weak supervision inference') # exp(-0.01): 0.1: 0.99
# calibweaklabelall = open('../../NLST/calibweaklabelall.csv', 'w') # open once
# csvwriter = csv.writer(calibweaklabelall)
# csvwriter.writerow(['fname', 'position', 'centerslice', 'lowz', 'upz', 'dataz', 'datax', 'datay'])
# weakpdfm = pd.read_csv('../../NLST/calibweaklabel.csv', names=['fname', 'position', 'centerslice'])
# weakdct = {}
# weakfnmlst = weakpdfm['fname'].tolist()[1:]
# weakposlst = weakpdfm['position'].tolist()[1:]
# weakcsllst = weakpdfm['centerslice'].tolist()[1:]
# for idx, fnm in enumerate(weakfnmlst):
#     if fnm not in weakdct: weakdct[fnm] = [[int(float(weakposlst[idx]))-1, float(weakcsllst[idx])]]
#     else: weakdct[fnm].append([int(float(weakposlst[idx]))-1, float(weakcsllst[idx])])
# weakdct2 = {}
# for fnm, weaklabel in weakdct.iteritems():
#     print('process'+ fnm)
#     data = np.load('../../NLST/preprocessnp/'+fnm+'_clean.npy')
#     z = data.shape[1]
#     weakdct2[fnm] = []
#     for wl in weaklabel:
#         zlst = []
#         flag = False
#         zlst = [max(0, wl[1]-13), min(z-1, wl[1]+13)] # luna16 data distribution
#         # for zidx in xrange(z):
#         #     if not flag and np.exp(-np.square(abs(zidx-wl[1])/z)) >= 0.99:
#         #         zlst.append(zidx)
#         #         flag = True
#         #     elif flag and np.exp(-np.square(abs(zidx-wl[1])/z)) <= 0.99:
#         #         zlst.append(zidx)
#         #         flag = False 
#         #         break
#         # if len(zlst) != 2: print(fnm, wl)
#         # if len(zlst) == 1 and flag: zlst.append(data.shape[1]-1)
#         if len(zlst) != 2 or zlst[0] == zlst[1]: 
#             print(fnm, wl, zlst, data.shape)
#             assert 1==0
#         weakdct2[fnm].append([wl[0], zlst[0], zlst[1], data.shape[1], data.shape[2], data.shape[3]])
#         csvwriter.writerow([fnm, wl[0], wl[1], zlst[0], zlst[1], data.shape[1], data.shape[2], data.shape[3]])
# calibweaklabelall.close()
weakdct = {} # for the next use
weakpdfm = pd.read_csv('../../NLST/calibweaklabelallsmall.csv', names=['fname', 'position', 'centerslice', 'lowz', 'upz', 'dataz', 'datax', 'datay'])
weakfnmlst = weakpdfm['fname'].tolist()[1:]
weakposlst = weakpdfm['position'].tolist()[1:]
weakcsllst = weakpdfm['centerslice'].tolist()[1:]
weaklwzlst = weakpdfm['lowz'].tolist()[1:]
weakupzlst = weakpdfm['upz'].tolist()[1:]
weakdtzlst = weakpdfm['dataz'].tolist()[1:]
weakdtxlst = weakpdfm['datax'].tolist()[1:]
weakdtylst = weakpdfm['datay'].tolist()[1:]
for idx, fnm in enumerate(weakfnmlst):
    # print(fnm, float(weakposlst[idx]), float(weakcsllst[idx]), float(weaklwzlst[idx]), weakupzlst[idx], weakdtzlst[idx], weakdtxlst[idx], \
    #     weakdtylst[idx])
    value = [int(float(weakposlst[idx])), float(weakcsllst[idx]), float(weaklwzlst[idx]), float(weakupzlst[idx]), \
        int(weakdtzlst[idx]), int(weakdtxlst[idx]), int(weakdtylst[idx])]
    if fnm not in weakdct: weakdct[fnm] = [value]
    else: weakdct[fnm].append(value)
print('load done!')
def main():
    global args
    args = parser.parse_args()
    print(args.config)
    config_training = import_module(args.config)
    config_training = config_training.config
    # from config_training import config as config_training
    torch.manual_seed(0)
    torch.cuda.set_device(0)

    model = import_module(args.model)
    config, net, loss, get_pbb = model.get_model()
    start_epoch = args.start_epoch
    save_dir = args.save_dir

    model2 = torch.nn.Sequential()
    model2.add_module('linear', torch.nn.Linear(3, 6, bias=True))
    model2.linear.weight = torch.nn.Parameter(torch.randn(6,3))
    model2.linear.bias = torch.nn.Parameter(torch.randn(6))
    loss2 = torch.nn.CrossEntropyLoss()
    optimizer2 = optim.SGD(model2.parameters(), lr=args.lr, momentum=0.9)#, weight_decay=args.weight_decay)
    
    if args.resume:
        print('resume from ', args.resume)
        checkpoint = torch.load(args.resume)
        # if start_epoch == 0:
        #     start_epoch = checkpoint['epoch'] + 1
        # if not save_dir:
        #     save_dir = checkpoint['save_dir']
        # else:
        #     save_dir = os.path.join('results',save_dir)
        # print(checkpoint.keys())
        net.load_state_dict(checkpoint['state_dict'])
        if start_epoch != 0:
            model2.load_state_dict(checkpoint['state_dict2'])
    # else:
    if start_epoch == 0:
        start_epoch = 1
    if not save_dir:
        exp_id = time.strftime('%Y%m%d-%H%M%S', time.localtime())
        save_dir = os.path.join('results', args.model + '-' + exp_id)
    else:
        save_dir = os.path.join('results',save_dir)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logfile = os.path.join(save_dir,'log')
    if args.test!=1:
        sys.stdout = Logger(logfile)
        pyfiles = [f for f in os.listdir('./') if f.endswith('.py')]
        for f in pyfiles:
            shutil.copy(f,os.path.join(save_dir,f))
    n_gpu = setgpu(args.gpu)
    args.n_gpu = n_gpu
    net = net.cuda()
    loss = loss.cuda()
    cudnn.benchmark = False #True
    net = DataParallel(net)
    traindatadir = config_training['train_preprocess_result_path']
    valdatadir = config_training['val_preprocess_result_path']
    testdatadir = config_training['test_preprocess_result_path']
    trainfilelist = []
    print config_training['train_data_path']
    for folder in config_training['train_data_path']:
        print folder
        for f in os.listdir(folder):
            if f.endswith('.mhd') and f[:-4] not in config_training['black_list']:
                if f[:-4] not in fnamedct:
                    trainfilelist.append(folder.split('/')[-2]+'/'+f[:-4])
                else:
                    trainfilelist.append(folder.split('/')[-2]+'/'+fnamedct[f[:-4]])
    valfilelist = []
    for folder in config_training['val_data_path']:
        for f in os.listdir(folder):
            if f.endswith('.mhd') and f[:-4] not in config_training['black_list']:
                if f[:-4] not in fnamedct:
                    valfilelist.append(folder.split('/')[-2]+'/'+f[:-4])
                else:
                    valfilelist.append(folder.split('/')[-2]+'/'+fnamedct[f[:-4]])
    testfilelist = []
    for folder in config_training['test_data_path']:
        for f in os.listdir(folder):
            if f.endswith('.mhd') and f[:-4] not in config_training['black_list']:
                if f[:-4] not in fnamedct:
                    testfilelist.append(folder.split('/')[-2]+'/'+f[:-4])
                else:
                    testfilelist.append(folder.split('/')[-2]+'/'+fnamedct[f[:-4]])
    if args.test == 1:
        margin = 32
        sidelen = 144
        import data
        split_comber = SplitComb(sidelen,config['max_stride'],config['stride'],margin,config['pad_value'])
        dataset = data.DataBowl3Detector(
            testdatadir,
            testfilelist,
            config,
            phase='test',
            split_comber=split_comber)
        test_loader = DataLoader(
            dataset,
            batch_size = 1,
            shuffle = False,
            num_workers = args.workers,
            collate_fn = data.collate,
            pin_memory=False)

        for i, (data, target, coord, nzhw) in enumerate(test_loader): # check data consistency
            if i >= len(testfilelist)/args.batch_size:
                break
        
        test(test_loader, net, get_pbb, save_dir,config)
        return
    #net = DataParallel(net)
    import data
    print len(trainfilelist)
    dataset = data.DataBowl3Detector(
        traindatadir,
        trainfilelist,
        config,
        phase = 'train')
    train_loader = DataLoader(
        dataset,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = args.workers,
        pin_memory=True)

    dataset = data.DataBowl3Detector(
        valdatadir,
        valfilelist,
        config,
        phase = 'val')
    val_loader = DataLoader(
        dataset,
        batch_size = args.batch_size,
        shuffle = False,
        num_workers = args.workers,
        pin_memory=True)
    # load weak data
    # weakdata = pd.read_csv(config_training['weaktrain_annos_path'], names=['fname', 'position', 'centerslice'])
    # weakfilename = weakdata['fname'].tolist()[1:]
    # weakfilename = list(set(weakfilename))
    # print('#weakdata', len(weakfilename))
    for i, (data, target, coord) in enumerate(train_loader): # check data consistency
        if i >= len(trainfilelist)/args.batch_size:
            break

    for i, (data, target, coord) in enumerate(val_loader): # check data consistency
        if i >= len(valfilelist)/args.batch_size:
            break
    optimizer = torch.optim.SGD(
        net.parameters(),
        args.lr,
        momentum = 0.9,
        weight_decay = args.weight_decay)
    npars = 0
    for par in net.parameters():
        curnpar = 1
        for s in par.size():
            curnpar *= s 
        npars += curnpar
    print('network size', npars)
    def get_lr(epoch):
        if epoch <= args.epochs * 1/2: #0.5:
            lr = args.lr
        elif epoch <= args.epochs * 3/4: #0.8:
            lr = 0.5 * args.lr
        # elif epoch <= args.epochs * 0.8:
        #     lr = 0.05 * args.lr
        else:
            lr = 0.1 * args.lr
        return lr
    for epoch in range(start_epoch, args.epochs + 1):
        # if epoch % 10 == 0:
        import data
        margin = 32
        sidelen = 144
        split_comber = SplitComb(sidelen,config['max_stride'],config['stride'],margin,config['pad_value'])
        dataset = weakdatav2.DataBowl3Detector(
                config_training['weaktrain_data_path'],
                weakdct.keys(),
                config,
                phase='test',
                split_comber=split_comber)
        weaktest_loader = DataLoader(
            dataset,
            batch_size = 1,
            shuffle = False,
            num_workers = args.workers,
            collate_fn = data.collate,
            pin_memory=False)
        print(len(weaktest_loader))
        for i, (data, target, coord, nzhw) in enumerate(weaktest_loader): # check data consistency
            if i >= len(testfilelist)/args.batch_size:
                break
        srslst, cdxlst, cdylst, cdzlst, dimlst, prblst, poslst, lwzlst, upzlst = weaktest(weaktest_loader, model2, net, get_pbb, save_dir, config, epoch)
        config['ep'] = epoch
        config['save_dir'] = save_dir
        with open(config['save_dir']+'weakinferep'+str(config['ep'])+'srs', 'wb') as fp:
            pickle.dump(srslst, fp)
        with open(config['save_dir']+'weakinferep'+str(config['ep'])+'cdx', 'wb') as fp:
            pickle.dump(cdxlst, fp)
        with open(config['save_dir']+'weakinferep'+str(config['ep'])+'cdy', 'wb') as fp:
            pickle.dump(cdylst, fp)
        with open(config['save_dir']+'weakinferep'+str(config['ep'])+'cdz', 'wb') as fp:
            pickle.dump(cdzlst, fp)
        with open(config['save_dir']+'weakinferep'+str(config['ep'])+'dim', 'wb') as fp:
            pickle.dump(dimlst, fp)
        with open(config['save_dir']+'weakinferep'+str(config['ep'])+'prb', 'wb') as fp:
            pickle.dump(prblst, fp)
        with open(config['save_dir']+'weakinferep'+str(config['ep'])+'pos', 'wb') as fp:
            pickle.dump(poslst, fp)
        pdfrm = pd.read_csv(config['save_dir']+'weakinferep'+str(config['ep'])+'.csv', names=['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter', 'probability', 'position'])
        if srslst:
            fnmlst = srslst #pdfrm['seriesuid'].tolist()[1:]
        dataset = weakdatav2.DataBowl3Detector(
            config_training['weaktrain_data_path'],
            list(set(fnmlst)),
            config,
            phase = 'train', fnmlst=srslst, cdxlst=cdxlst, cdylst=cdylst, cdzlst=cdzlst, dimlst=dimlst, prblst=prblst, poslst=poslst, lwzlst=lwzlst, upzlst=upzlst)
        weaktrain_loader = DataLoader(
            dataset,
            batch_size = args.batch_size,
            shuffle = True,
            num_workers = args.workers,
            pin_memory=True)
        print(len(weaktrain_loader))
        for i, (data, target, coord, prob, pos, feat) in enumerate(weaktrain_loader): # check data consistency
            # print(data.size(), target.size(), coord.size(), prob.size(), pos.size(), feat.size())
            if i >= len(trainfilelist)/args.batch_size:
                break
        weaktrain(weaktrain_loader, model2, loss2, optimizer2, net, loss, epoch, optimizer, get_lr, save_dir)#, args.save_freq, save_dir)
        train(train_loader, net, loss, epoch, optimizer, get_lr, args.save_freq, save_dir)
        validate(val_loader, net, loss)

def weaktrain(data_loader, model2, loss2, optimizer2, net, loss, epoch, optimizer, get_lr, save_dir):#, save_freq, save_dir):
    start_time = time.time()
    net.train()
    lr = get_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr*10

    metrics = []
    for i, (data, target, coord, prob, position, feat) in enumerate(data_loader):
        # print(data.size(), target.size(), coord.size(), prob.size(), position.size(), feat.size())
        data = Variable(data.cuda(async = True))
        target = Variable(target.cuda(async = True))
        coord = Variable(coord.cuda(async = True))

        output = net(data, coord)
        loss_output = loss(output, target, prob=prob, isp=True) # weighted loss
        optimizer.zero_grad()
        (loss_output[0]).backward()
        optimizer.step()

        loss_output[0] = loss_output[0].data[0]
        metrics.append(loss_output)

        optimizer2.zero_grad()
        # print(feat.size(), type(feat))
        fx = model2.forward(Variable(feat, requires_grad=False))#.cuda(async=True)))
        # print(fx.size(), position.size())
        position = Variable(position.view((-1,)), requires_grad=False)
        # print(fx.data.numpy(), position.data.numpy())
        output = loss2.forward(fx, position)
        (output*(prob.numpy()[0,0])).backward()

        # (loss2.forward(, 1)*(prob.numpy()[0,0])).backward()
        optimizer2.step()

    if epoch % args.save_freq == 0:            
        state_dict = net.module.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()

        state_dict2 = model2.state_dict()
        for key in state_dict2.keys():
            state_dict2[key] = state_dict2[key].cpu()
            
        torch.save({
            'epoch': epoch,
            'save_dir': save_dir,
            'state_dict': state_dict,
            'state_dict2': state_dict2,
            'args': args},
            os.path.join(save_dir, 'weak%03d.ckpt' % epoch))

    end_time = time.time()

    metrics = np.asarray(metrics, np.float32)
    print('Epoch %03d (lr %.5f)' % (epoch, lr))
    print('Weak Train:      tpr %3.2f, tnr %3.2f, total pos %d, total neg %d, time %3.2f' % (
        100.0 * np.sum(metrics[:, 6]) / np.sum(metrics[:, 7]),
        100.0 * np.sum(metrics[:, 8]) / np.sum(metrics[:, 9]),
        np.sum(metrics[:, 7]),
        np.sum(metrics[:, 9]),
        end_time - start_time))
    print('loss %2.4f, classify loss %2.4f, regress loss %2.4f, %2.4f, %2.4f, %2.4f' % (
        np.mean(metrics[:, 0]),
        np.mean(metrics[:, 1]),
        np.mean(metrics[:, 2]),
        np.mean(metrics[:, 3]),
        np.mean(metrics[:, 4]),
        np.mean(metrics[:, 5])))
    print

def train(data_loader, net, loss, epoch, optimizer, get_lr, save_freq, save_dir):
    start_time = time.time()
    
    net.train()
    lr = get_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    metrics = []

    for i, (data, target, coord) in enumerate(data_loader):
        data = Variable(data.cuda(async = True))
        target = Variable(target.cuda(async = True))
        coord = Variable(coord.cuda(async = True))

        output = net(data, coord)
        loss_output = loss(output, target)
        optimizer.zero_grad()
        loss_output[0].backward()
        optimizer.step()

        loss_output[0] = loss_output[0].data[0]
        metrics.append(loss_output)

    if epoch % args.save_freq == 0:            
        state_dict = net.module.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()
            
        torch.save({
            'epoch': epoch,
            'save_dir': save_dir,
            'state_dict': state_dict,
            'args': args},
            os.path.join(save_dir, '%03d.ckpt' % epoch))

    end_time = time.time()

    metrics = np.asarray(metrics, np.float32)
    print('Epoch %03d (lr %.5f)' % (epoch, lr))
    print('Train:      tpr %3.2f, tnr %3.2f, total pos %d, total neg %d, time %3.2f' % (
        100.0 * np.sum(metrics[:, 6]) / np.sum(metrics[:, 7]),
        100.0 * np.sum(metrics[:, 8]) / np.sum(metrics[:, 9]),
        np.sum(metrics[:, 7]),
        np.sum(metrics[:, 9]),
        end_time - start_time))
    print('loss %2.4f, classify loss %2.4f, regress loss %2.4f, %2.4f, %2.4f, %2.4f' % (
        np.mean(metrics[:, 0]),
        np.mean(metrics[:, 1]),
        np.mean(metrics[:, 2]),
        np.mean(metrics[:, 3]),
        np.mean(metrics[:, 4]),
        np.mean(metrics[:, 5])))
    print

def validate(data_loader, net, loss):
    start_time = time.time()
    
    net.eval()

    metrics = []
    for i, (data, target, coord) in enumerate(data_loader):
        data = Variable(data.cuda(async = True), volatile = True)
        target = Variable(target.cuda(async = True), volatile = True)
        coord = Variable(coord.cuda(async = True), volatile = True)

        output = net(data, coord)
        loss_output = loss(output, target, train = False)

        loss_output[0] = loss_output[0].data[0]
        metrics.append(loss_output)    
    end_time = time.time()

    metrics = np.asarray(metrics, np.float32)
    print('Validation: tpr %3.2f, tnr %3.8f, total pos %d, total neg %d, time %3.2f' % (
        100.0 * np.sum(metrics[:, 6]) / np.sum(metrics[:, 7]),
        100.0 * np.sum(metrics[:, 8]) / np.sum(metrics[:, 9]),
        np.sum(metrics[:, 7]),
        np.sum(metrics[:, 9]),
        end_time - start_time))
    print('loss %2.4f, classify loss %2.4f, regress loss %2.4f, %2.4f, %2.4f, %2.4f' % (
        np.mean(metrics[:, 0]),
        np.mean(metrics[:, 1]),
        np.mean(metrics[:, 2]),
        np.mean(metrics[:, 3]),
        np.mean(metrics[:, 4]),
        np.mean(metrics[:, 5])))
    print
    print

def test(data_loader, net, get_pbb, save_dir, config):
    start_time = time.time()
    save_dir = os.path.join(save_dir,'bbox')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(save_dir)
    net.eval()
    namelist = []
    split_comber = data_loader.dataset.split_comber
    for i_name, (data, target, coord, nzhw) in enumerate(data_loader):
        s = time.time()
        target = [np.asarray(t, np.float32) for t in target]
        lbb = target[0]
        nzhw = nzhw[0]
        name = data_loader.dataset.filenames[i_name].split('/')[-1].split('_clean')[0] #.split('-')[0]  wentao change
        data = data[0][0]
        coord = coord[0][0]
        isfeat = False
        if 'output_feature' in config:
            if config['output_feature']:
                isfeat = True
        n_per_run = args.n_test
        print(data.size())
        splitlist = range(0,len(data)+1,n_per_run)
        if splitlist[-1]!=len(data):
            splitlist.append(len(data))
        outputlist = []
        featurelist = []

        for i in range(len(splitlist)-1):
            input = Variable(data[splitlist[i]:splitlist[i+1]], volatile=True).cuda()
            inputcoord = Variable(coord[splitlist[i]:splitlist[i+1]], volatile=True).cuda()
            if isfeat:
                output,feature = net(input,inputcoord)
                featurelist.append(feature.data.cpu().numpy())
            else:
                output = net(input,inputcoord)
            outputlist.append(output.data.cpu().numpy())
        output = np.concatenate(outputlist,0)
        output = split_comber.combine(output,nzhw=nzhw)
        if isfeat:
            feature = np.concatenate(featurelist,0).transpose([0,2,3,4,1])[:,:,:,:,:,np.newaxis]
            feature = split_comber.combine(feature,sidelen)[...,0]

        thresh = args.testthresh # -8 #-3
        print 'pbb thresh', thresh
        pbb,mask = get_pbb(output,thresh,ismask=True)
        if isfeat:
            feature_selected = feature[mask[0],mask[1],mask[2]]
            np.save(os.path.join(save_dir, name+'_feature.npy'), feature_selected)
        #tp,fp,fn,_ = acc(pbb,lbb,0,0.1,0.1)
        #print([len(tp),len(fp),len(fn)])
        print([i_name,name])
        e = time.time()
        np.save(os.path.join(save_dir, name+'_pbb.npy'), pbb)
        np.save(os.path.join(save_dir, name+'_lbb.npy'), lbb)
    np.save(os.path.join(save_dir, 'namelist.npy'), namelist)
    end_time = time.time()
    print('elapsed time is %3.2f seconds' % (end_time - start_time))
    print
    print

def weaktest(data_loader, model2, net, get_pbb, save_dir, config, epoch):
    start_time = time.time()
    # save_dir = os.path.join(save_dir,'bbox')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # print(save_dir)
    net.eval()
    namelist = []
    split_comber = data_loader.dataset.split_comber
    datacount = 0
    annocount = 0
    fid = open(save_dir+'weakinferep'+str(epoch)+'.csv', 'w')
    csvwriter = csv.writer(fid)
    csvwriter.writerow(['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter', 'probability', 'position', 'lwz', 'upz'])
    srslst, cdxlst, cdylst, cdzlst, dimlst, prblst, poslst = [], [], [], [], [], [], []
    lwzlst, upzlst = [], []
    for i_name, (data, target, coord, nzhw) in enumerate(data_loader):
        datacount += 1
        s = time.time()
        target = [np.asarray(t, np.float32) for t in target]
        lbb = target[0]
        nzhw = nzhw[0]
        name = data_loader.dataset.filenames[i_name].split('/')[-1].split('_clean')[0] #.split('-')[0]  wentao change
        data = data[0][0]
        coord = coord[0][0]
        isfeat = False
        if 'output_feature' in config:
            if config['output_feature']:
                isfeat = True
        n_per_run = args.n_test
        # print(data.size())
        splitlist = range(0,len(data)+1,n_per_run)
        if splitlist[-1]!=len(data):
            splitlist.append(len(data))
        outputlist = []
        featurelist = []

        for i in range(len(splitlist)-1):
            input = Variable(data[splitlist[i]:splitlist[i+1]], volatile=True).cuda()
            inputcoord = Variable(coord[splitlist[i]:splitlist[i+1]], volatile=True).cuda()
            if isfeat:
                output,feature = net(input,inputcoord)
                featurelist.append(feature.data.cpu().numpy())
            else:
                output = net(input,inputcoord)
            outputlist.append(output.data.cpu().numpy())
        output = np.concatenate(outputlist,0)
        output = split_comber.combine(output,nzhw=nzhw)
        if isfeat:
            feature = np.concatenate(featurelist,0).transpose([0,2,3,4,1])[:,:,:,:,:,np.newaxis]
            feature = split_comber.combine(feature,sidelen)[...,0]

        thresh = args.weakdetp # -8 #-3
        # print 'pbb thresh', thresh

        pbb = get_pbb(output,thresh,ismask=True, weaklst=weakdct[name], model2=model2)
        e = time.time()
        for idx, wl in enumerate(weakdct[name]):
            if pbb[idx].shape[0]==0: continue
            # p = 1/(1+np.exp(-pbb[0,0])) * prob[0,0]
            annocount += 1
            srslst.append(name)
            cdxlst.append(pbb[idx][2])
            cdylst.append(pbb[idx][3])
            cdzlst.append(pbb[idx][1])
            dimlst.append(pbb[idx][4])
            prblst.append(pbb[idx][0])
            poslst.append(wl[0])
            lwzlst.append(wl[2])
            upzlst.append(wl[3])
            csvwriter.writerow([name, pbb[idx][2], pbb[idx][3], pbb[idx][1], pbb[idx][4], pbb[idx][0], wl[0], wl[2], wl[3]]) # x, y, z, d, p, pos
        # np.save(os.path.join(save_dir, name+'_pbb.npy'), pbb)
        # np.save(os.path.join(save_dir, name+'_lbb.npy'), lbb)
        if datacount > args.n_weak and annocount > args.n_weak:
            break
    # np.save(os.path.join(save_dir, 'namelist.npy'), namelist)
    fid.close()
    end_time = time.time()
    print('weak test elapsed time is %3.2f seconds' % (end_time - start_time))
    print
    print
    return srslst, cdxlst, cdylst, cdzlst, dimlst, prblst, poslst, lwzlst, upzlst

def singletest(data,net,config,splitfun,combinefun,n_per_run,margin = 64,isfeat=False):
    z, h, w = data.size(2), data.size(3), data.size(4)
    print(data.size())
    data = splitfun(data,config['max_stride'],margin)
    data = Variable(data.cuda(async = True), volatile = True,requires_grad=False)
    splitlist = range(0,args.split+1,n_per_run)
    outputlist = []
    featurelist = []
    for i in range(len(splitlist)-1):
        if isfeat:
            output,feature = net(data[splitlist[i]:splitlist[i+1]])
            featurelist.append(feature)
        else:
            output = net(data[splitlist[i]:splitlist[i+1]])
        output = output.data.cpu().numpy()
        outputlist.append(output)
        
    output = np.concatenate(outputlist,0)
    output = combinefun(output, z / config['stride'], h / config['stride'], w / config['stride'])
    if isfeat:
        feature = np.concatenate(featurelist,0).transpose([0,2,3,4,1])
        feature = combinefun(feature, z / config['stride'], h / config['stride'], w / config['stride'])
        return output,feature
    else:
        return output
if __name__ == '__main__':
    main()
