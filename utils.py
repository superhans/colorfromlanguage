from __future__ import division
import torch.utils.data as data
from skimage import io, color
import cv2
import pudb
import sklearn.neighbors as sknn
import numpy as np
from skimage.transform import resize
import h5py as h5
import time, collections, math
from collections import defaultdict, Counter
import re

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal, kaiming_uniform


def cvrgb2lab(img_rgb):
    cv_im_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype('float32')
    cv_im_lab[:, :, 0] *= (100. / 255)
    cv_im_lab[:, :, 1:] -= 128.
    return cv_im_lab


def init_modules(modules, init='uniform'):
    if init.lower() == 'normal':
        init_params = kaiming_normal
    elif init.lower() == 'uniform':
        init_params = kaiming_uniform
    else:
        return
    for m in modules:
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            init_params(m.weight)

def create_vocab(list_of_sentences):
    v = {}
    for sentence in list_of_sentences:
        sentence = re.sub(r'[^\w\s]','', sentence)
        for word in sentence.split(' '):
            if word not in v:
                v[word] = len(v)+1
    return v


def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def error_metric(rec_img_rgb, orig_img_rgb):
    eps = 0.000001

    r, c, _ = rec_img_rgb.shape

    rec_r, rec_g, rec_b = cv2.split(rec_img_rgb.astype('float32'))
    orig_r, orig_g, orig_b = cv2.split(orig_img_rgb.astype('float32'))
    
    Ir_r = (rec_r + rec_g + rec_b) / 3
    Io_r = (orig_r + orig_g + orig_b) / 3

    Ir_a = rec_b / (Ir_r+eps) - (rec_r + rec_g) / (2*Ir_r+eps)
    Ir_b = (rec_r - rec_g) / (Ir_r+eps)

    Io_a = orig_b / (Io_r+eps) - (orig_r + orig_g) / (2*Io_r+eps)
    Io_b = (orig_r - orig_g) / (Io_r+eps)

    error = np.sum((Ir_a - Io_a)**2) + np.sum((Ir_b - Io_b)**2)
    return error / (r*c) # remove this hardcoding later

# compute unmasked / masked rmse for batch of ab values
def rmse_ab(rec_img_lab, orig_img_lab, masks):
    # image shapes are bsz x 224 x 224 x 2
    delta = orig_img_lab - rec_img_lab
    rmse = np.sqrt(np.mean(delta ** 2))

    masked_delta = delta * masks[:,:,:,None]
    masked_delta = masked_delta[masked_delta != 0]
    masked_rmse = np.sqrt(np.mean(masked_delta ** 2))

    return rmse, masked_rmse


# support for top-k accuracy (input is bsz x num_labels)
def accuracy(output, target, topk=(1,5), mask=None):

    maxk = max(topk)
    if mask is not None:
        batch_size = mask.sum().cpu().data[0]
    else:
        batch_size = target.size(0)


    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred)).float()

    if mask is not None:
        correct = correct * mask[None, :]

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).sum(0, keepdim=True)
        res.append(correct_k * (100.0 / batch_size))
    return res


def cvrgb2lab(img_rgb):  
    cv_im_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype('float32')
    cv_im_lab[:, :, 0] *= (100. / 255)
    cv_im_lab[:, :, 1:] -= 128.
    return cv_im_lab 


def create_mask(bboxes):
    bsz = len(bboxes)
    mask = np.zeros((bsz, 56, 56))
    for i, bbox in enumerate(bboxes):
        x, y, w, h = bbox
        m = np.zeros((224, 224))
        m[y:y+h, x:x+w] = 1
        mask[i] = (cv2.resize(m, (56, 56)) > 0).astype('float32')
    return mask


def filter_bboxes(bboxes, threshold_low=1500, threshold_high=20000, exclude_duplicates=True):
    # look at areas of bboxes
    # filter out those examples where bboxes are too small
    # or bboxes are all zero, and remove duplicate labels

    idxs = []
    bboxes_filt = []
    label_count = Counter()
    for i, box in enumerate(bboxes):
        filt_box = box[(box[:, 1] > threshold_low) & (box[:, 1] < threshold_high)]
        if len(filt_box) > 0:

            if exclude_duplicates:
                for z in filt_box:
                    label_count[z[0]] += 1

                new_filtbox = []
                for z in filt_box:
                    if label_count[z[0]] == 1:
                        new_filtbox.append(z)

                if len(new_filtbox) > 0:
                    idxs.append(i)
                    bboxes_filt.append(new_filtbox)

            else:        
                idxs.append(i)
                bboxes_filt.append(filt_box)

    return idxs, bboxes_filt


def anneal(x, T):
    return np.exp(np.log(x)/T) / (np.sum(np.exp(np.log(x)/T), axis=0)+np.finfo(float).eps)

def annealing(input_, T=0.38):
    '''
        # input_ is batch_size x 224 x 224 x 313
        # T is annealed temperature
        # output is after annealing, and is of size batch_size x 224 x 224 x 313
    '''
#    input_ = input_.transpose(0, 2, 3, 1) # now becomes b x 224 x 224 x 313
    batch_size, h, w, q = input_.shape
    input_ = input_.reshape((batch_size*h*w, q)) 
    input_ = np.apply_along_axis(anneal, 1, input_, T)
    # input_ = np.mean(input_, axis=0)
    output_ = input_.reshape((batch_size, h, w, q))
    output_ = output_.transpose(0, 3, 1, 2) # now, back to b x 313 x 224 x 224
    return output_

def prior_boosting(prior_file, alpha, gamma):
    prior_probs = np.load(prior_file)

    # define uniform probability
    uni_probs = np.zeros_like(prior_probs)
    uni_probs[prior_probs!=0] = 1.
    uni_probs = uni_probs/np.sum(uni_probs)

    # convex combination of empirical prior and uniform distribution       
    prior_mix = (1-gamma)*prior_probs + gamma*uni_probs

    # set prior factor
    prior_factor = prior_mix**-alpha
    prior_factor[prior_factor==np.inf] = 0. # mask out unused classes
    prior_factor = prior_factor/np.sum(prior_probs*prior_factor) # re-normalize


    # implied empirical prior
    # implied_prior = prior_probs*prior_factor
    # implied_prior = implied_prior/np.sum(implied_prior) # re-normalize
    return prior_factor

class h5pyLoader(data.Dataset):
    def __init__(self, hfile, transform, fold):
        self.train_ims = h5.File(hfile, 'r')['train/images']
        self.val_ims = h5.File(hfile, 'r')['dev/images']
        self.transform = transform
        self.fold = fold

    def __getitem__(self, index):
        if self.fold == 'train':
            img = self.train_ims[index][0]
            img_lab = rgbim2lab(img)
            img_l = img_lab[:, :, 0] - 50
            target = enc(img_lab)

            return img_l, target

    def __len__(self):
        return len(self.train_ims)

def display(img):
    cv2.imshow('img', img.astype('uint8'))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def check_value(inds, val):
    ''' Check to see if an array is a single element equaling a particular value
    for pre-processing inputs in a function '''
    if(np.array(inds).size==1):
        if(inds==val):
            return True
    return False

def produce_minibatch_idxs(n, b):
    # n : total examples
    # b : batch size
    n_batches = n // b
    minibatches = [(i*b, (i+1)*b) for i in range(n_batches)]
    if (i+1)*b != n:
        minibatches.append( ((i+1)*b, n) )
    return minibatches

def na(): # shorthand for new axis
    return np.newaxis

def flatten_nd_array(pts_nd,axis=1):
    ''' Flatten an nd array into a 2d array with a certain axis
    INPUTS
        pts_nd       N0xN1x...xNd array
        axis         integer
    OUTPUTS
        pts_flt     prod(N \ N_axis) x N_axis array     '''
    NDIM = pts_nd.ndim
    SHP = np.array(pts_nd.shape)
    nax = np.setdiff1d(np.arange(0,NDIM),np.array((axis))) # non axis indices
    NPTS = np.prod(SHP[nax])
    axorder = np.concatenate((nax,np.array(axis).flatten()),axis=0)
    pts_flt = pts_nd.transpose((axorder))
    pts_flt = pts_flt.reshape(NPTS,SHP[axis])
    return pts_flt

def unflatten_2d_array(pts_flt,pts_nd,axis=1,squeeze=False):
    ''' Unflatten a 2d array with a certain axis
    INPUTS
        pts_flt     prod(N \ N_axis) x M array
        pts_nd      N0xN1x...xNd array
        axis        integer
        squeeze     bool     if true, M=1, squeeze it out
    OUTPUTS
        pts_out     N0xN1x...xNd array        '''
    NDIM = pts_nd.ndim
    SHP = np.array(pts_nd.shape)
    nax = np.setdiff1d(np.arange(0,NDIM),np.array((axis))) # non axis indices
    NPTS = np.prod(SHP[nax])

    if(squeeze):
        axorder = nax
        axorder_rev = np.argsort(axorder)
        M = pts_flt.shape[1]
        NEW_SHP = SHP[nax].tolist()
        pts_out = pts_flt.reshape(NEW_SHP)
        pts_out = pts_out.transpose(axorder_rev)
    else:
        axorder = np.concatenate((nax,np.array(axis).flatten()),axis=0)
        axorder_rev = np.argsort(axorder)

        M = pts_flt.shape[1]
        NEW_SHP = SHP[nax].tolist()
        NEW_SHP.append(M)
        pts_out = pts_flt.reshape(NEW_SHP)
        pts_out = pts_out.transpose(axorder_rev)

    return pts_out

def rgbfile2lab(im_file, resize_to=(224, 224)):
    rgb = cv2.imread(im_file)
    rgb = resize(rgb, resize_to, mode='reflect')
    return rgbim2lab(rgb)

def rgbim2lab(rgb):
    return color.rgb2lab(rgb) 

def labim2rgb(lab):
    # return (255.*color.lab2rgb(lab)).astype('uint8')
    return (255.*cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)).astype('uint8')


def labim2rgb_batch(lab):
    # lab is batchxhxwx3
    b, h, w, c = lab.shape
    rgb = np.zeros_like(lab)
    for i, lab_ in enumerate(lab):
        rgb[i] = labim2rgb(lab_)
    return rgb


class LookupEncode():
    '''Encode points using lookups'''
    def __init__(self, km_filepath=''):

        self.cc = np.load(km_filepath)
        self.offset = np.abs(np.amin(self.cc)) + 17 # add to get rid of negative numbers
        self.x_mult = 59 # differentiate x from y
        self.labels = {}
        for idx, (x,y) in enumerate(self.cc):
            x += self.offset
            x *= self.x_mult
            y += self.offset
            self.labels[x+y] = idx


    # returns bsz x 224 x 224 of bin labels (625 possible labels)
    def encode_points(self, pts_nd, grid_width=10):

        pts_flt = pts_nd.reshape((-1, 2))

        # round AB coordinates to nearest grid tick
        pgrid = np.round(pts_flt / grid_width) * grid_width

        # get single number by applying offsets
        pvals = pgrid + self.offset
        pvals = pvals[:, 0] * self.x_mult + pvals[:, 1]

        labels = np.zeros(pvals.shape,dtype='int32')

        # lookup in label index and assign values
        for k in self.labels:
            labels[pvals == k] = self.labels[k]

        return labels.reshape(pts_nd.shape[:-1])


    # return lab grid marks from probability distribution over bins
    def decode_points(self, pts_enc):
        print pts_enc
        return pts_enc.dot(self.cc)


class NNEncode():
    ''' Encode points using NN search and Gaussian kernel '''
    def __init__(self,NN,sigma,km_filepath='',cc=-1):
        if(check_value(cc,-1)):
            self.cc = np.load(km_filepath)
        else:
            self.cc = cc
        
        self.K = self.cc.shape[0]
        self.NN = int(NN)
        self.sigma = sigma
        self.nbrs = sknn.NearestNeighbors(n_neighbors=self.NN, algorithm='auto').fit(self.cc)

        self.alreadyUsed = False

    def encode_points_mtx_nd(self,pts_nd,axis=1,returnSparse=False,sameBlock=True):
        
        pts_flt = flatten_nd_array(pts_nd,axis=axis)
        P = pts_flt.shape[0]

        if(sameBlock and self.alreadyUsed):
            self.pts_enc_flt[...] = 0 # already pre-allocated
        else:
            self.alreadyUsed = True
            self.pts_enc_flt = np.zeros((P,self.K))
            self.p_inds = np.arange(0,P,dtype='int')[:,na()]
        
        P = pts_flt.shape[0]

        (dists,inds) = self.nbrs.kneighbors(pts_flt)

        wts = np.exp(-dists**2/(2*self.sigma**2))
        wts = wts/np.sum(wts,axis=1)[:,na()]

        self.pts_enc_flt[self.p_inds,inds] = wts
        pts_enc_nd = unflatten_2d_array(self.pts_enc_flt,pts_nd,axis=axis)

        return pts_enc_nd

    def decode_points_mtx_nd(self,pts_enc_nd,axis=1):
        pts_enc_flt = flatten_nd_array(pts_enc_nd,axis=axis)
        pts_dec_flt = np.dot(pts_enc_flt,self.cc)
        pts_dec_nd = unflatten_2d_array(pts_dec_flt,pts_enc_nd,axis=axis)
        return pts_dec_nd

    def decode_1hot_mtx_nd(self,pts_enc_nd,axis=1,returnEncode=False):
        pts_1hot_nd = nd_argmax_1hot(pts_enc_nd,axis=axis)
        pts_dec_nd = self.decode_points_mtx_nd(pts_1hot_nd,axis=axis)
        if(returnEncode):
            return (pts_dec_nd,pts_1hot_nd)
        else:
            return pts_dec_nd


def enc_noswap(img):   # b x h x w x 3
    # img = img.transpose(0, 3, 1, 2)
    out = nnenc.encode_points_mtx_nd(img[None, 1:, :, :], axis=1)
    return out

def enc(img):
    img = img.transpose(2, 0, 1)
    out = nnenc.encode_points_mtx_nd(img[None, 1:, :, :], axis=1)
    return out

def enc_batch(img):
    img = img.transpose(0, 3, 1, 2)
    t1 = time.time()
    out = nnenc.encode_points_mtx_nd(img[:, 1:, :, :], axis=1)
    t2 = time.time()
    print (t2-t1)
    return out

def enc_batch_nnenc(img, nnenc):
    img = img.transpose(0, 3, 1, 2)
    out = nnenc.encode_points_mtx_nd(img[:, 1:, :, :], axis=1)
    return out


def extract_l_ab(img_lab):
    if np.ndim(img_lab) == 3:
        pass
    elif np.ndim(img_lab) == 4:
        pass

# y = (color.lab2rgb(color.rgb2lab(x/255.))*255).astype('uint8')
def decode(nnenc, img_l, preds):
    decode_out = nnenc.decode_points_mtx_nd(preds)
    img_lab_out = np.concatenate((img_l[None, None, :, :], decode_out), axis=1).transpose(0, 2, 3, 1)
    return (255*color.lab2rgb(img_lab_out[0])).astype('uint8')

def decode_lookup(img_l, bins):
    img_lab_out = np.concatenate((img_l[None, :, :, None], bins), axis=3)
    return (255*color.lab2rgb(img_lab_out[0])).astype('uint8')

def test_img_ops():
    # t1 = time.time()
    read = 0.
    n = 100
    img_lab_ = np.zeros((n, 224, 224, 3))
    #for i in range(n):
    img_lab = rgbfile2lab('elephant.jpg')
    img_l = img_lab[:, :, 0]
    #    img_ab = img_lab[:, :, 1:]
    #    img_lab_[i] = img_lab

    img_ab_enc = enc(img_lab)
    # pudb.set_trace()
    # t2 = time.time()
    # print (t2 - t1)
    print np.max(img_ab_enc), ' ', np.min(img_ab_enc)
    print img_ab_enc.shape
    pudb.set_trace()
    img_dec = decode(nnenc, img_l, img_ab_enc)
    # img_dec = np.concatenate((img_l[:, :, np.newaxis], img_ab), axis=2)
    # img_dec = 255*color.lab2rgb(img_lab, illuminant='D50') # (255*np.clip(color.lab2rgb(img_lab), 0, 1)).astype('uint8')
    display(img_dec)

if __name__ == '__main__':
    gt_im = cvrgb2lab(cv2.imread('../data/sanity_check/color1.jpg'))[1:]
    film = cvrgb2lab(cv2.imread('../data/sanity_check/film1.jpg'))[1:]
    shit = cvrgb2lab(cv2.imread('../data/sanity_check/shit1.jpg'))[1:]
    zhang = cvrgb2lab(cv2.imread('../data/sanity_check/zhang1.jpg'))[1:]

    print 'gt_im gt_im %f %f' % (((gt_im - gt_im) ** 2).mean(), psnr(gt_im, gt_im))
    print 'gt_im film %f %f' % (((gt_im - film) ** 2).mean(), psnr(gt_im, film))
    print 'gt_im shit %f %f' % (((gt_im - shit) ** 2).mean(), psnr(gt_im, shit))
    print 'gt_im zhang %f %f' % (((gt_im - zhang) ** 2).mean(), psnr(gt_im, zhang))

