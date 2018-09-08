import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

from collections import defaultdict, Counter
import cPickle as pickle
import h5py as h5
import json
import numpy as np
import cv2
import string
import time
import random
import os, sys
import argparse
import scipy.ndimage.interpolation as sni
from skimage import io, color
from random import shuffle
from itertools import izip
from utils import decode_lookup, produce_minibatch_idxs, rgbim2lab,\
    enc_batch, prior_boosting, display, decode,init_modules,cvrgb2lab,\
    annealing, enc_batch_nnenc, LookupEncode, labim2rgb, error_metric, rmse_ab
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack

# standard bilstm
class CaptionEncoder(nn.Module):
    def __init__(self, word_embedding_dim, hidden_dim, vocab_size, train_vocab_embeddings):
        super(CaptionEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, word_embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(train_vocab_embeddings))
        self.hidden_size = hidden_dim
        self.lstm = nn.LSTM(word_embedding_dim, hidden_dim, num_layers=1,
            bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, captions, lens):
        bsz, max_len = captions.size()
        embeds = self.dropout(self.embedding(captions))

        lens, indices = torch.sort(lens, 0, True)
        _, (enc_hids, _) = self.lstm(pack(embeds[indices], lens.tolist(), batch_first=True))
        enc_hids = torch.cat( (enc_hids[0], enc_hids[1]), 1)
        _, _indices = torch.sort(indices, 0)
        enc_hids = enc_hids[_indices]

        return enc_hids


class FiLM(nn.Module):
    """
    A Feature-wise Linear Modulation Layer from
    'FiLM: Visual Reasoning with a General Conditioning Layer'

    How this layer works : 
    x = Variable(torch.randn(2, 64, 32 ,32))       
    gammas = Variable(torch.randn(2, 64)) # gammas and betas have to be 64 
    betas = Variable(torch.randn(2, 64))           
    y = film(x, gammas, betas)
    print y.size()
    y is : [2, 64, 32, 32]
 
    """
    def forward(self, x, gammas, betas):
        gammas = gammas.unsqueeze(2).unsqueeze(3).expand_as(x)
        betas = betas.unsqueeze(2).unsqueeze(3).expand_as(x)
        return (gammas * x) + betas

class FilMedResBlock(nn.Module):
    expansion = 1
    '''
    A much simplified version
    '''
    def __init__(self, in_dim, out_dim, stride=1, padding=1, dilation=1):
        super(FilMedResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, in_dim, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=stride, padding=1,
		dilation=dilation) # bias=False? check what perez did
        self.bn2 = nn.BatchNorm2d(out_dim)
        self.film = FiLM()
        init_modules(self.modules())

    def forward(self, x, gammas, betas):
        out = x
        out = F.relu(self.conv1(out))
        out = self.bn2(F.relu(self.conv2(out)))
        out = F.relu(self.film(out, gammas, betas))
        out += x
        return out

class AutocolorizeResnet(nn.Module):
    def __init__(self, vocab_size, feature_dim=(512, 28, 28), d_hid=256, d_emb = 300, num_modules=4, num_classes=625, train_vocab_embeddings=None):
        super(AutocolorizeResnet, self).__init__()
        self.num_modules = num_modules
	self.n_lstm_hidden = d_hid
        self.block = FilMedResBlock
        self.in_dim = feature_dim[0]
        self.num_classes = num_classes
        dilations = [1, 1, 1, 1]
        self.caption_encoder = CaptionEncoder(d_emb, d_hid, vocab_size, train_vocab_embeddings)

#        self.function_modules = {}
#        for fn_num in range(self.num_modules):
#        self.add_module(str(fn_num), mod)
#        self.function_modules[fn_num] = mod

        self.mod1 = self.block(self.in_dim, self.in_dim, dilations[0])
        self.mod2 = self.block(self.in_dim, self.in_dim, dilations[1])
        self.mod3 = self.block(self.in_dim, self.in_dim, dilations[2])
        self.mod4 = self.block(self.in_dim, self.in_dim, dilations[3])

	# put this in a loop later # there's an *2 because of bilstm and because of film
	self.dense_film_1 = nn.Linear(self.n_lstm_hidden*2, self.in_dim*2) 
	self.dense_film_2 = nn.Linear(self.n_lstm_hidden*2, self.in_dim*2)
	self.dense_film_3 = nn.Linear(self.n_lstm_hidden*2, self.in_dim*2)
	# out = x # 2x512x28x28
	# out = F.relu(self.conv1(out)) # 2x512x28x28
	self.dense_film_4 = nn.Linear(self.n_lstm_hidden*2, self.in_dim*2)
	self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        self.classifier = nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, dilation=1)	


    def forward(self, x, captions, caption_lens):
	caption_features = self.caption_encoder(captions, caption_lens)
        # out = F.relu(self.bn1(self.conv1(x)))

	dense_film_1 = self.dense_film_1(caption_features)
	dense_film_2 = self.dense_film_2(caption_features)
	dense_film_3 = self.dense_film_3(caption_features)
	dense_film_4 = self.dense_film_4(caption_features) # bsz * 128

	gammas1, betas1 = torch.split(dense_film_1, self.in_dim, dim=-1)
	gammas2, betas2 = torch.split(dense_film_2, self.in_dim, dim=-1)
	gammas3, betas3 = torch.split(dense_film_3, self.in_dim, dim=-1)
	gammas4, betas4 = torch.split(dense_film_4, self.in_dim, dim=-1)

        out = self.mod1(x, gammas1, betas1) # out is 2x512x28x28
        out = self.mod2(out, gammas2, betas2) # out is 2x512x28x28
        out = self.mod3(out, gammas3, betas3)
        out_last = self.mod4(out, gammas4, betas4)
	
	out = self.upsample(out_last)
        out = self.classifier(out)
        out = out.permute(0, 2, 3, 1).contiguous()
        out = out.view(-1, self.num_classes)
        return out, out_last
     

def train(minibatches, net, optimizer, epoch, prior_probs, img_save_folder):
    stime = time.time()
    c = Counter()
    for i, (batch_start, batch_end) in enumerate(minibatches):
	img_rgbs = train_origs[batch_start:batch_end]
        img_labs = np.array([cvrgb2lab(img_rgb) for img_rgb in img_rgbs])

        input_ = torch.from_numpy(train_ims[batch_start:batch_end])
        target = torch.from_numpy(lookup_enc.encode_points(img_labs[:, ::4, ::4, 1:]))

	# rand_idx = np.random.randint(5) # 5 captions per batch
	input_captions_ = train_words[batch_start:batch_end]
	input_lengths_ = train_lengths[batch_start:batch_end]

	# for now just choose first caption
	input_captions = Variable(torch.from_numpy(\
	    input_captions_.astype('int32')).long().cuda())
	input_caption_lens = torch.from_numpy(\
	    input_lengths_.astype('int32')).long().cuda()

        input_ims = Variable(input_.float().cuda())
        target = Variable(target.long()).cuda()

        optimizer.zero_grad()
        output, _ = net(input_ims, input_captions, input_caption_lens)

        loss = loss_function(output, target.view(-1)) 

        loss.backward()
        optimizer.step()
        if i % 50 == 0:
            print 'loss at epoch %d, batch %d / %d = %f, time: %f s' % \
                (epoch, i, len(minibatches), loss.data[0], time.time()-stime)
            stime = time.time()

            if True: # args.logs:

                # softmax output and multiply by grid
                dec_inp = nn.Softmax()(output) # 12544x625
                AB_vals = dec_inp.mm(cuda_cc) # 12544x2
                # reshape and select last image of batch]
                AB_vals = AB_vals.view(len(img_labs), 56, 56, 2)[-1].data.cpu().numpy()[None,:,:,:]
                AB_vals = cv2.resize(AB_vals[0], (224, 224),
                     interpolation=cv2.INTER_CUBIC)
                img_dec = labim2rgb(np.dstack((np.expand_dims(img_labs[-1, :, :, 0], axis=2), AB_vals)))
                img_labs_tosave = labim2rgb(img_labs[-1])

		word_list = list(input_captions_[-1, :input_lengths_[-1]])     
		words = '_'.join(vrev.get(w, 'unk') for w in word_list) 

                cv2.imwrite('%s/%d_%d_bw.jpg'%(img_save_folder, epoch, i),
                    cv2.cvtColor(img_rgbs[-1].astype('uint8'), 
                    cv2.COLOR_RGB2GRAY))
                cv2.imwrite('%s/%d_%d_color.jpg'%(img_save_folder, epoch, i),
                    img_rgbs[-1].astype('uint8'))
                cv2.imwrite('%s/%d_%d_rec_%s.jpg'%(img_save_folder, epoch, i, words),
                    img_dec.astype('uint8'))
           
                if i == 0: 
                    torch.save({
                        'epoch': epoch + 1,
                        'state_dict': net.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'loss': loss.data[0],
                    }, args.model_save_file+'_' + str(epoch)+'_'+str(i)+'.pth.tar')
    return net


def scale_attention_map(x):                                                                   
    x = (x - np.min(x)) / (np.max(x) - np.min(x))                                
    y = x * 255.                                                                 
    y = cv2.cvtColor(y.astype('uint8'), cv2.COLOR_GRAY2RGB).astype('uint8')      
    y = cv2.applyColorMap(y, cv2.COLORMAP_JET)                                   
    return cv2.resize(y, (224, 224), interpolation = cv2.INTER_LANCZOS4)         


def evaluate_attention_maps(minibatches, net, epoch, img_save_folder, save_every=20):
    stime = time.time()
    c = Counter()
    val_full_loss = 0.
    val_masked_loss = 0.
    val_loss = 0.

    n_val_ims = 0
    for i, (batch_start, batch_end) in enumerate(val_minibatches):
        img_rgbs = val_origs[batch_start:batch_end]
        img_labs = np.array([cvrgb2lab(img_rgb) for img_rgb in img_rgbs])

        input_ = torch.from_numpy(val_ims[batch_start:batch_end])
        gt_abs = img_labs[:, ::4, ::4, 1:]
        target = torch.from_numpy(lookup_enc.encode_points(gt_abs))

        input_captions_ = val_words[batch_start:batch_end]
        input_lengths_ = val_lengths[batch_start:batch_end]

        input_captions = Variable(torch.from_numpy(\
            input_captions_.astype('int32')).long().cuda())
        input_caption_lens = torch.from_numpy(\
            input_lengths_.astype('int32')).long().cuda()

        input_ims = Variable(input_.float().cuda())
        target = Variable(target.long()).cuda()
    
        output, output_maps = net(input_ims, input_captions, input_caption_lens)

        # softmax output and multiply by grid
        dec_inp = nn.Softmax()(output) # 12544x625
        AB_vals = dec_inp.mm(cuda_cc) # 12544x2
        # reshape and select last image of batch]
        AB_vals = AB_vals.view(len(img_labs), 56, 56, 2).data.cpu().numpy()

        n_val_ims += len(AB_vals)
        for k, (img_rgb, AB_val) in enumerate(zip(img_rgbs, AB_vals)):
	       # attention stuff
               AB_val = cv2.resize(AB_val, (224, 224),
                     interpolation=cv2.INTER_CUBIC)
               img_dec = labim2rgb(np.dstack((np.expand_dims(img_labs[k, :, :, 0], axis=2), AB_val)))
              
               val_loss += error_metric(img_dec, img_rgb)
               if k == 0 and i%save_every == 0:
	           output_maps = torch.mean(output_maps, dim=1)
                   output_maps = output_maps.data.cpu().numpy()
                   output_maps = scale_attention_map(output_maps[k])

		   word_list = list(input_captions_[k, :input_lengths_[k]])     
		   words = '_'.join(vrev.get(w, 'unk') for w in word_list) 

                   img_labs_tosave = labim2rgb(img_labs[k])
                   cv2.imwrite('%s/%d_%d_bw.jpg'%(img_save_folder, epoch, i),
                       cv2.cvtColor(img_rgbs[k].astype('uint8'), 
                       cv2.COLOR_RGB2GRAY))
                   cv2.imwrite('%s/%d_%d_color.jpg'%(img_save_folder, epoch, i),
                       img_rgbs[k].astype('uint8'))
                   cv2.imwrite('%s/%d_%d_rec_%s.jpg'%(img_save_folder, epoch, i, words),
                       img_dec.astype('uint8'))
	   	   cv2.imwrite('%s/%d_%d_att.jpg'%(img_save_folder, epoch, i), output_maps)
          
    return val_loss / len(val_minibatches) # , val_masked_loss / len(val_minibatches)



def evaluate(minibatches, net, epoch, img_save_folder, save_every=20):
    stime = time.time()
    c = Counter()
    val_full_loss = 0.
    val_masked_loss = 0.
    val_loss = 0.

    n_val_ims = 0
    for i, (batch_start, batch_end) in enumerate(val_minibatches):
        img_rgbs = val_origs[batch_start:batch_end]
        img_labs = np.array([cvrgb2lab(img_rgb) for img_rgb in img_rgbs])

        input_ = torch.from_numpy(val_ims[batch_start:batch_end])
        gt_abs = img_labs[:, ::4, ::4, 1:]
        target = torch.from_numpy(lookup_enc.encode_points(gt_abs))

        input_captions_ = val_words[batch_start:batch_end]
        input_lengths_ = val_lengths[batch_start:batch_end]

        input_captions = Variable(torch.from_numpy(\
            input_captions_.astype('int32')).long().cuda())
        input_caption_lens = torch.from_numpy(\
            input_lengths_.astype('int32')).long().cuda()

        input_ims = Variable(input_.float().cuda())
        target = Variable(target.long()).cuda()
    
        output, _ = net(input_ims, input_captions, input_caption_lens)

        # softmax output and multiply by grid
        dec_inp = nn.Softmax()(output) # 12544x625
        AB_vals = dec_inp.mm(cuda_cc) # 12544x2
        # reshape and select last image of batch]
        AB_vals = AB_vals.view(len(img_labs), 56, 56, 2).data.cpu().numpy()

        n_val_ims += len(AB_vals)
        for k, (img_rgb, AB_val) in enumerate(zip(img_rgbs, AB_vals)):
               AB_val = cv2.resize(AB_val, (224, 224),
                     interpolation=cv2.INTER_CUBIC)
               img_dec = labim2rgb(np.dstack((np.expand_dims(img_labs[k, :, :, 0], axis=2), AB_val)))
                
               val_loss += error_metric(img_dec, img_rgb)
               if k == 0 and i%save_every == 0:

		   word_list = list(input_captions_[k, :input_lengths_[k]])     
		   words = '_'.join(vrev.get(w, 'unk') for w in word_list) 

                   img_labs_tosave = labim2rgb(img_labs[k])
                   cv2.imwrite('%s/%d_%d_bw.jpg'%(img_save_folder, epoch, i),
                       cv2.cvtColor(img_rgbs[k].astype('uint8'), 
                       cv2.COLOR_RGB2GRAY))
                   cv2.imwrite('%s/%d_%d_color.jpg'%(img_save_folder, epoch, i),
                       img_rgbs[k].astype('uint8'))
                   cv2.imwrite('%s/%d_%d_rec_%s.jpg'%(img_save_folder, epoch, i, words),
                       img_dec.astype('uint8'))
          
    return val_loss / len(val_minibatches) # , val_masked_loss / len(val_minibatches)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='resnet coco colorization')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--start-epoch', '-se', type=int, default=0, help='starting epoch')
    parser.add_argument('--end-epoch', '-ee', type=int, default=30, help='ending epoch')
    parser.add_argument('--gpuid', '-g', default='1', type=str, help='which gpu to use')
    parser.add_argument('--batch_size', '-b', default=24, type=int, help='batch size')
    parser.add_argument('--d_emb', default=300, type=int, help='word-embedding dimension')
    parser.add_argument('--d_hid', default=150, type=float, help='lstm hidden dimension')
    parser.add_argument('--h5_file', help='h5 file which contains everything except features')
    parser.add_argument('--features_file', help='h5 file which contains features')
    parser.add_argument('--vocab_file_name', help='vocabulary file')
    parser.add_argument('--image_save_folder', help='prefix of the folders where images are stored')
    parser.add_argument('--model_save_file', help='prefix of the model save file')
    parser.add_argument('--save_attention_maps', default=0, help='save maps as well')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
    train_vocab = pickle.load(open(args.vocab_file_name, 'r'))
    train_vocab_embeddings = pickle.load(open('./priors/w2v_embeddings_colors.p', 'r'))


    # seeds                     
    torch.manual_seed(1000)     
    torch.cuda.manual_seed(1000)
    random.seed(1000)           
    np.random.seed(1000)        

    # initialize quantized LAB encoder
    lookup_enc = LookupEncode('./priors/full_lab_grid_10.npy')
    num_classes = lookup_enc.cc.shape[0]

    cuda_cc = Variable(torch.from_numpy(lookup_enc.cc).float().cuda())

    hfile = args.h5_file
    hf = h5.File(hfile, 'r')
    features_file = args.features_file
    ff = h5.File(features_file, 'r')

    # color rebalancing
    alpha = 1.
    gamma = 0.5
    gradient_prior_factor = Variable(torch.from_numpy(
        prior_boosting('./priors/coco_priors_onehot_625.npy', alpha, gamma)).float().cuda())
    print 'rebalancing'
    loss_function = nn.CrossEntropyLoss(weight=gradient_prior_factor)
    vrev = dict((v,k) for (k,v) in train_vocab.iteritems())          
    n_vocab = len(train_vocab)

    net = AutocolorizeResnet(n_vocab, train_vocab_embeddings=train_vocab_embeddings) # leave other stuff at default values
    net.cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    train_origs = hf['train_ims']
    train_ims = ff['train_features']             
    train_words = hf['train_words']                                         
    train_lengths = hf['train_length']                                      
                                    
    val_origs = hf['val_ims']                                     
    val_ims = ff['val_features']                                                 
    val_words = hf['val_words']                                             
    val_lengths = hf['val_length']                                          
                                                                         
    n_train_ims = len(train_ims)                                             
    minibatches = produce_minibatch_idxs(n_train_ims, args.batch_size)[:-1]  
    n_val_ims = len(val_ims)                                                 
    val_minibatches = produce_minibatch_idxs(n_val_ims, 4)[:-1]

    val_img_save_folder = args.image_save_folder+'_val'
    if not os.path.exists(val_img_save_folder): 
        os.makedirs(val_img_save_folder) 
                                                                                      
    img_save_folder = args.image_save_folder+'_train' 
    if not os.path.exists(img_save_folder):
        os.makedirs(img_save_folder)                                    

    print img_save_folder
    print 'start training ....'

    for epoch in range(args.start_epoch, args.end_epoch):
	random.shuffle(minibatches)
        random.shuffle(val_minibatches)

	net = train(minibatches, net, optimizer, epoch, gradient_prior_factor,
	    img_save_folder)
	t = time.time()
        if args.save_attention_maps == 0:
            val_full_loss = evaluate(val_minibatches, net, epoch, val_img_save_folder)
        else:
            val_full_loss = evaluate_attention_maps(val_minibatches, net, epoch, val_img_save_folder)
        # print 'full image rmse: %f' % (val_full_loss)


        
