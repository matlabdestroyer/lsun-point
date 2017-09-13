import os
import cv2
import skimage.io as sio
import tqdm
import torch
import torch.nn.functional as F
import torch.nn.init as weight_init
from torch.autograd import Variable
from logger import Logger
from utils import timeit
import math
from PIL import Image
from model import *

class Operator():
    def __init__(self, name, pretrained, nb_class):
        self.name = name
        self.model = fcn(pretrained=pretrained, nb_class=nb_class).cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.criterion = Criterion()
        self.accuracy = Accuracy()
    def train(self, train_loader, validate_loader, epochs):
        for epoch in range(1, epochs+1):
            self.model.train()
            self.epoch = epoch
            hist = EpochHistory(length=len(train_loader))
            progress = tqdm.tqdm(train_loader)

            for img, lbl in progress:
                self.optimizer.zero_grad()
                loss, loss_term, acc = self.forward(img, lbl)
                loss.backward()
                self.optimizer.step()
                hist.add(loss, loss_term, acc)
                
                progress.set_description('Epoch#%i' % epoch)
                progress.set_postfix(
                    loss = '%.04f' % loss.data[0],
                    acc= '%.04f' % acc.data[0]
                    )
            metrics = dict(**hist.metric(), **self.evaluate(validate_loader, prefix='val_'))
            print(
                '---> Epoch#{}:\n loss: {loss:.4f}, acc: {accuracy:.4f}'
                'val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}'
                .format(self.epoch, **metrics)            
            )
    def forward(self, img, lbl, is_eval=False):
        def to_var(t):
            return Variable(t, volatile=is_eval).cuda()
        img, lbl = to_var(img), to_var(lbl)
        output = self.model(img)
        loss, loss_term = self.criterion(output, lbl)
        acc = self.accuracy(output, lbl)

        return (loss, loss_term, acc, output) if is_eval==True else (loss, loss_term, acc)
    
    @timeit
    def evaluate(self, data_loader, prefix=''):
        self.model.eval() 
        hist = EpochHistory(length=len(data_loader))
        for i, (img, lbl) in enumerate(data_loader):
            loss, loss_term, acc, ouptut = self.forward(img, lbl, is_eval=True)
            hist.add(loss, loss_term, acc)

        return hist.metric(prefix=prefix)

class EpochHistory():
    def __init__(self, length):
        self.count = 0
        self.len = length
        self.loss_term = {'xent':None, 'l1':None}
        self.losses = np.zeros(self.len)
        self.accuracies = np.zeros(self.len)
    
    def add(self, loss, loss_term, acc):
        self.losses[self.count] = loss.data[0]
        self.accuracies[self.count] = acc.data[0]

        for k,v in loss_term.items():
            if self.loss_term[k] is None:
                self.loss_term[k] = np.zeros(self.len)
            self.loss_term[k][self.count] = v.data[0]
        self.count += 1
    
    def metric(self, prefix=''):
        terms = {
            prefix + 'loss': self.losses.mean(),
            prefix + 'accuracy': self.accuracies.mean()
        }
        terms.update({
            prefix + k:v.mean() for k,v in self.loss_term.items()
            if v is not None
        })
        return terms

class Accuracy():
    def __call__(self, output, lbl):
        return self.pixelwise_accuracy(output, lbl)
    
    def pixelwise_accuracy(self, output, lbl):
        _, output = torch.max(output, 1)
        return (output == lbl).float().mean()

class Criterion():
    def __init__(self, l1_portion=0.1, weights=None):
        self.l1_criterion = nn.L1Loss().cuda()
        self.crossentropy = nn.CrossEntropyLoss(weight=weights).cuda()
        self.l1_portion = l1_portion
    
    def __call__(self, pred, target) -> (float,dict):
        loss, loss_term = self.pixelwise_loss(pred, target)
        return loss, loss_term
    
    def pixelwise_loss(self, pred, target):
        log_pred = F.log_softmax(pred)
        xent_loss = self.crossentropy(log_pred, target)

        onehot_target = (
            torch.FloatTensor(pred.size())
            .zero_().cuda()
            .scatter_(1, target.data.unsqueeze(1),1)
        )
        l1_loss = self.l1_criterion(pred, Variable(onehot_target))

        return xent_loss + self.l1_portion*l1_loss, {'xent':xent_loss, 'l1':l1_loss}