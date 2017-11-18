import random

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import zoom
from skimage import color

def train_model_class(model, optimizer, loader, class_weights, num_epochs=10, show_every=20):
    for epoch in range(num_epochs):
        print('Epoch %s' % epoch)
        print('=' * 10)
        
        running_loss = []
        for i, data in enumerate(iter(loader)):
            input, labels = data
            input, labels = Variable(input), Variable(labels)
            output = model(input)
            
            optimizer.zero_grad()
            logits = -F.log_softmax(output)
            loss_per_pixel = logits * labels
            loss_per_pixel = torch.sum(loss_per_pixel, 1)
            
            _, true_labels = torch.max(labels.data, 1)
            true_labels = true_labels.type(torch.LongTensor)
            
            b, w, h = true_labels.size()
            
            px_weights = torch.index_select(class_weights, 0, true_labels.view(b, -1).view(-1))
            px_weights = px_weights.view(b, -1).view(b, w, h)
                        
            loss_per_pixel = loss_per_pixel * Variable(px_weights)    
            loss = torch.mean(loss_per_pixel)
            
            loss.backward()
            optimizer.step()
            
            running_loss.append(loss.data[0])
            if show_every is not None and i % show_every == 0:
                print('Iter %s: %s' % (i, np.mean(running_loss)))
        print('Average loss: %s' % (np.mean(running_loss)))
    
    return model


def train_model_reg(model, optimizer, loader, num_epochs=10, show_every=20):
    for epoch in range(num_epochs):
        print('Epoch %s' % epoch)
        print('=' * 10)
        
        running_loss = []
        for i, data in enumerate(iter(loader)):
            input, labels = data
            input, labels = Variable(input), Variable(labels)
            output = model(input)
            
            optimizer.zero_grad()
            loss = F.mse_loss(output, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss.append(loss.data[0])
            if show_every is not None and i % show_every == 0:
                print('Iter %s: %s' % (i, np.mean(running_loss)))
        print('Average loss: %s' % (np.mean(running_loss)))
    
    return model


def predict_class(model, dset, cat2ab):
    input, label = dset[random.choice(np.arange(len(dset)))]
    L = input.numpy() + 50
    input = Variable(input.unsqueeze(0))
    out = model(input).squeeze(0)
    _, out = torch.max(out, 0)
    out = out.data.numpy()
    plt.hist(out.reshape(-1))
    out_actual = np.zeros((2,) + out.shape)
    for i in range(out_actual.shape[1]):
        for j in range(out_actual.shape[2]):
            a, b = cat2ab[out[i, j]]
            out_actual[:, i, j] = [a + 5, b + 5]
    out_actual = zoom(out_actual, (1, 4, 4))
    pred = np.concatenate((L, out_actual), axis=0).transpose(1, 2, 0).clip(-128, 128)
    pred = color.lab2rgb(pred.astype(np.float64))
    
    plt.figure()
    plt.title('Grayscale')
    plt.imshow(L.squeeze(0), cmap='gray')
    
    plt.figure()
    plt.title('Predicted')
    plt.imshow(pred)


def predict_reg(model, dset):
    input, label = dset[random.choice(np.arange(len(dset)))]
    L = input.numpy() + 50
    input = Variable(input.unsqueeze(0))
    out = model(input).squeeze(0)

    out_actual = out.data.numpy()
    out_actual = zoom(out_actual, (1, 4, 4))
    pred = np.concatenate((L, out_actual), axis=0).transpose(1, 2, 0).clip(-128, 128)
    pred = color.lab2rgb(pred.astype(np.float64))
    
    plt.figure()
    plt.title('Grayscale')
    plt.imshow(L.squeeze(0), cmap='gray')
    
    plt.figure()
    plt.title('Predicted')
    plt.imshow(pred)
