import argparse
import time

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import  DataLoader
from torchvision import models, datasets, transforms
import torch.nn.functional as F
from torch import optim
from collections import OrderedDict

from PIL import Image as I

from utils import save_checkpoint, load_checkpoint

def parse_args():
  parser = argparse.ArgumentParser(description="Load data and train model")
  parser.add_argument("--data_dir",action="store")
  parser.add_argument("--arch",dest="arch",default="vgg16",help="vgg16")
  parser.add_argument("--learning_rate",dest="learning rate",default="0.001")
  parser.add_argument("--epochs",dest="epochs",default="3")
  parser.add_argument("--save_dir",dest="save dir",action="store",default="checkpoint.pth")
  return parser.parse_args()

# making function for test_loss and accuracy (without backpropagation)
def validetion(model,testloader,criterion):
  test_loss = 0
  accuracy = 0
  for images , labels in testloader:
    # transferin images, labels to device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device;
    images,labels = images.to(device),labels.to(device)

    # feedforwarding for images
    output = model.forward(images)

    # finding loss
    test_loss += criterion(output,labels).item()

    # log to exp (convert - values to + values)
    ps = torch.exp(output)

    # comparing result to actual value
    equality = (labels.data == ps.max(1)[1])
    # accuracy mean fo equlity
    accuracy += equality.type(torch.FloatTensor).mean()
  return test_loss, accuracy

# train model and count(tranning loss,test loss, accuracy)

def train(model,trainloader,testloader,criterion,optimizer,epochs=3,print_every=10):
  steps = 0
  tranning_loss = 0
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  device;

  # measure time for whole process
  whole_time = time.time()
  for e in range(epochs):
    model.train() # train mode (on dropout)
    epoch_time = time.time() # measure time for per epoch
    for images, labels in trainloader:
      steps += 1
      iter_time = time.time() # measure time for per 10 iteration for trainloader and all iteration for testloader
      # transferin images, labels to device
      images,labels = images.to(device),labels.to(device)
      # zero gradient
      optimizer.zero_grad()
      # feedforwarding for images
      output = model.forward(images)
      # finding loss
      loss = criterion(output,labels)
      # backprop 
      loss.backward()
      # updating weights and bias
      optimizer.step()

      tranning_loss += loss.item()
      
      # for printing loss and accuracy
      if steps % print_every == 0 :
        model.eval()
        with torch.no_grad():
          test_loss, accuracy = validetion(model,testloader,criterion)
        print("Epoch:{}/{}...".format(e+1,epochs),
              "Tranning loss:{:.3f}...".format(tranning_loss/print_every),
              "Test loss:{:.3f}....".format(test_loss/len(testloader)),
              "Accuracy:{:.3f}...".format(accuracy/len(testloader)),
              "Time per {} iteration: {:.3f} second..".format(print_every,time.time()-iter_time),
              "Steps:{}...".format(steps))
        tranning_loss = 0
    print("Time For {} Epoch:{:.3f} minute...".format(e+1,(time.time()-epoch_time)/60))
  print("Time For Whole process:{:.3f} minute...".format((time.time()-whole_time)/60)) 
def main():
  print("working")
  args = parse_args()
  # add directory
  data_dir = 'flowers'
  train_dir = data_dir + '/train'
  test_dir = data_dir + '/test'
  valid_dir = data_dir+ '/valid'

  # transform
  train_transform = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),                                    
                                        transforms.ToTensor(),
                                        transforms.Normalize( [0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
  test_transform = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])
  valid_transform =transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])


  # datasets and dataloader
  # datasets
  train_dataset = datasets.ImageFolder(train_dir , transform = train_transform)
  test_dataset = datasets.ImageFolder(test_dir,transform=test_transform)
  valid_dataset = datasets.ImageFolder(valid_dir,transform=valid_transform)

  # dataloader
  trainloader = DataLoader(train_dataset, batch_size=32,shuffle=True)
  testloader = DataLoader(test_dataset,batch_size=32,shuffle=True)
  validloader = DataLoader(valid_dataset,batch_size=16,shuffle=True)


  # download pre-train model
  model = models.vgg16(pretrained=True)
  model;
  for param in model.parameters():
    param.requires_grad = False

  classifier = nn.Sequential(OrderedDict([('fc1',nn.Linear(25088,1024)),
                                                ('relu1',nn.ReLU()),
                                                ('fc2',nn.Linear(1024,256)),
                                                ('relu2',nn.ReLU()),
                                                ('dropout',nn.Dropout(p=0.2)),
                                                ('output',nn.Linear(256,102)),
                                                ('softmax',nn.LogSoftmax(dim=1))]))
  criterion = nn.NLLLoss()
  model.classifier = classifier
  optimizer = optim.Adam(model.classifier.parameters(),lr=0.001)
  # move model to device (cuda)
  model.to('cuda');
  criterion = nn.NLLLoss() # using criterion and optimizer similar to pytorch lectures (densenet)
  optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
  epochs = int(args.epochs)
  model.class_to_idx = train_dataset.class_to_idx
  train(model,trainloader,testloader,criterion,optimizer,epochs=3,print_every=10)
  path = 'checkpoint.pth' # get the new save location 
  save_checkpoint(path, model, optimizer, args, classifier)

if __name__ == "__main__":
  main()
