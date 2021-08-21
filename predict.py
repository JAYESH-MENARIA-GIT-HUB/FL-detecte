import argparse

import torch
from torch.autograd import Variable
from torchvision import transforms, models
import torch.nn.functional as F

import numpy as np

from PIL import Image

import json
import os
import random

from utils import load_checkpoint, load_cat_names

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', action='store', default='checkpoint.pth')
    parser.add_argument('--filepath', dest='filepath', default='flowers/valid/2/image_05094.jpg') # use a deafault filepath to a primrose image 
    parser.add_argument('--category_names', dest='category_names', default='cat_to_name.json')
    parser.add_argument('--gpu', action='store', default='gpu')
    return parser.parse_args()

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    img_pil = Image.open(image) # use Image
   
    adjustments = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = adjustments(img_pil)
    
    return image

def predict(model,image_path):
  img = process_image(image_path).to('cuda')
  img = img.float().unsqueeze_(0)
  with torch.no_grad():
    output = model.forward(img).to('cuda')
  ps = torch.exp(output)
  probs, indices =ps.topk(5)
  probs = probs.cpu().numpy()[0]
  indices = indices.cpu().numpy()[0]

  idx_to_class = {v:k for k,v in model.class_to_idx.items()}
  classes = [idx_to_class[x] for x in indices ]

  return probs, classes

def main(): 
    args = parse_args()
    gpu = args.gpu
    model = load_checkpoint(args.checkpoint)
    cat_to_name = load_cat_names(args.category_names)
    
    img_path = args.filepath
    probs, classes = predict(model, img_path)
    labels = [cat_to_name[str(index)] for index in classes]
    probability = probs
    print('File selected: ' + img_path)
    
    print(labels)
    print(probability)
    
    i=0 # this prints out top k classes and probs as according to user 
    while i < len(labels):
        print("{} with a probability of {}".format(labels[i], probability[i]))
        i += 1 # cycle through

if __name__ == "__main__":
    main()