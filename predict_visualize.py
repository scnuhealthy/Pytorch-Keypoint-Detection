import datetime
import os
import time

import torch
import torch.utils.data
from torch import nn
import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn

from torchvision import transforms

from coco_utils import get_coco, get_coco_kp

from group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from engine import train_one_epoch, evaluate

import utils
import transforms as T

from PIL import Image
from plot import plot_poses
import numpy as np
def get_dataset(name, image_set, transform):
    paths = {
        "coco": ('/home/hzj/data/COCO2017/', get_coco, 91),
        "coco_kp": ('/home/hzj/data/COCO2017/', get_coco_kp, 2)
    }
    p, ds_fn, num_classes = paths[name]

    ds = ds_fn(p, image_set=image_set, transforms=transform)
    return ds, num_classes


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def main():

    device = torch.device("cuda:0")

    # Data loading code
    print("Loading data")

    #dataset, num_classes = get_dataset(args.dataset, "train", get_transform(train=True))
    dataset_test, num_classes = get_dataset("coco_kp", "val", get_transform(train=False))

    print("Creating data loaders")

    #train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)


    #train_batch_sampler = torch.utils.data.BatchSampler(
    #   train_sampler, args.batch_size, drop_last=True)

    #data_loader = torch.utils.data.DataLoader(
    #    dataset, batch_sampler=train_batch_sampler, num_workers=args.workers,
    #    collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=4,
        collate_fn=utils.collate_fn)

    print("Creating model")
    model = torchvision.models.detection.__dict__['keypointrcnn_resnet50_fpn'](num_classes=num_classes,
                                                              pretrained=True)
    model.to(device)

    #checkpoint = torch.load(args.resume, map_location='cpu')
    #model_without_ddp.load_state_dict(checkpoint['model'])
    #optimizer.load_state_dict(checkpoint['optimizer'])
    #lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    model.eval()
    
    detect_threshold = 0.7
    keypoint_score_threshold = 2
    with torch.no_grad():
        for i in range(20):
            img,_ = dataset_test[i]
            prediction = model([img.to(device)])
            keypoints = prediction[0]['keypoints'].cpu().numpy()
            scores = prediction[0]['scores'].cpu().numpy()
            keypoints_scores = prediction[0]['keypoints_scores'].cpu().numpy()
            idx = np.where(scores>detect_threshold)
            keypoints = keypoints[idx]
            keypoints_scores = keypoints_scores[idx]
            for j in range(keypoints.shape[0]):
                for num in range(17):
                    if keypoints_scores[j][num]<keypoint_score_threshold:
                        keypoints[j][num]=[0,0,0]
            img = img.mul(255).permute(1, 2, 0).byte().numpy()
            plot_poses(img,keypoints,save_name='./result/'+str(i)+'.jpg')

    
if __name__ == "__main__":

    main()
