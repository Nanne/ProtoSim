import argparse
import os
import sys
import datetime
import time
import math
import json
import bisect
from pathlib import Path

import numpy as np
import scipy.sparse as sparse
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models

sys.path.insert(0, './dino')
import utils
import protosim as pvits
from protosim_utils import ReturnIndexWrapper, build_dataset
from vision_transformer import DINOHead

from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from main import get_args_parser

def analyse_proto(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    test_transform = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    dataset_gallery, nb_classes = build_dataset(data_path=args.data_path, transform=test_transform, indexed=True, index_labels=True)

    sampler = torch.utils.data.DistributedSampler(dataset_gallery, shuffle=False)
    data_loader_gallery = torch.utils.data.DataLoader(
        dataset_gallery,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # ============ building student and teacher networks ... ============
    if args.arch in pvits.__dict__.keys():
        student = pvits.__dict__[args.arch](
            num_prototypes=args.num_prototypes, 
            patch_size=args.patch_size,
        )
        embed_dim = student.embed_dim
    else:
        print(f"Unknow architecture: {args.arch}")

    student = utils.MultiCropWrapper(student, nn.Identity())

    # move networks to gpu
    student = student.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)

    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    print(f"Student built: it's a {args.arch} network.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
    )
    start_epoch = to_restore["epoch"]

    # get rid off Multi crop - just needed it for loading
    student.module = student.module.backbone

    #start_time = time.time()
    print("Calculating dataset stats")
    calc_stats(student, data_loader_gallery, nb_classes, args)

def calc_stats(model, gallery_loader, nb_classes, args):

    model.eval()

    proto_n = model.module.protoAT.num_prototypes

    class_counts = np.zeros((nb_classes, proto_n), dtype=np.int32) # class, prototype
    token_counts = np.zeros((197, proto_n), dtype=np.int32) # token, prototype
    instance_counts = sparse.dok_matrix((len(gallery_loader.dataset), proto_n), dtype=np.int32) # instance, prototype
    instance_class = np.zeros((len(gallery_loader.dataset), 2), dtype=np.int32)

    with torch.no_grad():
        for samples, index, label in tqdm(gallery_loader):

            samples = samples.cuda(non_blocking=True)

            sim = model(samples, return_attn=True)#.clone() # B, Prototypes, Tokens

            B, P, N = sim.shape
            #sim = sim.transpose(-2,-1).reshape(B*N, P)
            mv, mi = torch.max(sim, dim=1) # what is the max prototype for the batch item - and per token

            for xp, l, ix in zip(mi, label, index):
                dataset_idx = bisect.bisect_right(gallery_loader.dataset.cumulative_sizes, ix.item())
                instance_class[ix.item(), 1] = dataset_idx # which dataset
                instance_class[ix.item(), 0] = l.item() # label within that dataset

                for i, pp in enumerate(xp):
                    token_counts[i, pp.item()] += 1
                    class_counts[l.item(), pp.item()] += 1 
                    instance_counts[ix.item(), pp.item()] += 1

    np.savez(os.path.join(args.output_dir, "stats.npz"), token_counts=token_counts, class_counts=class_counts, instance_counts=instance_counts.tocoo(), instance_class=instance_class)

    print("Done calculating stats!")
    print("Most frequent prototypes for token 0:", np.argsort(-token_counts[0, :])[:5])
    print("Most frequent classes:", np.argsort(-class_counts.sum(axis=1)[:5]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Analyse ProtoSim', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    analyse_proto(args)
