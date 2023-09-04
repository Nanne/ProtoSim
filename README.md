# ProtoSim
Code and instructions accompanying Protoype-based Dataset Comparison (ICCV'23).

## Pre-trained Checkpoints

Pre-trained checkpoints for the PASSNET and Art datasets can be downloaded at: https://drive.google.com/drive/folders/1fQYi7vNgcrpVB98wmlD3oxsBJB-3em8R 

The pre-trained DINO checkpoint for the ViT backbone is available here: https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain_full_checkpoint.pth

## Usage

Train the model on PASSNET as follows:

```
python -m torch.distributed.run --nproc_per_node=1 main.py --arch pvit_small --batch_size_per_gpu=128 --min_lr=0.00005 --use_fp16=False --out_dim=65536 --epochs=20 --output_dir=checkpoints_passnet --num_prototypes=8192 --patch_size=16 --data-set=PASSNET --data_path=/path/to/data/parent/dir/
```

Extract the token and prototype statistics:

```
python -m torch.distributed.run --nproc_per_node=1 visualise_protosim.py --do_stats=True --arch pvit_small --batch_size_per_gpu=256 --use_fp16=False  --output_dir=checkpoints_passnet --num_prototypes=8192 --patch_size=16 --data-set=PASSNET --data_path=/path/to/data/parent/dir/
```

Use the notebooks to guide your comparison of the two datasets.

## Dataset structure

TBD
