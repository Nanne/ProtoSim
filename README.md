# ProtoSim
Code and instructions accompanying Prototype-based Dataset Comparison (ICCV'23).

## Pre-trained Checkpoints

Pre-trained checkpoints for the PASSNET and Art datasets can be downloaded at: https://drive.google.com/drive/folders/1fQYi7vNgcrpVB98wmlD3oxsBJB-3em8R 

The pre-trained DINO checkpoint for the ViT backbone is available here: https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain_full_checkpoint.pth

## Usage

Train the model on PASSNET as follows:

```
python -m torch.distributed.run main.py --arch pvit_small --batch_size_per_gpu=128 --min_lr=0.00005 --use_fp16=False --epochs=20 --output_dir=checkpoints --num_prototypes=8192 --patch_size=16 --data_path=/path/to/first/dataset/,/path/to/second/dataset/
```

For linear evaluation on ImageNet:

```
python -m torch.distributed.run eval_protosim_linear.py --arch=pvit_small --num_prototypes=8192 --patch_size=16 --data_path=/path/to/ImageNet1K --pretrained_weights=checkpoints/checkpoint.pth --output_dir=linear_eval --epochs=20 --batch_size_per_gpu=256
```

### Dataset Comparison

To prepare the comparison you first need to extract statistics about the prototypes and their activations with:

```
python -m torch.distributed.run prepare_analysis.py --arch pvit_small --batch_size_per_gpu=256 --use_fp16=False --output_dir=checkpoints --num_prototypes=8192 --patch_size=16 --data_path=/path/to/imagenet/,/path/to/pass/
```

Subsequently, you can use the notebooks to guide your comparison of the two datasets by loading the `stats.npz` file generated with prepare_analysis.py. Within the notebooks various mechanisms for discovering   
