# Repository for Dynamic Feature Map Size, and Rank Adaptation Multilinear Transformation


## Usages

### Training
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --model dynamic_mtl_resnet18 --num_classes 10 --batch_size 32 --epochs 100 --learning_rate 0.001 --exp_name test_dynamic_resnet18_cifar10_5000tr_500test_100e --wandb_project DynamicRankMTL --lambda_sparse 0.0001
```

### Access visualization
