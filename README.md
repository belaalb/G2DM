# Adversarial target-invariant domain generalization

## Requirements
```
Python >= 3.6
pytorch >= 1.2
torchvision >= 0.4.1
Scikit-learn >= 0.19
h5py
tqdm
pandas
seaborn
```
To Install requirements:

```
  pip install -r requirements.txt
```

## Download PACS
- Downlaod the original splits from the folder "train val splits and h5py files pre-read" found at https://drive.google.com/drive/folders/0B6x7gtvErXgfUU1WcGY5SzdwZVk
- Move it to ./data/pacs/prepared_data
and then


### Prepare hdf files for PACS
```
cd data/pacs
python prep_hdf.py --train-val-test train
python prep_hdf.py --train-val-test val
python prep_hdf.py --train-val-test test

```

## Download VLCS
Download it from http://www.mediafire.com/file/7yv132lgn1v267r/vlcs.tar.gz/file, extract, move it to ./data/vlcs/ prepared_data/ and then


## Download pre-trained AlexNet 
Download it from https://drive.google.com/file/d/1wUJTH1Joq2KAgrUDeKJghP1Wf7Q9w4z-/view?usp=sharing and move it to ./
Or

```
python download_alexnet.py
```

## Table 1
Example considering Caltech101 as target domain.

### Running ours [Done]
```
cd vlcs-ours
python train.py --lr-task 0.001 --lr-domain 0.005 --l2 0.005 --smoothing 0.2 --lr-threshold 0.0001 --factor 0.3 --alpha 0.8 --rp-size 3500 --patience 60 --warmup-its 300 --source1 PASCAL --source2 LABELME --source3 SUN --target CALTECH
```

### Running ERM [Done]
```
cd vlcs-ours
python baseline_train.py --lr 0.001 --l2 0.00001 --patience 120 --source1 PASCAL --source2 LABELME --source3 SUN --target CALTECH
```

### Running IRM 
```
cd IRM-vlcs
python train.py --lr 0.0004898536566546834 --l2 0.00221589136 --penalty_weight 91257.18613115903 --penalty_anneal_epochs 78 --source1 PASCAL --source2 LABELME --source3 SUN --target CALTECH
```

## Table 2 
Example considering SUN09 and Caltech-101 as target domains. 

### Running ours
```
cd vlcs-2sources
python train.py --lr-task 0.001 --lr-domain 0.005 --l2 0.005 --smoothing 0.2 --lr-threshold 0.0001 --factor 0.3 --alpha 0.8 --rp-size 3500 --patience 60 --warmup-its 300 --source1 PASCAL --source2 LABELME --target1 SUN --target2 CALTECH
```

### Running ERM
```
cd vlcs-2sources
python baseline_train.py --lr 0.001 --l2 0.00001 --patience 120 --source1 PASCAL --source2 LABELME --target1 SUN --target2 CALTECH
```

## Table 3 
Example considering art painting as target domain.

### Running ours
```
cd pacs-ours
python train.py --lr-task 0.01 --lr-domain 0.0005 --l2 0.0005 --smoothing 0.2 --lr-threshold 0.00001 --factor 0.5 --alpha 0.8 --rp-size 1000 --patience 80 --warmup-its 300 --source1 photo --source2 cartoon --source3 sketch --target artpainting
```

### Running ERM
```
cd pacs-ours
python baseline_train.py --lr 0.001 --l2 0.0001 --momentum 0.9 --patience 120 --source1 photo --source2 cartoon --source3 sketch --target artpainting
```

### Running IRM
```
cd IRM-pacs
python train.py --lr 0.0004898536566546834 --l2 0.00221589136 --penalty_weight 91257.18613115903 --penalty_anneal_epochs 78 --source1 photo --source2 cartoon --source3 sketch --target artpainting
```

## Figure 3
```
cd pacs-ours
python h_divergence.py --batch-size 500 --encoder-path path-to-trained-model --dg-type ['erm', 'adversarial'] 
```

## Table 4 [Done]
- For running AlexNet experiments, use same code from Table 3 experiments.
- For running Jigsaw, see authors original implementation at https://github.com/fmcarlucci/JigenDG.
- For running ResNet experiments:
```
cd pacs-resnet
python train.py --train-model resnet18 --smoothing 0 --train-mode hv --nadir-slack 2.5 --alpha 0.8 --lr-task 0.01 --lr-domain 0.005 --patience 20 --l2 0.0005 --rp-size 512 --source1 photo --source2 cartoon --source3 sketch --target art_painting
```
