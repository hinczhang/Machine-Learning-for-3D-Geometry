## Small revisions
train.py:  
> Line 28: set the train data as `cifar10` (`imagenet` is too large)  
> Line 30: set the data dictionary as `./data` (Located in `./main/data`)  
> Line 74: set the default GPU as `cuda:0`  
> Line 246: add 'download = True', which allows us to download the `cifar10` dataset automatically  
> Line 493: change `.view()` to `.reshape()`: due to the error report during compling  


## Dependencies
All parts of the code assume that `torch` is of version 1.4 or higher. There might be instability issues on previous versions.

## Installation

You can build the repo through the following commands:
```
$ git clone https://github.com/alexandrosstergiou/SoftPool.git
$ cd SoftPool-master/pytorch
$ make install
--- (optional) ---
$ make test
```


## Usage

You can load any of the 1D, 2D or 3D variants after the installation with:

```python
import softpool_cuda
from SoftPool import soft_pool1d, SoftPool1d
from SoftPool import soft_pool2d, SoftPool2d
from SoftPool import soft_pool3d, SoftPool3d
```

+ `soft_poolxd`: Is a functional interface for SoftPool.
+ `SoftPoolxd`: Is the class-based version which created an object that can be referenced later in the code.

## Some parameters might be interesting
> `-d`: dataset. We can choose several dataset: 'imagenet','cifar10' and 'cifar100'  
> `-a`: existing architecture: 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'resnext101_32x4d', 'resnext101_64x4d', 'wide_resnet50_2', 'wide_resnet101_2', 'densenet121', 'densenet161', 'densenet169', 'densenet201', 'inception_v3'  
> `--epochs`: set epoch  
> `--use_softpool`: True or False: decide whether to use softpool  
> `-b`: batch size. Default is 256  
> `--lr`: learning rate  
> `--momentum`: momentum  
> `--wd`: weight decay  
> `-p`: control pirnting frequency  
> `--seed`: a seed number (int) for initialization  
> `--gpu`: GPU name. already set to `cuda:0`  

## ImageNet models

ImageNet weight can be downloaded from the following links:

|Network|link|
|:-----:|:--:|
| ResNet-18 | [link](https://drive.google.com/file/d/11me4z74Fp4FkGGv_WbMZRQxTr4YJxUHS/view?usp=sharing) |
| ResNet-34 | [link](https://drive.google.com/file/d/1-5O-r3hCJ7JSrrfVowrUZpaHcp7TcKKT/view?usp=sharing) |
| ResNet-50 | [link](https://drive.google.com/file/d/1HpBESqJ-QLO_O0pozgh1T3xp4n5MOQLU/view?usp=sharing) |
| ResNet-101 | [link](https://drive.google.com/file/d/1fng3DFm48W6h-qbFUk-IPZf9s8HsGbdw/view?usp=sharing) |
| ResNet-152 | [link](https://drive.google.com/file/d/1ejuMgP4DK9pFcVnu1TZo6TELPlrhHJC_/view?usp=sharing) |
| DenseNet-121 | [link](https://drive.google.com/file/d/1EXIbVI19JyEjgY75caZK2B2-gaxKTVpK/view?usp=sharing) |
| DenseNet-161 | [link](https://drive.google.com/file/d/18Qs9XUXNPSgBe46_0OGZIcpvdoFZfjU5/view?usp=sharing) |
| DenseNet-169 | [link](https://drive.google.com/file/d/1shFZV_AIZ6SQFQs-C0YThfpOfZH88hm7/view?usp=sharing) |
| ResNeXt-50_32x4d | [link](hhttps://drive.google.com/file/d/1-3sd8paTlqa1X8KGUy6B5Eehv791tbVH/view?usp=sharing) |
| ResNeXt-101_32x4d | [link](https://drive.google.com/file/d/1URDkwAPxDgcQzkYFlV_m-1T5RjZvzabo/view?usp=sharing) |
| wide-ResNet50 | [link](https://drive.google.com/file/d/1X3A6P0enEJYLeNmY0pUTXA26FEQB1qMe/view?usp=sharing) |

