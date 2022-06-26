## [MSN TEST (MSN: Morphing and Sampling Network)](https://github.com/Colin97/MSN-Point-Cloud-Completion)
*Author: [Zhang, Jiongyan](https://github.com/hinczhang)* <img src="https://img.shields.io/badge/张炅焱-ZhangJiongyan-red"/>  
### Configuration
Please install Pytorch and CUDA and use `pip` to install `Open3D`. All necessary dependencies can be seen [here](https://github.com/Colin97/MSN-Point-Cloud-Completion)  
For high-level Pytorch, please revise all `AT_CHECK` to `TORCH_CHECK` in **/MDS/MDS.cpp**.  
For installing:  
`
cd emd  
python3 setup.py install  
cd expansion_penalty  
python3 setup.py install  
cd MDS  
python3 setup.py install  
`
### Dictionary Structure
Please download data [here](https://drive.google.com/drive/folders/1X143kUwtRtoPFxNRvUk9LuPlsf1lLKI7)  
Please put `train_model` and `data` in the first class like this:  
![image](https://user-images.githubusercontent.com/70082542/174184757-d0ca6a0d-d99c-4836-8932-2999ff956e2a.png)  
Please unzip the `val.zip` and `complete.zip` within the `data`.
### Some changes <img src="https://img.shields.io/badge/IMPORTANT-!!!-red"/>
- We do not need to use visdom package.  
- The visdom content in the `train.py` and `val.py` have already been deleted.  
- For convenience, in `dataset.py`, we set the train dataset as the validation dataset.
### RUN IT!
Use `train.py` to train and use `val.py` to validate. Actually two of them use the same dataset (*val*)  
Please notice: to reduce the batch size, it is suggested that you could set `--num_points` as the multiples of 1024, like:  
`train.py --num_points 2048`  
