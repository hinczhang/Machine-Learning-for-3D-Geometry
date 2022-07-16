## Prepare your data
*The related codes come from [here](https://github.com/wentaoyuan/pcn/tree/master/render)*
### Download ShapeNetCore.v1
Please download your dataset `wget https://shapenet.cs.stanford.edu/shapenet/obj-zip/ShapeNetCore.v1.zip`  
The file is about 30GB. Please remember to unzip your sub-data zip files in this dataset via [`unzip.py`](https://github.com/hinczhang/Machine-Learning-for-3D-Geometry/blob/main/MSN/data_pre/unzip.py)  
### Configure blender  
Please install your blender via 'snap', which means you should download snap first:  
`sudo apt install snapd`  
Do remember use **blender 2.8**:  
`sudo snap install blender --channel=2.8/stable --classic`  
If you use the non_GUI edition (especially in the SSH case), please install `xvfb-run` to creat a virtual screen and run like this:  
`xvfb-run --auto-servernum --server-args="-screen 0 1600x1024x16" ...`  
### Render
Use `xvfb-run --auto-servernum --server-args="-screen 0 1600x1024x16" blender -b -P render_depth.py [dataset_dir] [model_list] [output_list] [scan_number]` to run the code [`render_depth.py`](https://github.com/hinczhang/Machine-Learning-for-3D-Geometry/blob/main/MSN/data_pre/render_depth.py). It will create depth images.
## Generate final data
Use `python process_exr.py [model_list] [intrinsics_file] [output_dir] [scam_number]` to do it.  
Please install `OpenEXR` first before run the code:
> sudo apt-get update -y  
> sudo apt-get install libopenexr-dev  
> sudo apt-get install openexr  
> pip install OpenEXR  
  
Some source codes have already been revised to fit the new OpenEXR API.
