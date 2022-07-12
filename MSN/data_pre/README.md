## Prepare your data
### Download ShapeNetCore.v1
Please download your dataset `wget https://shapenet.cs.stanford.edu/shapenet/obj-zip/ShapeNetCore.v1.zip`  
The file is about 30GB. Please remember to unzip your sub-data zip files in this dataset via [`unzip.py`](https://github.com/hinczhang/Machine-Learning-for-3D-Geometry/blob/main/MSN/data_pre/unzip.py)  
### Configure blender  
Please install your blender via 'snap', which means you should download snap first. Do remember use **blender 2.8**  
If you use the non_GUI edition (especially in the SSH case), please install `xvfb-run` to creat a virtual screen and run like this:  
`xvfb-run --auto-servernum --server-args="-screen 0 1600x1024x16" ...`  
### Render
Use `xvfb-run --auto-servernum --server-args="-screen 0 1600x1024x16" blender -b -P render_depth.py ./ShapeNetCore.v1/ train.list ./train 3` to run the code [`render_depth.py`](https://github.com/hinczhang/Machine-Learning-for-3D-Geometry/blob/main/MSN/data_pre/render_depth.py). It will create depth images.
