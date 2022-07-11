## Prepare your data
### Download ShapeNetCore.v1
Please download your dataset `wget https://shapenet.cs.stanford.edu/shapenet/obj-zip/ShapeNetCore.v1.zip`  
The file is about 30GB. Please remember to unzip your sub-data zip files in this dataset via [`unzip.py`](https://github.com/hinczhang/Machine-Learning-for-3D-Geometry/blob/main/MSN/data_pre/unzip.py)  
### Configure blender  
Please install your blender via 'snap'. If you use the non_GUI edition, please install `xvfb-run` to creat a virtual screen and run like this:  
`xvfb-run --auto-servernum --server-args="-screen 0 1600x1024x16" ...`  
### Render
Use 
