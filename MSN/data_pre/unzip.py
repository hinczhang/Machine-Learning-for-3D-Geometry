data_dir = "../data/ShapeNetCore.v1"
import os
def getFlist(path):
    for root, dirs, files in os.walk(path):
        print('root_dir:', root)  #当前路径
        print('sub_dirs:', dirs)   #子文件夹
        print('files:', files)     #文件名称，返回list类型
    return files
file_name = getFlist(data_dir)
print(file_name)
zip_files = set()

from zipfile import ZipFile
for name in file_name:
  if(name.split(".")[-1]=="zip"):
    zip_files.add(name)
    with ZipFile(data_dir + '/' + 'name', 'r') as zipObj:
      zipObj.extractall()

for item in list(zip_files):
  print(item)
  os.remove(data_dir+'/'+item)
