# Dump SUNRGBD data to our format
# for each sample, we have RGB image, 2d boxes.
# point cloud (in camera coordinate), calibration and 3d boxes.
#
# Extract using V1 labels.
#
# Author: Charles R. Qi
import os
from scipy.io import loadmat

# V1 2D&3D BB and Seg masks
data = loadmat('../OFFICIAL_SUNRGBD/SUNRGBDtoolbox/Metadata/SUNRGBDMeta.mat', squeeze_me=True, struct_as_record=False)['SUNRGBDMeta']

# Create folders
det_label_folder = '../sunrgbd_trainval/label_v1/'
os.makedirs(det_label_folder, exist_ok=True)
# Read
for imageId in range(data.shape[0]):
    print(imageId)
    try:
        img_data = data[imageId]
        depthpath = img_data.depthpath[16:]
        depthpath = '../OFFICIAL_SUNRGBD/SUNRGBD' + depthpath
        rgbpath = img_data.rgbpath[16:]
        rgbpath = '../OFFICIAL_SUNRGBD/SUNRGBD' + rgbpath

        # Write 2D and 3D box label
        data2d = img_data
        txt_filename = f'{imageId:06}.txt'
        with open(os.path.join(det_label_folder, txt_filename), 'w') as fid:
            for j in range(len(img_data.groundtruth3DBB)):
                centroid = img_data.groundtruth3DBB[j].centroid
                classname = img_data.groundtruth3DBB[j].classname
                orientation = img_data.groundtruth3DBB[j].orientation
                coeffs = abs(img_data.groundtruth3DBB[j].coeffs)
                box2d = data2d.groundtruth2DBB[j].gtBb2D
                fid.write(f'{classname} {box2d[0]} {box2d[1]} {box2d[2]} {box2d[3]} {centroid[0]} {centroid[1]} {centroid[2]} {coeffs[0]} {coeffs[1]} {coeffs[2]} {orientation[0]} {orientation[1]}\n')
    except:
        pass
