import os
import shutil
from PIL import Image
import numpy as np
import cv2
from scipy.io import loadmat

def read_3d_pts_general(depthInpaint, K, depthInpaintsize, imageName):
    cx, cy = K[0, 2], K[1, 2]
    fx, fy = K[0, 0], K[1, 1]

    invalid = depthInpaint == 0
    if imageName:
        im = Image.open(imageName)
        rgb = np.array(im).astype(np.float32) / 255.0
    else:
        rgb = np.dstack((
            np.zeros((depthInpaintsize[0], depthInpaintsize[1])),
            np.ones((depthInpaintsize[0], depthInpaintsize[1])),
            np.zeros((depthInpaintsize[0], depthInpaintsize[1]))
        ))

    rgb = rgb.reshape(-1, 3)

    # 3D points
    x, y = np.meshgrid(np.arange(1, depthInpaintsize[1] + 1), np.arange(1, depthInpaintsize[0] + 1))
    x3 = (x - cx) * depthInpaint * 1 / fx
    y3 = (y - cy) * depthInpaint * 1 / fy
    z3 = depthInpaint

    points3dMatrix = np.dstack((x3, z3, -y3))
    points3dMatrix[invalid, :] = np.nan

    points3d = np.column_stack((x3.ravel(), z3.ravel(), -y3.ravel()))
    points3d[invalid.ravel(), :] = np.nan
    return rgb, points3d, points3dMatrix

def read3dPoints(data):
    depthVis = cv2.imread(data.depthpath, cv2.IMREAD_UNCHANGED)
    imsize = depthVis.shape
    depthInpaint = np.bitwise_or(depthVis >> 3, depthVis << 16-3)
    depthInpaint = depthInpaint.astype(np.float32) / 1000.0
    depthInpaint[depthInpaint > 8] = 8
    rgb, points3d, _ = read_3d_pts_general(depthInpaint, data.K, depthInpaint.shape, data.rgbpath)
    points3d = np.dot(data.Rtilt, points3d.T).T
    #print(points3d.shape)
    return rgb, points3d, depthInpaint, imsize

# V2 3DBB annotations (overwrites SUNRGBDMeta)
meta_3dbb = loadmat('../OFFICIAL_SUNRGBD/SUNRGBDMeta3DBB_v2.mat', squeeze_me=True, struct_as_record=False)['SUNRGBDMeta']
meta_2dbb = loadmat('../OFFICIAL_SUNRGBD/SUNRGBDMeta2DBB_v2.mat', squeeze_me=True, struct_as_record=False)['SUNRGBDMeta2DBB']

# Create folders
depth_folder = '../sunrgbd_trainval/depth/'
image_folder = '../sunrgbd_trainval/image/'
#calib_folder = '../sunrgbd_trainval/calib/'
det_label_folder = '../sunrgbd_trainval/label/'
seg_label13_folder = '../sunrgbd_trainval/seg_label13/'
seg_label37_folder = '../sunrgbd_trainval/seg_label37/'
os.makedirs(depth_folder, exist_ok=True)
os.makedirs(image_folder, exist_ok=True)
#os.makedirs(calib_folder, exist_ok=True)
os.makedirs(det_label_folder, exist_ok=True)
os.makedirs(seg_label13_folder, exist_ok=True)
os.makedirs(seg_label37_folder, exist_ok=True)
# Read
er = 0
for imageId in range(len(meta_2dbb)):
    print(imageId)
    try:
        data = meta_3dbb[imageId]
        data.depthpath = data.depthpath[16:]
        data.depthpath = '../OFFICIAL_SUNRGBD' +  os.path.dirname(data.depthpath)+ '_bfx/' + os.path.basename(data.depthpath)
        data.rgbpath = data.rgbpath[16:]
        data.rgbpath = '../OFFICIAL_SUNRGBD' + data.rgbpath
        """
        # 怪しい
        # Write point cloud in depth map
        rgb, points3d, _, imsize = read3dPoints(data)
        mask = np.isnan(points3d[:, 0])
        rgb = rgb[~mask]
        points3d = points3d[~mask]
        points3d_rgb = np.concatenate((points3d, rgb), axis=1)
        # MAT files are 3x smaller than TXT files. In Python we can use
        # scipy.io.loadmat('xxx.mat')['points3d_rgb'] to load the data.
        mat_filename = f'{imageId:06}'
        np.save(os.path.join(depth_folder, mat_filename), points3d_rgb)
        # Write calibration
        calib_filename = f'{imageId:06}.txt'
        calib = np.concatenate((data.Rtilt.flatten(), data.K.flatten()), axis=None)
        np.savetxt(os.path.join(calib_folder, calib_filename), calib, delimiter=' ')
        """
        if imageId < 5050:
            label13 = '../OFFICIAL_SUNRGBD/sunrgbd-meta-data-master/test13labels/img13labels-' + str(imageId+1).zfill(6) + ".png"
        else:
            label13 = '../OFFICIAL_SUNRGBD/sunrgbd-meta-data-master/train13labels/img13labels-' + str(imageId-5050+1).zfill(6) + ".png"
        label37 = '../OFFICIAL_SUNRGBD/sunrgbd-meta-data-master/sunrgbd_train_test_labels/img-' + str(imageId+1).zfill(6) + ".png"
        # Write rgb & depth images
        shutil.copy(data.rgbpath, os.path.join(image_folder, f'{imageId:06}.jpg'))
        shutil.copy(data.depthpath, os.path.join(depth_folder, f'{imageId:06}.png'))
        shutil.copy(label13, os.path.join(seg_label13_folder, f'{imageId:06}.png'))
        shutil.copy(label37, os.path.join(seg_label37_folder, f'{imageId:06}.png'))
        # Write 2D and 3D box label
        data2d = meta_2dbb[imageId]
        fid = open(os.path.join(det_label_folder, f'{imageId:06}.txt'), 'w')
        for j in range(len(data.groundtruth3DBB)):
            centroid = data.groundtruth3DBB[j].centroid
            classname = data.groundtruth3DBB[j].classname
            orientation = data.groundtruth3DBB[j].orientation
            coeffs = abs(data.groundtruth3DBB[j].coeffs)
            box2d = data2d.groundtruth2DBB[j].gtBb2D
            assert data2d.groundtruth2DBB[j].classname == classname
            fid.write(f'{classname} {box2d[0]} {box2d[1]} {box2d[2]} {box2d[3]} {centroid[0]} {centroid[1]} {centroid[2]} {coeffs[0]} {coeffs[1]} {coeffs[2]} {orientation[0]} {orientation[1]}\n')
        fid.close()
    except:
        er += 1
        print("ERROR")
        pass
print("pass num", er)

