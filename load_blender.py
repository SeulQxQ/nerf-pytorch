import os
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2

# 平移矩阵
trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

# 绕x轴旋转的旋转矩阵
rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

# 绕y轴旋转的旋转矩阵
rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()

# 相机坐标系到世界坐标系的变换矩阵
'''
phi: 方位角
theta: 仰角
radius: 距离球心的距离
'''
def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


def load_blender_data(basedir, half_res=False, testskip=1):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    """
    transforms_train.json:{
        camera_angle_x: 水平视场角, 用于求焦距f;
        file_path: 图片路径;
        transform_matrix: 4x4的矩阵, 用于求相机坐标系到世界坐标系的变换矩阵;
        rotation: (旋转角度, 用于求相机坐标系到世界坐标系的变换矩阵) 也许, 代码没有用到;
    """
    
    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s=='train' or testskip==0: # 加载所有图片
            skip = 1
        else:
            skip = testskip
            
            
        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix'])) #pose: 相机坐标系到世界坐标系的变换矩阵
        imgs = (np.array(imgs) / 255.).astype(np.float32) # 归一化 keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
    
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    
    imgs = np.concatenate(all_imgs, 0)              #整合所有图片
    poses = np.concatenate(all_poses, 0)            #整合所有pose
    
    H, W = imgs[0].shape[:2]                        #图片的高和宽
    camera_angle_x = float(meta['camera_angle_x'])  #camera_angle_x: 水平视场角β，用于计算焦距
    focal = .5 * W / np.tan(.5 * camera_angle_x)   #焦距：f = W / (2*tan(β/2))
            
            
    '''
    np.linspace(-180,180,40+1): 生成40个角度, -180到180, 间隔为360/40 = 9, [:-1]去掉最后一个角度。
    render_poses: 制作出来的40个测试poses, 用于测试 shape: (40, 4, 4)
    '''
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    
    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            # interpolation=cv2.INTER_AREA: 用于缩小图片时进行图像重采样。
            # 使用像素区域关系进行重采样，是一种计算代价较小且输出图像质量较高的插值方法之一。
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)  # 重设图像大小，图像缩小
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

        
    return imgs, poses, render_poses, [H, W, focal], i_split


