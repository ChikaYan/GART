# Use InsAV to process in the wild video, then load it

from torch.utils.data import Dataset
import logging
import json
import os
import numpy as np
from os.path import join
import os.path as osp
import pickle
import numpy as np
import torch.utils.data as data
from PIL import Image
import imageio
import cv2
from plyfile import PlyData
from tqdm import tqdm
from transforms3d.euler import euler2mat
import glob
import copy
from smplx.body_models import SMPLX
import torch
import math
from typing import List
from typing import NamedTuple
from pathlib import Path



def get_bound_2d_mask(bounds, K, pose, H, W):
    corners_3d = get_bound_corners(bounds)
    corners_2d = project(corners_3d, K, pose)
    corners_2d = np.round(corners_2d).astype(int)
    corners_2d[:,0] = np.clip(corners_2d[:,0], 0, W)
    corners_2d[:,1] = np.clip(corners_2d[:,1], 0, H)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 3, 2, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[4, 5, 7, 6, 4]]], 1) # 4,5,7,6,4
    cv2.fillPoly(mask, [corners_2d[[0, 1, 5, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[2, 3, 7, 6, 2]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 2, 6, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[1, 3, 7, 5, 1]]], 1)
    return mask

def get_bound_corners(bounds):
    min_x, min_y, min_z = bounds[0]
    max_x, max_y, max_z = bounds[1]
    corners_3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corners_3d

def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy

def load_smpl_param(path):
    smpl_params = dict(np.load(str(path)))
    if "thetas" in smpl_params:
        smpl_params["body_pose"] = smpl_params["thetas"][..., 3:]
        smpl_params["global_orient"] = smpl_params["thetas"][..., :3]
    return {
        "betas": smpl_params["betas"].astype(np.float32).reshape(1, 10),
        "body_pose": smpl_params["body_pose"].astype(np.float32),
        "global_orient": smpl_params["global_orient"].astype(np.float32),
        "transl": smpl_params["transl"].astype(np.float32),
    }

def get_camera_extrinsics_zju_mocap_refine(view_index, val=False, camera_view_num=36):
    def norm_np_arr(arr):
        return arr / np.linalg.norm(arr)

    def lookat(eye, at, up):
        zaxis = norm_np_arr(at - eye)
        xaxis = norm_np_arr(np.cross(zaxis, up))
        yaxis = np.cross(xaxis, zaxis)
        _viewMatrix = np.array([
            [xaxis[0], xaxis[1], xaxis[2], -np.dot(xaxis, eye)],
            [yaxis[0], yaxis[1], yaxis[2], -np.dot(yaxis, eye)],
            [-zaxis[0], -zaxis[1], -zaxis[2], np.dot(zaxis, eye)],
            [0       , 0       , 0       , 1     ]
        ])
        return _viewMatrix
    
    def fix_eye(phi, theta):
        camera_distance = 3
        return np.array([
            camera_distance * np.sin(theta) * np.cos(phi),
            camera_distance * np.sin(theta) * np.sin(phi),
            camera_distance * np.cos(theta)
        ])

    if val:
        eye = fix_eye(np.pi + 2 * np.pi * view_index / camera_view_num + 1e-6, np.pi/2 + np.pi/12 + 1e-6).astype(np.float32) + np.array([0, 0, -0.8]).astype(np.float32)
        at = np.array([0, 0, -0.8]).astype(np.float32)

        extrinsics = lookat(eye, at, np.array([0, 0, -1])).astype(np.float32)
    return extrinsics

class CameraInfo(NamedTuple):
    uid: int
    pose_id: int
    R: np.array
    T: np.array
    K: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    bkgd_mask: np.array
    bound_mask: np.array
    width: int
    height: int
    smpl_param: dict
    world_vertex: np.array
    world_bound: np.array
    big_pose_smpl_param: dict
    big_pose_world_vertex: np.array
    big_pose_world_bound: np.array


class Dataset(Dataset):
    # from instant avatar
    def __init__(
        self,
        data_root="data/people_snapshot_public_instant_avatar_processed",
        video_name="male-3-casual",
        split="train",
        image_zoom_ratio=0.5,
        start_end_skip=None,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        path = data_root
        self.video_name = video_name

        # if start_end_skip is not None:
        #     start, end, skip = start_end_skip
        # else:
        #     # raise NotImplementedError("Must specify, check the end+1")
        #     if split == "train":
        #         start, end, skip = 0, 41+1, 1
        #     elif split == "val":
        #         start, end, skip = 41, 42+1, 1
        #     elif split == "test":
        #         start, end, skip = 42, 51+1, 1


        # all_views = os.listdir(os.path.join(data_root, 'train')) + \
        #                 os.listdir(os.path.join(data_root, 'test'))
        
        all_views = [os.path.join(data_root, 'train', p) for p in os.listdir(os.path.join(data_root, 'train'))] + \
                        [os.path.join(data_root, 'test', p) for p in os.listdir(os.path.join(data_root, 'test'))]
        all_views = sorted(all_views)
        train_view = [all_views[0]]
        # train_view = [6]
        test_view = copy.deepcopy(all_views)
        test_view.remove(train_view[0])

        if split == 'train':
            views = train_view
        elif split == "val":
            views = test_view[:1]
        elif split == "test":
            views = test_view

        print(f"{split} views are: {views}")

        pose_start = 0
        if split == 'train':
            pose_interval = 1
            num_img = os.listdir(osp.join(data_root, views[0], 'render', 'image'))
            pose_num = len(num_img)
        elif split == 'test':
            pose_start = 0
            pose_interval = 5
            pose_num = 20


        self.image_zoom_ratio = image_zoom_ratio

        cams_dict = {}
        ims_dict = {}
        ims_name_dict = {}

        for view in views:
            root = osp.join(data_root, view)
            cam_numpy = np.load(os.path.join(view, 'render', 'cameras.npz'), allow_pickle=True)

            extr = cam_numpy['extrinsic']
            intr = cam_numpy['intrinsic']
            cams = {
                'K': intr,
                'R': extr[:,:3,:3],
                'T': extr[:,:3,3:4],
            }
            cams_dict[view] = cams

            ims_list = sorted(os.listdir(os.path.join(root, 'render/image')))
            ims = np.array([
                np.array(os.path.join(root, 'render/image', im))
                for im in ims_list[pose_start:pose_start + pose_num * pose_interval][::pose_interval]
            ])
            pose_num = len(ims)
            ims_dict[view] = ims

            img_name_list = sorted(os.listdir(os.path.join(root, 'render', 'image')))
            # img_name_list = [img_name
            #                  for img_name in img_name_list[pose_start:pose_start + pose_num * pose_interval][::pose_interval]]
            ims_name_dict[view] = img_name_list

        SMPLX_PKL_PATH = "/home/tw554/GART/models/smplx"
        smplx_zoo = {
            'male': SMPLX(model_path=f'{SMPLX_PKL_PATH}/SMPLX_MALE.pkl', ext='pkl',
                            use_face_contour=True, flat_hand_mean=False, use_pca=False,
                            num_betas=10, num_expression_coeffs=10),
            'female': SMPLX(model_path=f'{SMPLX_PKL_PATH}/SMPLX_FEMALE.pkl', ext='pkl',
                        use_face_contour=True, flat_hand_mean=False, use_pca=False,
                        num_betas=10, num_expression_coeffs=10),
        }
        with open(os.path.join(data_root, 'gender.txt'), 'r') as f:
            gender = f.readline()
        gender = gender.strip()
        smplx_model = smplx_zoo[gender]
        # '/home/hh29499/Datasets/X_Human/00016/mean_shape_smplx.npy'
        smpl_param_path = os.path.join(data_root, 'mean_shape_smplx.npy')
        template_shape = np.load(smpl_param_path)
        # SMPL in canonical space
        big_pose_smpl_param = {}
        big_pose_smpl_param['R'] = np.eye(3).astype(np.float32)
        big_pose_smpl_param['Th'] = np.zeros((1, 3)).astype(np.float32)
        big_pose_smpl_param['global_orient'] = np.zeros((1, 3)).astype(np.float32)
        big_pose_smpl_param['betas'] = np.zeros((1, 10)).astype(np.float32)
        big_pose_smpl_param['betas'] = template_shape[None].astype(np.float32)
        big_pose_smpl_param['body_pose'] = np.zeros((1, 63)).astype(np.float32)
        big_pose_smpl_param['jaw_pose'] = np.zeros((1, 3)).astype(np.float32)
        big_pose_smpl_param['left_hand_pose'] = np.zeros((1, 45)).astype(np.float32)
        big_pose_smpl_param['right_hand_pose'] = np.zeros((1, 45)).astype(np.float32)
        big_pose_smpl_param['leye_pose'] = np.zeros((1, 3)).astype(np.float32)
        big_pose_smpl_param['reye_pose'] = np.zeros((1, 3)).astype(np.float32)
        big_pose_smpl_param['expression'] = np.zeros((1, 10)).astype(np.float32)
        big_pose_smpl_param['transl'] = np.zeros((1, 3)).astype(np.float32)
        big_pose_smpl_param['body_pose'][0, 2] = 45 / 180 * np.array(np.pi)
        big_pose_smpl_param['body_pose'][0, 5] = -45 / 180 * np.array(np.pi)
        big_pose_smpl_param['body_pose'][0, 20] = -30 / 180 * np.array(np.pi)
        big_pose_smpl_param['body_pose'][0, 23] = 30 / 180 * np.array(np.pi)
        big_pose_smpl_param_tensor = {}
        for key in big_pose_smpl_param.keys():
            big_pose_smpl_param_tensor[key] = torch.from_numpy(big_pose_smpl_param[key])
        body_model_output = smplx_model(
            global_orient=big_pose_smpl_param_tensor['global_orient'],
            betas=big_pose_smpl_param_tensor['betas'],
            body_pose=big_pose_smpl_param_tensor['body_pose'],
            jaw_pose=big_pose_smpl_param_tensor['jaw_pose'],
            left_hand_pose=big_pose_smpl_param_tensor['left_hand_pose'],
            right_hand_pose=big_pose_smpl_param_tensor['right_hand_pose'],
            leye_pose=big_pose_smpl_param_tensor['leye_pose'],
            reye_pose=big_pose_smpl_param_tensor['reye_pose'],
            expression=big_pose_smpl_param_tensor['expression'],
            transl=big_pose_smpl_param_tensor['transl'],
            return_full_pose=True,
        )
        big_pose_smpl_param['poses'] = body_model_output.full_pose.detach()
        big_pose_smpl_param['shapes'] = np.concatenate([big_pose_smpl_param['betas'], big_pose_smpl_param['expression']],
                                                    axis=-1)
        big_pose_xyz = np.array(body_model_output.vertices.detach()).reshape(-1, 3).astype(np.float32)




        big_pose_min_xyz = np.min(big_pose_xyz, axis=0)
        big_pose_max_xyz = np.max(big_pose_xyz, axis=0)
        big_pose_min_xyz -= 0.05
        big_pose_max_xyz += 0.05
        big_pose_world_bound = np.stack([big_pose_min_xyz, big_pose_max_xyz], axis=0)


        idx = 0
        cam_infos = []
        novel_view_vis = False
        white_background = False
        for pose_index in range(pose_num):
            # pose_index = 100
            for view in views:

                if novel_view_vis:
                    view_index_look_at = view
                    view = 0

                # Load image, mask, K, D, R, T
                # try:
                image_path = os.path.join(path, ims_dict[view][pose_index].replace('\\', '/'))
                # except:
                #     a = 1
                #     print('error')
                if 'train' in image_path:
                    split_flag = 'train'
                else:
                    split_flag = 'test'
                image_name = ims_dict[view][pose_index].split('.')[0]
                image = np.array(imageio.imread(image_path).astype(np.float32) / 255.)

                # msk_path = image_path.replace('image', 'mask').replace('jpg', 'png')
                # msk = imageio.imread(msk_path)
                # msk = (msk != 0).astype(np.uint8)

                msk_path = image_path.replace('image', 'mask_new').replace('jpg', 'png')
                msk_alpha = imageio.imread(msk_path)
                msk_alpha = msk_alpha / 255.
                # msk = (msk_alpha != 0).astype(np.uint8)
                msk = (msk_alpha > 0.3).astype(np.uint8)

                if not novel_view_vis:
                    K = np.array(cams_dict[view]['K'])
                    # D = np.array(cams['D'])
                    # R = np.array(cams_dict[view_index]['R'][int(image_name.split('_')[-1])-1])
                    # T = np.array(cams_dict[view_index]['T'][int(image_name.split('_')[-1])-1]) #/ 1000.
                    R = np.array(cams_dict[view]['R'][ims_name_dict[view].index(image_name.split('/')[-1]+'.png')])
                    T = np.array(cams_dict[view]['T'][ims_name_dict[view].index(image_name.split('/')[-1]+'.png')])  # / 1000.
                    # image = cv2.undistort(image, K)
                    # msk = cv2.undistort(msk, K)
                else:
                    pose = np.matmul(np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]),
                                    get_camera_extrinsics_zju_mocap_refine(view_index_look_at, val=True))
                    R = pose[:3, :3]
                    T = pose[:3, 3].reshape(-1, 1)
                    cam_ind = cam_inds[pose_index][view]
                    K = np.array(cams['K'][cam_ind])

                # image[msk == 0] = 1 if white_background else 0
                image = image * msk_alpha[..., None]

                # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
                w2c = np.eye(4)
                w2c[:3, :3] = R
                w2c[:3, 3:4] = T

                # get the world-to-camera transform and set R, T
                R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
                T = w2c[:3, 3]

                # Reduce the image resolution by ratio, then remove the back ground
                ratio = image_zoom_ratio
                # ratio = 1
                if ratio != 1.:
                    H, W = int(image.shape[0] * ratio), int(image.shape[1] * ratio)
                    image = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)
                    msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
                    K[:2] = K[:2] * ratio

                image = Image.fromarray(np.array(image * 255.0, dtype=np.byte), "RGB")

                focalX = K[0, 0]
                focalY = K[1, 1]
                def focal2fov(focal, pixels):
                    return 2*math.atan(pixels/(2*focal))
                FovX = focal2fov(focalX, image.size[0])
                FovY = focal2fov(focalY, image.size[1])

                # load smplx data 'mesh-f00001_smplx'
                id = os.path.basename(image_path).split('.')[0].split('_')[1]
                vertices_path = os.path.join(view,
                                            'SMPLX', 'mesh-f'+id[1:]+'_smplx.ply')
                vert_data = PlyData.read(vertices_path)
                vertices = vert_data['vertex']
                xyz = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T

                smpl_param_path = os.path.join(view,
                                            'SMPLX', 'mesh-f'+id[1:]+'_smplx.pkl')
                with open(smpl_param_path, 'rb') as f:
                    smpl_param = pickle.load(f)

                ###
                # load smpl data
                smpl_param = {
                    'global_orient': np.expand_dims(smpl_param['global_orient'].astype(np.float32), axis=0),
                    'transl': np.expand_dims(smpl_param['transl'].astype(np.float32), axis=0),
                    'body_pose': np.expand_dims(smpl_param['body_pose'].astype(np.float32), axis=0),
                    'jaw_pose': np.expand_dims(smpl_param['jaw_pose'].astype(np.float32), axis=0),
                    'betas': np.expand_dims(smpl_param['betas'].astype(np.float32), axis=0),
                    'expression': np.expand_dims(smpl_param['expression'].astype(np.float32), axis=0),
                    'leye_pose': np.expand_dims(smpl_param['leye_pose'].astype(np.float32), axis=0),
                    'reye_pose': np.expand_dims(smpl_param['reye_pose'].astype(np.float32), axis=0),
                    'left_hand_pose': np.expand_dims(smpl_param['left_hand_pose'].astype(np.float32), axis=0),
                    'right_hand_pose': np.expand_dims(smpl_param['right_hand_pose'].astype(np.float32), axis=0),
                    }
                smpl_param['R'] = np.eye(3).astype(np.float32)
                smpl_param['Th'] = smpl_param['transl'].astype(np.float32)
                smpl_param_tensor = {}
                for key in smpl_param.keys():
                    smpl_param_tensor[key] = torch.from_numpy(smpl_param[key])
                body_model_output = smplx_model(
                    global_orient=smpl_param_tensor['global_orient'],
                    betas=smpl_param_tensor['betas'],
                    body_pose=smpl_param_tensor['body_pose'],
                    jaw_pose=smpl_param_tensor['jaw_pose'],
                    left_hand_pose=smpl_param_tensor['left_hand_pose'],
                    right_hand_pose=smpl_param_tensor['right_hand_pose'],
                    leye_pose=smpl_param_tensor['leye_pose'],
                    reye_pose=smpl_param_tensor['reye_pose'],
                    expression=smpl_param_tensor['expression'],
                    transl=smpl_param_tensor['transl'],
                    return_full_pose=True,
                )
                smpl_param['poses'] = body_model_output.full_pose.detach()
                smpl_param['shapes'] = np.concatenate([smpl_param['betas'], smpl_param['expression']], axis=-1)

                # from nosmpl.vis.vis_o3d import vis_mesh_o3d
                # vertices = body_model_output.vertices.squeeze()
                # faces = smplx_model.faces.astype(np.int32)
                # vis_mesh_o3d(vertices.detach().cpu().numpy(), faces)
                # vis_mesh_o3d(xyz, faces)
                ###

                # obtain the original bounds for point sampling
                min_xyz = np.min(xyz, axis=0)
                max_xyz = np.max(xyz, axis=0)
                min_xyz -= 0.05
                max_xyz += 0.05
                world_bound = np.stack([min_xyz, max_xyz], axis=0)

                # xy = get_2dkps(xyz, K, w2c[:3], image.size[1], image.size[0])
                # import matplotlib.pyplot as plt
                # plt.figure(figsize=(6.4, 3.6))
                # plt.scatter(xy[:, 0].tolist(), xy[:, 1].tolist())
                # plt.show()

                # get bounding mask and bcakground mask
                bound_mask = get_bound_2d_mask(world_bound, K, w2c[:3], image.size[1], image.size[0])
                bound_mask = Image.fromarray(np.array(bound_mask * 255.0, dtype=np.byte))

                try:
                    bkgd_mask = Image.fromarray(np.array(msk * 255.0, dtype=np.byte))
                except:
                    bkgd_mask = Image.fromarray(np.array(msk[:,:,0] * 255.0, dtype=np.byte))


                cam_infos.append(CameraInfo(uid=idx, pose_id=pose_index, R=R, T=T, K=K, FovY=FovY, FovX=FovX, image=image,
                                            image_path=image_path, image_name=image_name, bkgd_mask=bkgd_mask,
                                            bound_mask=bound_mask, width=image.size[0], height=image.size[1],
                                            smpl_param=smpl_param, world_vertex=xyz, world_bound=world_bound,
                                            big_pose_smpl_param=big_pose_smpl_param, big_pose_world_vertex=big_pose_xyz,
                                            big_pose_world_bound=big_pose_world_bound))

                idx += 1


            # break

        self.cam_infos: List[CameraInfo] = cam_infos



    def __len__(self):
        return len(self.cam_infos)

    def __getitem__(self, idx):
        cam: CameraInfo = self.cam_infos[idx]

        img = cam.image
        msk = cam.bkgd_mask

        pose = cam.smpl_param["poses"].reshape((-1, 3)).numpy()
        # pose = np.concatenate([cam.smpl_param["global_orient"][None], pose], 0)

        ret = {
            "rgb": ((np.asarray(img)/ 255.)).astype(np.float32),
            "mask": (np.asarray(msk) / 255.).astype(np.float32),
            "K": cam.K.copy(),
            "smpl_beta": cam.smpl_param["betas"][0],  # ! use the first beta!
            "smpl_pose": pose,
            "smpl_trans": cam.smpl_param["transl"][0],
            # "jaw_pose": cam["jaw_pose"],
            # "left_hand_pose": cam["left_hand_pose"],
            # "right_hand_pose": cam["right_hand_pose"],
            # "leye_pose": cam["leye_pose"],
            # "reye_pose": cam["reye_pose"],
            # "expression": cam["expression"],
            "idx": idx,
            "cam": cam,
        }

        meta_info = {
            # "video": self.video_name,
        }
        # viz_id = f"video{self.video_name}_dataidx{idx}"
        # meta_info["viz_id"] = viz_id
        return ret, meta_info



if __name__ == "__main__":
    dataset = Dataset(
        data_root="../data/insav_wild", video_name="aist_gBR_sBM_c01_d05_mBR1_ch06"
    )
    ret = dataset[0]
