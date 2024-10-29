import os
import cv2
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from torchvision.transforms import v2
import tarfile
import json

import kiui
from core.options import Options
from core.utils import get_rays, grid_distortion, orbit_camera_jitter, normalize_depth

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class ObjaverseDataset(Dataset):

    def _warn(self):
        raise NotImplementedError('this dataset is just an example and cannot be used directly, you should modify it to your own setting! (search keyword TODO)')

    def __init__(self, opt: Options, training=True):
        
        self.opt = opt
        self.training = training

        # TODO: remove this barrier
        # self._warn()

        # TODO: load the list of objects for training
        self.items = []
        with open('gobjaverse/gobjaverse_small.json', 'r') as f:
            data = json.load(f)
            for item in data:
                self.items.append(item)

        # naive split
        if self.training:
            self.items = self.items[:-self.opt.batch_size]
        else:
            self.items = self.items[-self.opt.batch_size:]
        
        # default camera intrinsics
        self.tan_half_fov = np.tan(0.5 * np.deg2rad(self.opt.fovy))
        self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32)
        self.proj_matrix[0, 0] = 1 / self.tan_half_fov
        self.proj_matrix[1, 1] = 1 / self.tan_half_fov
        self.proj_matrix[2, 2] = (self.opt.zfar + self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[3, 2] = - (self.opt.zfar * self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[2, 3] = 1


    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):

        uid = self.items[idx]
        results = {}

        # load num_views images
        images = []
        albedos = []
        normals = []
        depths = []
        masks = []
        cam_poses = []
        
        vid_cnt = 0

        # TODO: choose views, based on your rendering settings
        if self.training:
            vids = np.random.permutation(np.arange(0, 36))[:self.opt.num_input_views].tolist() + np.random.permutation(36).tolist()
        else:
            vids = [0,29,8,33,16,37,2,10,18,28]
        
        for vid in vids:

            try:
                tar_path = os.path.join(self.opt.data_path, f"{uid}.tar")
                uid_last = uid.split('/')[1]

                if self.opt.rar_data:
                    tar_path = os.path.join(self.opt.data_path, f"{uid}.tar")
                    image_path = os.path.join(uid_last, 'campos_512_v4', f"{vid:05d}/{vid:05d}.png")
                    meta_path = os.path.join(uid_last, 'campos_512_v4', f"{vid:05d}/{vid:05d}.json")
                    albedo_path = os.path.join(uid_last, 'campos_512_v4', f"{vid:05d}/{vid:05d}_albedo.png") # black bg...
                    nd_path = os.path.join(uid_last, 'campos_512_v4', f"{vid:05d}/{vid:05d}_nd.exr")
                    
                    with tarfile.open(tar_path, 'r') as tar:
                        with tar.extractfile(image_path) as f:
                            image = np.frombuffer(f.read(), np.uint8)
                        with tar.extractfile(albedo_path) as f:
                            albedo = np.frombuffer(f.read(), np.uint8)
                        with tar.extractfile(meta_path) as f:
                            meta = json.loads(f.read().decode())
                        with tar.extractfile(nd_path) as f:
                            nd = np.frombuffer(f.read(), np.uint8)
                else:
                    image_path = os.path.join(self.opt.data_path, uid, f"{vid:05d}/{vid:05d}.png")
                    meta_path = os.path.join(self.opt.data_path, uid, f"{vid:05d}/{vid:05d}.json")
                    nd_path = os.path.join(self.opt.data_path, uid, f"{vid:05d}/{vid:05d}_nd.exr")
                    
                    albedo_path = os.path.join(self.opt.data_path, uid, f"{vid:05d}/{vid:05d}_albedo.png")
                    
                    with open(image_path, 'rb') as f:
                        image = np.frombuffer(f.read(), dtype=np.uint8)
                        
                    with open(albedo_path, 'rb') as f:
                        albedo = np.frombuffer(f.read(), dtype=np.uint8)

                    with open(meta_path, 'r') as f:
                        meta = json.load(f)

                    with open(nd_path, 'rb') as f:
                        nd = np.frombuffer(f.read(), np.uint8)
                    
                image = torch.from_numpy(cv2.imdecode(image, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255) 
                albedo = torch.from_numpy(cv2.imdecode(albedo, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255)
                
                c2w = np.eye(4)
                c2w[:3, 0] = np.array(meta['x'])
                c2w[:3, 1] = np.array(meta['y'])
                c2w[:3, 2] = np.array(meta['z'])
                c2w[:3, 3] = np.array(meta['origin'])
                c2w = torch.tensor(c2w, dtype=torch.float32).reshape(4, 4)
                
                nd = cv2.imdecode(nd, cv2.IMREAD_UNCHANGED).astype(np.float32) # [512, 512, 4] in [-1, 1]
                normal = nd[..., :3] # in [-1, 1], bg is [0, 0, 1]
                depth = nd[..., 3] # in [0, +?), bg is 0

                # rectify normal directions
                normal = normal[..., ::-1]
                normal[..., 0] *= -1
                normal = torch.from_numpy(normal.astype(np.float32)).nan_to_num_(0) # there are nans in gt normal... 
                depth = torch.from_numpy(depth.astype(np.float32)).nan_to_num_(0)
            
            except Exception as e:
                print(f'[WARN] dataset {uid} {vid}: {e}')
                continue
            
            # blender world + opencv cam --> opengl world & cam
            c2w[1] *= -1
            c2w[[1, 2]] = c2w[[2, 1]]
            c2w[:3, 1:3] *= -1 # invert up and forward direction

            image = image.permute(2, 0, 1) # [4, 512, 512]
            mask = image[3:4] # [1, 512, 512]

            image = image[:3] * mask + (1 - mask) # [3, 512, 512], to white bg

            image = image[[2,1,0]].contiguous() # bgr to rgb

            # albdeo
            albedo = albedo.permute(2, 0, 1) 
            albedo = albedo[:3] * mask + (1 - mask) 
            albedo = albedo[[2,1,0]].contiguous()

            normal = normal.permute(2, 0, 1) # [3, 512, 512]
            normal = normal * mask # to [0, 0, 0] bg

            images.append(image)
            albedos.append(albedo)
            normals.append(normal)
            depths.append(depth)
            masks.append(mask.squeeze(0))
            cam_poses.append(c2w)

            vid_cnt += 1
            if vid_cnt == self.opt.num_views:
                break

        if vid_cnt < self.opt.num_views:
            print(f'[WARN] dataset {uid}: not enough valid views, only {vid_cnt} views found!')
            n = self.opt.num_views - vid_cnt
            images = images + [images[-1]] * n
            normals = normals + [normals[-1]] * n
            depths = depths + [depths[-1]] * n
            masks = masks + [masks[-1]] * n
            cam_poses = cam_poses + [cam_poses[-1]] * n
        
        images = torch.stack(images, dim=0) # [V, 3, H, W]
        albedos = torch.stack(albedos, dim=0)
        normals = torch.stack(normals, dim=0) # [V, 3, H, W]
        depths = torch.stack(depths, dim=0) # [V, H, W]
        masks = torch.stack(masks, dim=0) # [V, H, W]
        cam_poses = torch.stack(cam_poses, dim=0) # [V, 4, 4]
        
        depths = normalize_depth(depths)

        # normalized camera feats as in paper (transform the first pose to a fixed position)
        radius = torch.norm(cam_poses[0, :3, 3])
        cam_poses[:, :3, 3] *= self.opt.cam_radius / radius
        transform = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, self.opt.cam_radius], [0, 0, 0, 1]], dtype=torch.float32) @ torch.inverse(cam_poses[0])
        cam_poses = transform.unsqueeze(0) @ cam_poses  # [V, 4, 4]
        cam_poses_input = cam_poses[:self.opt.num_input_views].clone()

        # images_input = F.interpolate(images[:self.opt.num_input_views].clone(), size=(self.opt.input_size, self.opt.input_size), mode='bilinear', align_corners=False) # [V, C, H, W]
        images = F.interpolate(images, size=(self.opt.input_size, self.opt.input_size), mode='bilinear', align_corners=False) 
        albedos = F.interpolate(albedos, size=(self.opt.input_size, self.opt.input_size), mode='bilinear', align_corners=False)
        depths = F.interpolate(depths.unsqueeze(1), size=(self.opt.input_size, self.opt.input_size), mode='bilinear', align_corners=False)
        
        target_images = v2.functional.resize(
            images, self.opt.output_size, interpolation=3, antialias=True).clamp(0, 1)
        
        target_albedos = v2.functional.resize(
            albedos, self.opt.output_size, interpolation=3, antialias=True).clamp(0, 1)
        
        target_alphas = v2.functional.resize(
            masks.unsqueeze(1), self.opt.output_size, interpolation=0, antialias=True)
        
        target_depth = v2.functional.resize(
            depths, self.opt.output_size, interpolation=0, antialias=True)

        # target gt
        results['images_output'] = target_images
        results['albedos_output'] = target_albedos
        results['masks_output'] = target_alphas
        results['depth_output'] = target_depth
        
        # data augmentation
        images_input = images[:self.opt.num_input_views].clone()
        if self.training:
            # apply random grid distortion to simulate 3D inconsistency
            if random.random() < self.opt.prob_grid_distortion:
                images_input[1:] = grid_distortion(images_input[1:])
            # apply camera jittering (only to input!)
            if random.random() < self.opt.prob_cam_jitter:
                cam_poses_input[1:] = orbit_camera_jitter(cam_poses_input[1:])

        # images_input = TF.normalize(images_input, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

        # resize render ground-truth images, range still in [0, 1]
        # results['images_output'] = F.interpolate(images, size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False) # [V, C, output_size, output_size]
        # results['masks_output'] = F.interpolate(masks.unsqueeze(1), size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False) # [V, 1, output_size, output_size]

        # build rays for input views
        rays_embeddings = []
        for i in range(self.opt.num_input_views):
            rays_o, rays_d = get_rays(cam_poses_input[i], self.opt.input_size, self.opt.input_size, self.opt.fovy) # [h, w, 3]
            rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
            rays_embeddings.append(rays_plucker)

     
        rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous() # [V, 6, h, w]
        final_input = torch.cat([images_input, rays_embeddings], dim=1) # [V=4, 9, H, W]
        results['input'] = final_input

        # opengl to colmap camera for gaussian renderer
        cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
        
        # cameras needed by gaussian rasterizer
        cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
        cam_view_proj = cam_view @ self.proj_matrix # [V, 4, 4]
        cam_pos = - cam_poses[:, :3, 3] # [V, 3]
        
        results['cam_view'] = cam_view
        results['cam_view_proj'] = cam_view_proj
        results['cam_pos'] = cam_pos

        return results
