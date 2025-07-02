import os
import os.path as osp
from glob import glob
import shutil
from tqdm import tqdm

import mmcv
import cv2
import torch
import numpy as np
from mmcv.runner import get_dist_info
import torch.distributed as dist
from pytorch3d.io import save_obj

from config.config import cfg
from util.preprocessing import load_img, augmentation_keep_size
from util.formatting import DefaultFormatBundle
from detrsmpl.data.datasets.pipelines.transforms import Normalize
from detrsmpl.core.conventions.keypoints_mapping import convert_kps
from detrsmpl.core.visualization.visualize_smpl import render_smpl
from detrsmpl.models.body_models.builder import build_body_model
from detrsmpl.utils.ffmpeg_utils import video_to_images

class INFERENCE_youtube(torch.utils.data.Dataset):
    def __init__(self, img_dir=None, out_path=None):
        
        self.output_path = out_path
        self.img_dir = img_dir
        self.is_vid = False
        body_model_cfg = dict(
            type='smplx',
            keypoint_src='smplx',
            num_expression_coeffs=10,
            num_betas=10,
            gender='neutral',
            keypoint_dst='smplx_137',
            model_path='data/body_models/smplx',
            use_pca=False,
            use_face_contour=True,
            batch_size = cfg.batch_size)
        self.body_model = build_body_model(body_model_cfg).to('cuda') 
        
        # list video id
        self.video_list = [vid for vid in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, vid))]
        self.video_list = self.video_list[:20]
        self.segments = {}
        self.data_list = []
        for video_id in tqdm(self.video_list):
            vid_dir = os.path.join(img_dir, video_id)
            segment_list = [seg for seg in os.listdir(vid_dir) if os.path.isdir(os.path.join(vid_dir, seg))]
            self.segments[video_id] = segment_list
            for segment_id in segment_list:
                seg_dir = os.path.join(vid_dir, segment_id)
                img_fns = [fn for fn in os.listdir(seg_dir) if fn.endswith('.jpg') or fn.endswith('.png')]
                for img_fn in img_fns:
                    img_path = os.path.join(seg_dir, img_fn)
                    fn = img_fn.split('.')[0]
                    result_dir = os.path.join(self.output_path, video_id, segment_id)
                    result_path = os.path.join(result_dir, fn + '.pt')
                    result_bad_path = result_path.replace('.pt', '.bad.pt')
                    if osp.exists(result_path) or osp.exists(result_bad_path):
                        continue

                    self.data_list.append({
                        'video_id': video_id,
                        'segment_id': segment_id,
                        'img_path': img_path,
                        'fn': fn,
                        'result_dir': result_dir,
                        'result_path': result_path,
                    })

        self.num_person = cfg.num_person if 'num_person' in cfg else 0.1
        self.score_threshold = cfg.threshold if 'threshold' in cfg else 0.1  
        self.format = DefaultFormatBundle()
        self.normalize = Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
       
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]

        img = load_img(data['img_path'],'BGR')
        self.resolution = img.shape[:2]
        img_whole_bbox = np.array([0, 0, img.shape[1],img.shape[0]])
        img, img2bb_trans, bb2img_trans, _, _ = \
            augmentation_keep_size(img, img_whole_bbox, 'test')

        # cropped_img_shape=img.shape[:2]
        img = (img.astype(np.float32)) 
        
        inputs = {'img': img}
        targets = {
            'body_bbox_center': np.array(img_whole_bbox[None]),
            'body_bbox_size': np.array(img_whole_bbox[None])}
        meta_info = {
            'ori_shape':np.array(self.resolution),
            'img_shape': np.array(img.shape[:2]),
            'img2bb_trans': img2bb_trans,
            'bb2img_trans': bb2img_trans,
            'ann_idx': idx}
        result = {**inputs, **targets, **meta_info}
        
        result = self.normalize(result)
        result = self.format(result)
            
        return result
        
    def inference(self, outs):
        for out in outs:
            is_bad_segment = False
            ann_idx = out['image_idx']
            data = self.data_list[ann_idx]
            rlt = {}

            scores = out['scores'].clone().cpu().numpy()
            img_shape = out['img_shape'].cpu().numpy()[::-1] # w, h
            img = cv2.imread(data['img_path']) # h, w
            H, W = img.shape[:2]
            rlt['width'] = W
            rlt['height'] = H
            
            scale = W / img_shape[0]
            K = np.array(
                        [[5000, 0, img_shape[0]/2],
                         [0, 5000, img_shape[1]/2],
                         [0, 0, 1]])
            K = K * scale
            rlt['K'] = K

            body_bbox = out['body_bbox']
            body_bbox = body_bbox * scale
            rlt['boxes'] = body_bbox
            # joint_3d, _ =  convert_kps(out['smpl_kp3d'].clone().cpu().numpy(),src='smplx',dst='smplx', approximate=True)

            # results
            rlt['vertices'] = out['smpl_verts'] + out['cam_trans'][:, None, :]
            rlt['keypoints_coco'] = out['keypoints_coco']
            rlt['smpl_kp3d'] = out['smpl_kp3d']
            rlt['smplx_root_pose'] = out['smplx_root_pose']
            rlt['smplx_body_pose'] = out['smplx_body_pose']
            rlt['smplx_lhand_pose'] = out['smplx_lhand_pose']
            rlt['smplx_rhand_pose'] = out['smplx_rhand_pose']
            rlt['smplx_jaw_pose'] = out['smplx_jaw_pose']
            rlt['smplx_shape'] = out['smplx_shape']
            rlt['smplx_expr'] = out['smplx_expr']
            rlt['cam_trans'] = out['cam_trans']
            rlt['body_bbox'] = out['body_bbox'] * scale
            rlt['lhand_bbox'] = out['lhand_bbox'] * scale
            rlt['rhand_bbox'] = out['rhand_bbox'] * scale
            rlt['face_bbox'] = out['face_bbox'] * scale
            rlt['bb2img_trans'] = out['bb2img_trans']
            rlt['img2bb_trans'] = out['img2bb_trans']

            # screen valid results
            valid_idxs = [i for i, s in enumerate(scores) if s >= self.score_threshold]
            for key in rlt:
                if isinstance(rlt[key], torch.Tensor):
                    if len(rlt[key].shape) > 1:
                        rlt[key] = rlt[key][valid_idxs]

            if len(valid_idxs) == 0:
                print(f"No valid detections for {data['img_path']}")
                result_bad_path = data['result_path'].replace('.pt', '.bad.pt')
                torch.save(rlt, result_bad_path)
                is_bad_segment = True
                continue

            # check multiple large boxes
            all_boxes = rlt['boxes'].detach().cpu().numpy()
            calc_area = lambda x: x[0] * x[1]
            areas = [calc_area(bbox[2:4] - bbox[0:2]) for bbox in all_boxes]
            area_idxs = np.argsort(areas)[::-1]
            max_idx = area_idxs[0]
            if len(all_boxes) > 1:
                second_idx = area_idxs[1]
                if areas[second_idx] > 0.5 * areas[max_idx]:
                    print(f"Frame {data['img_path']} has multiple large boxes ({(areas[second_idx] / areas[max_idx]):.2f}). Skipping.")
                    is_bad_segment = True
            
            valid_idxs = [i for i in range(len(areas)) if areas[i] > 0.5 * areas[max_idx]]
            for key in rlt:
                if isinstance(rlt[key], torch.Tensor):
                    if len(rlt[key].shape) > 1:
                        rlt[key] = rlt[key][valid_idxs]

            # img = cv2.resize(img, (img_shape[0],img_shape[1]), cv2.INTER_CUBIC)
            vis_ratio = 0.5
            vis_K = K * vis_ratio
            vis_H = int(H * vis_ratio)
            vis_W = int(W * vis_ratio)
            vis_bbox = rlt['body_bbox'].detach().cpu().numpy() * vis_ratio
            vis_face_bbox = rlt['face_bbox'].detach().cpu().numpy() * vis_ratio
            vis_lhand_bbox = rlt['lhand_bbox'].detach().cpu().numpy() * vis_ratio
            vis_rhand_bbox = rlt['rhand_bbox'].detach().cpu().numpy() * vis_ratio
            vis = cv2.resize(img, (vis_W, vis_H), interpolation=cv2.INTER_CUBIC)

            # save results
            os.makedirs(data['result_dir'], exist_ok=True)
            if is_bad_segment:
                result_path = data['result_path'].replace('.pt', '.bad.pt')
            else:
                result_path = data['result_path']
            torch.save(rlt, result_path)

            # visualize results
            fn = data['fn']
            if fn != '000000' and not is_bad_segment:
                continue
            vis_fn = os.path.join(data['result_dir'], f'{fn}.jpg')

            for i in range(rlt['vertices'].shape[0]):
                vis = mmcv.imshow_bboxes(vis, vis_bbox[i:i+1], show=False, colors='green') 
                vis = mmcv.imshow_bboxes(vis, vis_face_bbox[i:i+1], show=False, colors='white') 
                vis = mmcv.imshow_bboxes(vis, vis_lhand_bbox[i:i+1], show=False, colors='red') 
                vis = mmcv.imshow_bboxes(vis, vis_rhand_bbox[i:i+1], show=False, colors='blue') 

                vertices = rlt['vertices'][i:i+1]
                vis = render_smpl(
                    verts=vertices,
                    body_model=self.body_model,
                    K=vis_K,
                    R=None,
                    T=None,
                    output_path=vis_fn,
                    image_array=vis,
                    in_ndc=False,
                    alpha=0.9,
                    convention='opencv',
                    projection='perspective',
                    overwrite=True,
                    no_grad=True,
                    device='cuda',
                    resolution=[vis_H, vis_W],
                    render_choice='hq',
                    return_tensor=True,
                )
                vis = vis.squeeze(0).detach().cpu().numpy()

        return None

