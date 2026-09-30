import sys
sys.path.append('')

"""VAD 推理结果可视化脚本。

整体流程：
1. 从 pkl 文件中读取检测、地图和自车规划预测；
2. 针对每个 nuScenes sample 生成一张鸟瞰图（BEV）；
3. 读取同一时刻的六路相机图像，并把预测框变换到各相机坐标系；
4. 在前视相机中额外绘制自车规划轨迹；
5. 将六路相机图和 BEV 图拼接，逐帧写入 vis.mp4。

检测框最初位于 nuScenes 全局坐标系，绘制时会按需转换到 LiDAR 或相机
坐标系。预测轨迹通常保存为逐时间步位移，绘图前需要累加。
"""
import os
import argparse
import os.path as osp
from PIL import Image
from tqdm import tqdm
from typing import List, Dict

import cv2
import mmcv
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from mmdet.datasets.pipelines import to_tensor
from matplotlib.collections import LineCollection
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.eval.common.data_classes import EvalBoxes, EvalBox
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility

from projects.mmdet3d_plugin.core.bbox.structures.nuscenes_box import CustomNuscenesBox, CustomDetectionBox, color_map
from projects.mmdet3d_plugin.datasets.nuscenes_vad_dataset import VectorizedLocalMap, LiDARInstanceLines


# nuScenes 标准六相机名称；部分单帧调试函数会使用。
cams = ['CAM_FRONT',
 'CAM_FRONT_RIGHT',
 'CAM_BACK_RIGHT',
 'CAM_BACK',
 'CAM_BACK_LEFT',
 'CAM_FRONT_LEFT']


def render_annotation(
        anntoken: str,
        margin: float = 10,
        view: np.ndarray = np.eye(4),
        box_vis_level: BoxVisibility = BoxVisibility.ANY,
        out_path: str = 'render.png',
        extra_info: bool = False) -> None:
    """
    同时在 LiDAR 鸟瞰图和可见相机中绘制一个指定的 GT 标注。

    这是单目标调试工具，并非文件末尾批量生成视频的主流程。
    :param anntoken: Sample_annotation token.
    :param margin: How many meters in each direction to include in LIDAR view.
    :param view: LIDAR view point.
    :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
    :param out_path: Optional path to save the rendered figure to disk.
    :param extra_info: Whether to render extra information below camera view.
    """
    # 由 annotation 找到所属 sample 和该帧的全部传感器数据。
    ann_record = nusc.get('sample_annotation', anntoken)
    sample_record = nusc.get('sample', ann_record['sample_token'])
    assert 'LIDAR_TOP' in sample_record['data'].keys(), 'Error: No LIDAR_TOP in data, unable to render.'

    # 遍历六路相机，找出能够看到该目标的相机。
    boxes, cam = [], []
    cams = [key for key in sample_record['data'].keys() if 'CAM' in key]
    all_bboxes = []
    select_cams = []
    for cam in cams:
        _, boxes, _ = nusc.get_sample_data(sample_record['data'][cam], box_vis_level=box_vis_level,
                                           selected_anntokens=[anntoken])
        if len(boxes) > 0:
            all_bboxes.append(boxes)
            select_cams.append(cam)
            # We found an image that matches. Let's abort.
    # assert len(boxes) > 0, 'Error: Could not find image where annotation is visible. ' \
    #                      'Try using e.g. BoxVisibility.ANY.'
    # assert len(boxes) < 2, 'Error: Found multiple annotations. Something is wrong!'

    num_cam = len(all_bboxes)

    fig, axes = plt.subplots(1, num_cam + 1, figsize=(18, 9))
    select_cams = [sample_record['data'][cam] for cam in select_cams]
    print('bbox in cams:', select_cams)
    # 左侧子图：在 LIDAR_TOP 坐标系下绘制点云和 3D 框。
    lidar = sample_record['data']['LIDAR_TOP']
    data_path, boxes, camera_intrinsic = nusc.get_sample_data(lidar, selected_anntokens=[anntoken])
    LidarPointCloud.from_file(data_path).render_height(axes[0], view=view)
    for box in boxes:
        c = np.array(get_color(box.name)) / 255.0
        box.render(axes[0], view=view, colors=(c, c, c))
        corners = view_points(boxes[0].corners(), view, False)[:2, :]
        axes[0].set_xlim([np.min(corners[0, :]) - margin, np.max(corners[0, :]) + margin])
        axes[0].set_ylim([np.min(corners[1, :]) - margin, np.max(corners[1, :]) + margin])
        axes[0].axis('off')
        axes[0].set_aspect('equal')

    # 其余子图：把同一个 3D 框投影到每个可见相机平面。
    for i in range(1, num_cam + 1):
        cam = select_cams[i - 1]
        data_path, boxes, camera_intrinsic = nusc.get_sample_data(cam, selected_anntokens=[anntoken])
        im = Image.open(data_path)
        axes[i].imshow(im)
        axes[i].set_title(nusc.get('sample_data', cam)['channel'])
        axes[i].axis('off')
        axes[i].set_aspect('equal')
        for box in boxes:
            c = np.array(get_color(box.name)) / 255.0
            box.render(axes[i], view=camera_intrinsic, normalize=True, colors=(c, c, c))

        # Print extra information about the annotation below the camera view.
        axes[i].set_xlim(0, im.size[0])
        axes[i].set_ylim(im.size[1], 0)

    # 可选：显示类别、点数、距离和目标尺寸等 GT 信息。
    if extra_info:
        rcParams['font.family'] = 'monospace'

        w, l, h = ann_record['size']
        category = ann_record['category_name']
        lidar_points = ann_record['num_lidar_pts']
        radar_points = ann_record['num_radar_pts']

        sample_data_record = nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])
        pose_record = nusc.get('ego_pose', sample_data_record['ego_pose_token'])
        dist = np.linalg.norm(np.array(pose_record['translation']) - np.array(ann_record['translation']))

        information = ' \n'.join(['category: {}'.format(category),
                                  '',
                                  '# lidar points: {0:>4}'.format(lidar_points),
                                  '# radar points: {0:>4}'.format(radar_points),
                                  '',
                                  'distance: {:>7.3f}m'.format(dist),
                                  '',
                                  'width:  {:>7.3f}m'.format(w),
                                  'length: {:>7.3f}m'.format(l),
                                  'height: {:>7.3f}m'.format(h)])

        plt.annotate(information, (0, 0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top')

    if out_path is not None:
        plt.savefig(out_path)


def get_sample_data(sample_data_token: str,
                    box_vis_level: BoxVisibility = BoxVisibility.ANY,
                    selected_anntokens=None,
                    use_flat_vehicle_coordinates: bool = False):
    """
    读取传感器数据及其 GT 框，并将框从全局坐标系变换到传感器坐标系。

    坐标变换链为 global -> ego -> sensor。
    :param sample_data_token: Sample_data token.
    :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
    :param selected_anntokens: If provided only return the selected annotation.
    :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
                                         aligned to z-plane in the world.
    :return: (data_path, boxes, camera_intrinsic <np.array: 3, 3>)
    """

    # 标定记录描述 sensor -> ego，位姿记录描述 ego -> global。
    sd_record = nusc.get('sample_data', sample_data_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    data_path = nusc.get_sample_data_path(sample_data_token)

    if sensor_record['modality'] == 'camera':
        cam_intrinsic = np.array(cs_record['camera_intrinsic'])
        imsize = (sd_record['width'], sd_record['height'])
    else:
        cam_intrinsic = None
        imsize = None

    # 获取指定标注，或获取当前 sample_data 对应的全部 GT 框。
    if selected_anntokens is not None:
        boxes = list(map(nusc.get_box, selected_anntokens))
    else:
        boxes = nusc.get_boxes(sample_data_token)

    # Box 原本处于全局坐标系，依次逆变换到 ego 和传感器坐标系。
    box_list = []
    for box in boxes:
        if use_flat_vehicle_coordinates:
            # Move box to ego vehicle coord system parallel to world z plane.
            yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)
        else:
            # Move box to ego vehicle coord system.
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(pose_record['rotation']).inverse)

            #  Move box to sensor coord system.
            box.translate(-np.array(cs_record['translation']))
            box.rotate(Quaternion(cs_record['rotation']).inverse)

        if sensor_record['modality'] == 'camera' and not \
                box_in_image(box, cam_intrinsic, imsize, vis_level=box_vis_level):
            continue

        box_list.append(box)

    return data_path, box_list, cam_intrinsic


def get_predicted_data(sample_data_token: str,
                       box_vis_level: BoxVisibility = BoxVisibility.ANY,
                       selected_anntokens=None,
                       use_flat_vehicle_coordinates: bool = False,
                       pred_anns=None
                       ):
    """
    将传入的预测框从全局坐标系转换到指定传感器坐标系。

    相机模式下还会剔除画面外的预测框。
    :param sample_data_token: Sample_data token.
    :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
    :param selected_anntokens: If provided only return the selected annotation.
    :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
                                         aligned to z-plane in the world.
    :return: (data_path, boxes, camera_intrinsic <np.array: 3, 3>)
    """

    # Retrieve sensor & pose records
    sd_record = nusc.get('sample_data', sample_data_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    data_path = nusc.get_sample_data_path(sample_data_token)

    if sensor_record['modality'] == 'camera':
        cam_intrinsic = np.array(cs_record['camera_intrinsic'])
        imsize = (sd_record['width'], sd_record['height'])
    else:
        cam_intrinsic = None
        imsize = None

    # Retrieve all sample annotations and map to sensor coordinate system.
    # if selected_anntokens is not None:
    #    boxes = list(map(nusc.get_box, selected_anntokens))
    # else:
    #    boxes = nusc.get_boxes(sample_data_token)
    boxes = pred_anns
    # Make list of Box objects including coord system transforms.
    box_list = []
    for box in boxes:
        if use_flat_vehicle_coordinates:
            # Move box to ego vehicle coord system parallel to world z plane.
            yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)
        else:
            # Move box to ego vehicle coord system.
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(pose_record['rotation']).inverse)

            #  Move box to sensor coord system.
            box.translate(-np.array(cs_record['translation']))
            box.rotate(Quaternion(cs_record['rotation']).inverse)

        if sensor_record['modality'] == 'camera' and not \
                box_in_image(box, cam_intrinsic, imsize, vis_level=box_vis_level):
            continue
        box_list.append(box)

    return data_path, box_list, cam_intrinsic


def lidiar_render(sample_token, data, out_path=None, out_name=None, traj_use_perstep_offset=True):
    """整理单帧 GT/预测框，并调用 visualize_sample 生成 BEV 图。

    data['results'][sample_token] 保存检测和运动预测，地图及规划结果则由
    visualize_sample 继续从 data 中读取。
    """
    bbox_gt_list = []
    bbox_pred_list = []
    sample_rec = nusc.get('sample', sample_token)
    anns = sample_rec['anns']
    sd_record = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    # 构造 GT 框，并沿 annotation.next 读取未来 6 帧真实轨迹。
    for ann in anns:
        content = nusc.get('sample_annotation', ann)
        gt_fut_trajs, gt_fut_masks = get_gt_fut_trajs(
            nusc=nusc, anno=content, cs_record=cs_record, 
            pose_record=pose_record, fut_ts=6
        )
        try:
            bbox_gt_list.append(CustomDetectionBox(
                sample_token=content['sample_token'],
                translation=tuple(content['translation']),
                size=tuple(content['size']),
                rotation=tuple(content['rotation']),
                velocity=nusc.box_velocity(content['token'])[:2],
                fut_trajs=tuple(gt_fut_trajs),
                ego_translation=(0.0, 0.0, 0.0) if 'ego_translation' not in content
                else tuple(content['ego_translation']),
                num_pts=-1 if 'num_pts' not in content else int(content['num_pts']),
                detection_name=category_to_detection_name(content['category_name']),
                detection_score=-1.0 if 'detection_score' not in content else float(content['detection_score']),
                attribute_name=''))
        except:
            pass

    # 将当前帧模型预测转换为统一的 CustomDetectionBox。
    bbox_anns = data['results'][sample_token]
    for content in bbox_anns:
        bbox_pred_list.append(CustomDetectionBox(
            sample_token=content['sample_token'],
            translation=tuple(content['translation']),
            size=tuple(content['size']),
            rotation=tuple(content['rotation']),
            velocity=tuple(content['velocity']),
            fut_trajs=tuple(content['fut_traj']),
            ego_translation=(0.0, 0.0, 0.0) if 'ego_translation' not in content
            else tuple(content['ego_translation']),
            num_pts=-1 if 'num_pts' not in content else int(content['num_pts']),
            detection_name=content['detection_name'],
            detection_score=-1.0 if 'detection_score' not in content else float(content['detection_score']),
            attribute_name=content['attribute_name']))
    # EvalBoxes 以 sample_token 为键组织框。
    gt_annotations = EvalBoxes()
    pred_annotations = EvalBoxes()
    gt_annotations.add_boxes(sample_token, bbox_gt_list)
    pred_annotations.add_boxes(sample_token, bbox_pred_list)
    # print('green is ground truth')
    # print('blue is the predited result')
    visualize_sample(nusc, sample_token, gt_annotations, pred_annotations,
                     savepath=out_path, traj_use_perstep_offset=traj_use_perstep_offset, pred_data=data)


def get_color(category_name: str):
    """
    将 VAD 检测类别名映射为 nuScenes 默认颜色。
    """
    a = ['noise', 'animal', 'human.pedestrian.adult', 'human.pedestrian.child', 'human.pedestrian.construction_worker',
     'human.pedestrian.personal_mobility', 'human.pedestrian.police_officer', 'human.pedestrian.stroller',
     'human.pedestrian.wheelchair', 'movable_object.barrier', 'movable_object.debris',
     'movable_object.pushable_pullable', 'movable_object.trafficcone', 'static_object.bicycle_rack', 'vehicle.bicycle',
     'vehicle.bus.bendy', 'vehicle.bus.rigid', 'vehicle.car', 'vehicle.construction', 'vehicle.emergency.ambulance',
     'vehicle.emergency.police', 'vehicle.motorcycle', 'vehicle.trailer', 'vehicle.truck', 'flat.driveable_surface',
     'flat.other', 'flat.sidewalk', 'flat.terrain', 'static.manmade', 'static.other', 'static.vegetation',
     'vehicle.ego']
    class_names = [
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
        'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ]
    #print(category_name)
    if category_name == 'bicycle':
        return nusc.colormap['vehicle.bicycle']
    elif category_name == 'construction_vehicle':
        return nusc.colormap['vehicle.construction']
    elif category_name == 'traffic_cone':
        return nusc.colormap['movable_object.trafficcone']

    for key in nusc.colormap.keys():
        if category_name in key:
            return nusc.colormap[key]
    return [0, 0, 0]

# TODO: whether to rotate traj
def boxes_to_sensor(boxes: List[EvalBox], pose_record: Dict, cs_record: Dict):
    """
    将一组 EvalBox 从全局坐标系变换到当前传感器坐标系。
    :param boxes: The boxes in global coordinates.
    :param pose_record: The pose record of the vehicle at the current timestamp.
    :param cs_record: The calibrated sensor record of the sensor.
    :return: The transformed boxes.
    """
    boxes_out = []
    for box in boxes:
        # Create Box instance.
        box = CustomNuscenesBox(
            box.translation, box.size, Quaternion(box.rotation), box.fut_trajs, name=box.detection_name
        )
        # Move box to ego vehicle coord system.
        box.translate(-np.array(pose_record['translation']))
        box.rotate(Quaternion(pose_record['rotation']).inverse)
        # Move box to sensor coord system.
        box.translate(-np.array(cs_record['translation']))
        box.rotate(Quaternion(cs_record['rotation']).inverse)

        boxes_out.append(box)

    return boxes_out


def get_gt_fut_trajs(nusc: NuScenes,
                     anno,
                     cs_record,
                     pose_record,
                     fut_ts) -> None:
    """
    获取一个 GT 目标未来 fut_ts 帧的逐步位移及有效掩码。

    若目标后续不再出现，剩余位移和 mask 均为 0。
    """
    box = Box(anno['translation'], anno['size'], Quaternion(anno['rotation']))
    # Move box to ego vehicle coord system.
    box.translate(-np.array(pose_record['translation']))
    box.rotate(Quaternion(pose_record['rotation']).inverse)
    #  Move box to sensor coord system.
    box.translate(-np.array(cs_record['translation']))
    box.rotate(Quaternion(cs_record['rotation']).inverse)
    
    # 在当前 LiDAR 坐标系中沿 annotation.next 向后查找目标。
    gt_fut_trajs = np.zeros((fut_ts, 2))  # [fut_ts*2]
    gt_fut_masks = np.zeros((fut_ts))  # [fut_ts]
    gt_fut_trajs[:] = box.center[:2]
    cur_box = box
    cur_anno = anno
    for i in range(fut_ts):
        if cur_anno['next'] != '':
            anno_next = nusc.get('sample_annotation', cur_anno['next'])
            box_next = Box(
                anno_next['translation'], anno_next['size'], Quaternion(anno_next['rotation'])
            )
            # Move box to ego vehicle coord system.
            box_next.translate(-np.array(pose_record['translation']))
            box_next.rotate(Quaternion(pose_record['rotation']).inverse)
            #  Move box to sensor coord system.
            box_next.translate(-np.array(cs_record['translation']))
            box_next.rotate(Quaternion(cs_record['rotation']).inverse)
            # gt_fut_trajs[i] = box_next.center[:2]
            gt_fut_trajs[i] = box_next.center[:2] - cur_box.center[:2]
            gt_fut_masks[i] = 1
            cur_anno = anno_next
            cur_box = box_next
        else:
            # gt_fut_trajs[i:] = gt_fut_trajs[i-1]
            gt_fut_trajs[i:] = 0
            break         

    return gt_fut_trajs.reshape(-1).tolist(), gt_fut_masks.reshape(-1).tolist()

def get_gt_vec_maps(
    sample_token,
    data_root='data/nuscenes/',
    pc_range=[-15.0, -30.0, -4.0, 15.0, 30.0, 4.0],
    padding_value=-10000,
    map_classes=['divider', 'ped_crossing', 'boundary'],
    map_fixed_ptsnum_per_line=20
) -> None:
    """
    提取指定 sample 周围的 GT 矢量地图元素及类别标签。

    当前视频主流程只绘制预测地图，没有直接调用本函数。
    """
    sample_rec = nusc.get('sample', sample_token)
    sd_record = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
    lidar2ego_translation = cs_record['translation'],
    lidar2ego_rotation = cs_record['rotation'],
    ego2global_translation = pose_record['translation'],
    ego2global_rotation = pose_record['rotation'],
    map_location = nusc.get('log', nusc.get('scene', sample_rec['scene_token'])['log_token'])['location']

    # 组合 lidar->ego 与 ego->global，得到 LiDAR 的全局位姿。
    lidar2ego = np.eye(4)
    lidar2ego[:3,:3] = Quaternion(cs_record['rotation']).rotation_matrix
    lidar2ego[:3, 3] = cs_record['translation']
    ego2global = np.eye(4)
    ego2global[:3,:3] = Quaternion(pose_record['rotation']).rotation_matrix
    ego2global[:3, 3] = pose_record['translation']
    lidar2global = ego2global @ lidar2ego
    lidar2global_translation = list(lidar2global[:3,3])
    lidar2global_rotation = list(Quaternion(matrix=lidar2global).q)
    patch_h = pc_range[4]-pc_range[1]
    patch_w = pc_range[3]-pc_range[0]
    patch_size = (patch_h, patch_w)

    vector_map = VectorizedLocalMap(data_root, patch_size=patch_size,
                                    map_classes=map_classes, 
                                    fixed_ptsnum_per_line=map_fixed_ptsnum_per_line,
                                    padding_value=padding_value)


    anns_results = vector_map.gen_vectorized_samples(
        map_location, lidar2global_translation, lidar2global_rotation
    )
    
    '''
    anns_results, type: dict
        'gt_vecs_pts_loc': list[num_vecs], vec with num_points*2 coordinates
        'gt_vecs_pts_num': list[num_vecs], vec with num_points
        'gt_vecs_label': list[num_vecs], vec with cls index
    '''
    gt_vecs_label = to_tensor(anns_results['gt_vecs_label'])
    if isinstance(anns_results['gt_vecs_pts_loc'], LiDARInstanceLines):
        gt_vecs_pts_loc = anns_results['gt_vecs_pts_loc']
    else:
        gt_vecs_pts_loc = to_tensor(anns_results['gt_vecs_pts_loc'])
        try:
            gt_vecs_pts_loc = gt_vecs_pts_loc.flatten(1).to(dtype=torch.float32)
        except:
            gt_vecs_pts_loc = gt_vecs_pts_loc
    
    return gt_vecs_pts_loc, gt_vecs_label


def visualize_sample(nusc: NuScenes,
                     sample_token: str,
                     gt_boxes: EvalBoxes,
                     pred_boxes: EvalBoxes,
                     nsweeps: int = 1,
                     conf_th: float = 0.4,
                     pc_range: list = [-30.0, -30.0, -4.0, 30.0, 30.0, 4.0],
                     verbose: bool = True,
                     savepath: str = None,
                     traj_use_perstep_offset: bool = True,
                     data_root='data/nuscenes/',
                     map_pc_range: list = [-15.0, -30.0, -4.0, 15.0, 30.0, 4.0],
                     padding_value=-10000,
                     map_classes=['divider', 'ped_crossing', 'boundary'],
                     map_fixed_ptsnum_per_line=20,
                     gt_format=['fixed_num_pts'],
                     colors_plt = ['cornflowerblue', 'royalblue', 'slategrey'],
                     pred_data = None) -> None:
    """
    绘制一帧 VAD 鸟瞰预测图并保存为 bev_pred.png。

    内容包括预测矢量地图、目标框与未来轨迹，以及自车规划轨迹。
    :param nusc: NuScenes object.
    :param sample_token: The nuScenes sample token.
    :param gt_boxes: Ground truth boxes grouped by sample.
    :param pred_boxes: Prediction grouped by sample.
    :param nsweeps: Number of sweeps used for lidar visualization.
    :param conf_th: The confidence threshold used to filter negatives.
    :param eval_range: Range in meters beyond which boxes are ignored.
    :param verbose: Whether to print to stdout.
    :param savepath: If given, saves the the rendering here instead of displaying.
    """
    # 获取顶置 LiDAR 标定和车辆位姿，供 global -> lidar 变换使用。
    sample_rec = nusc.get('sample', sample_token)
    sd_record = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
    # 取出当前帧 GT 和预测框，并统一转换到 LiDAR 坐标系。
    boxes_gt_global = gt_boxes[sample_token]
    boxes_est_global = pred_boxes[sample_token]
    # Map GT boxes to lidar.
    boxes_gt = boxes_to_sensor(boxes_gt_global, pose_record, cs_record)
    # Map EST boxes to lidar.
    boxes_est = boxes_to_sensor(boxes_est_global, pose_record, cs_record)
    # Add scores to EST boxes.
    for box_est, box_est_global in zip(boxes_est, boxes_est_global):
        box_est.score = box_est_global.detection_score

    # 初始化以 LiDAR 为中心、前后左右各 30 m 的 BEV 画布。
    fig, axes = plt.subplots(1, 1, figsize=(4, 4))
    plt.xlim(xmin=-30, xmax=30)
    plt.ylim(ymin=-30, ymax=30)

    # 1) 绘制预测矢量地图，仅保留置信度不低于 0.6 的折线。
    result_dic = pred_data['map_results'][sample_token]['vectors']

    for vector in result_dic:
        if vector['confidence_level'] < 0.6:
            continue
        pred_pts_3d = vector['pts']
        pred_label_3d = vector['type']
        pts_x = np.array([pt[0] for pt in pred_pts_3d])
        pts_y = np.array([pt[1] for pt in pred_pts_3d])

        axes.plot(pts_x, pts_y, color=colors_plt[pred_label_3d],linewidth=1,alpha=0.8,zorder=-1)
        axes.scatter(pts_x, pts_y, color=colors_plt[pred_label_3d],s=1,alpha=0.8,zorder=-1)  

    # ignore_list = ['barrier', 'motorcycle', 'bicycle', 'traffic_cone']
    ignore_list = ['barrier', 'bicycle', 'traffic_cone']

    # 2) 绘制预测目标框及未来轨迹，并按类别、分数和范围过滤。
    for i, box in enumerate(boxes_est):
        if box.name in ignore_list:
            continue
        # Show only predictions with a high score.
        assert not np.isnan(box.score), 'Error: Box score cannot be NaN!'
        if box.score < conf_th or abs(box.center[0]) > 15 or abs(box.center[1]) > 30:
            continue
        box.render(axes, view=np.eye(4), colors=('tomato', 'tomato', 'tomato'), linewidth=1, box_idx=None)
        # if box.name in ['pedestrian']:
        #     continue
        if traj_use_perstep_offset:
            mode_idx = [0, 1, 2, 3, 4, 5]
            box.render_fut_trajs_grad_color(axes, linewidth=1, mode_idx=mode_idx, fut_ts=6, cmap='autumn')
        else:
            box.render_fut_trajs_coords(axes, color='tomato', linewidth=1)

    # 3) 绘制自车轮廓和自车规划轨迹。
    axes.plot([-0.9, -0.9], [-2, 2], color='mediumseagreen', linewidth=1, alpha=0.8)
    axes.plot([-0.9, 0.9], [2, 2], color='mediumseagreen', linewidth=1, alpha=0.8)
    axes.plot([0.9, 0.9], [2, -2], color='mediumseagreen', linewidth=1, alpha=0.8)
    axes.plot([0.9, -0.9], [-2, -2], color='mediumseagreen', linewidth=1, alpha=0.8)
    axes.plot([0.0, 0.0], [0.0, 2], color='mediumseagreen', linewidth=1, alpha=0.8)
    # 选择导航命令对应的分支；逐步位移累加后才是轨迹坐标。
    plan_cmd = np.argmax(pred_data['plan_results'][sample_token][1][0,0,0])
    plan_traj = pred_data['plan_results'][sample_token][0][plan_cmd]
    plan_traj[abs(plan_traj) < 0.01] = 0.0
    plan_traj = plan_traj.cumsum(axis=0)
    plan_traj = np.concatenate((np.zeros((1, plan_traj.shape[1])), plan_traj), axis=0)
    plan_traj = np.stack((plan_traj[:-1], plan_traj[1:]), axis=1)

    # 将轨迹段细分，以便绘制连续的时间渐变色。
    plan_vecs = None
    for i in range(plan_traj.shape[0]):
        plan_vec_i = plan_traj[i]
        x_linspace = np.linspace(plan_vec_i[0, 0], plan_vec_i[1, 0], 51)
        y_linspace = np.linspace(plan_vec_i[0, 1], plan_vec_i[1, 1], 51)
        xy = np.stack((x_linspace, y_linspace), axis=1)
        xy = np.stack((xy[:-1], xy[1:]), axis=1)
        if plan_vecs is None:
            plan_vecs = xy
        else:
            plan_vecs = np.concatenate((plan_vecs, xy), axis=0)

    cmap = 'winter'
    y = np.sin(np.linspace(1/2*np.pi, 3/2*np.pi, 301))
    colors = color_map(y[:-1], cmap)
    line_segments = LineCollection(plan_vecs, colors=colors, linewidths=1, linestyles='solid', cmap=cmap)
    axes.add_collection(line_segments)

    axes.axes.xaxis.set_ticks([])
    axes.axes.yaxis.set_ticks([])
    axes.axis('off')
    fig.set_tight_layout(True)
    fig.canvas.draw()
    plt.savefig(savepath+'/bev_pred.png', bbox_inches='tight', dpi=200)
    plt.close()


def obtain_sensor2top(nusc,
                      sensor_token,
                      l2e_t,
                      l2e_r_mat,
                      e2g_t,
                      e2g_r_mat,
                      sensor_type='lidar'):
    """计算任意传感器坐标系到当前顶置 LiDAR 的旋转和平移。

    变换链为 sensor -> ego -> global -> 当前 ego -> LIDAR_TOP。

    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        sensor_token (str): Sample data token corresponding to the
            specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str): Sensor to calibrate. Default: 'lidar'.

    Returns:
        sweep (dict): Sweep information after transformation.
    """
    sd_rec = nusc.get('sample_data', sensor_token)
    cs_record = nusc.get('calibrated_sensor',
                         sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    data_path = str(nusc.get_sample_data_path(sd_rec['token']))
    if os.getcwd() in data_path:  # path from lyftdataset is absolute path
        data_path = data_path.split(f'{os.getcwd()}/')[-1]  # relative path
    sweep = {
        'data_path': data_path,
        'type': sensor_type,
        'sample_data_token': sd_rec['token'],
        'sensor2ego_translation': cs_record['translation'],
        'sensor2ego_rotation': cs_record['rotation'],
        'ego2global_translation': pose_record['translation'],
        'ego2global_rotation': pose_record['rotation'],
        'timestamp': sd_rec['timestamp']
    }

    l2e_r_s = sweep['sensor2ego_rotation']
    l2e_t_s = sweep['sensor2ego_translation']
    e2g_r_s = sweep['ego2global_rotation']
    e2g_t_s = sweep['ego2global_translation']

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                  ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    sensor2lidar_rotation = R.T  # points @ R.T + T
    sensor2lidar_translation = T

    return sensor2lidar_rotation, sensor2lidar_translation

def render_sample_data(
        sample_toekn: str,
        with_anns: bool = True,
        box_vis_level: BoxVisibility = BoxVisibility.ANY,
        axes_limit: float = 40,
        ax=None,
        nsweeps: int = 1,
        out_path: str = None,
        out_name: str = None,
        underlay_map: bool = True,
        use_flat_vehicle_coordinates: bool = True,
        show_lidarseg: bool = False,
        show_lidarseg_legend: bool = False,
        filter_lidarseg_labels=None,
        lidarseg_preds_bin_path: str = None,
        verbose: bool = True,
        show_panoptic: bool = False,
        pred_data=None,
        traj_use_perstep_offset: bool = True
      ) -> None:
    """
    单帧 BEV 渲染入口；当前实现最终转调 lidiar_render。
    :param sample_data_token: Sample_data token.
    :param with_anns: Whether to draw box annotations.
    :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
    :param axes_limit: Axes limit for lidar and radar (measured in meters).
    :param ax: Axes onto which to render.
    :param nsweeps: Number of sweeps for lidar and radar.
    :param out_path: Optional path to save the rendered figure to disk.
    :param underlay_map: When set to true, lidar data is plotted onto the map. This can be slow.
    :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
        aligned to z-plane in the world. Note: Previously this method did not use flat vehicle coordinates, which
        can lead to small errors when the vertical axis of the global frame and lidar are not aligned. The new
        setting is more correct and rotates the plot by ~90 degrees.
    :param show_lidarseg: When set to True, the lidar data is colored with the segmentation labels. When set
        to False, the colors of the lidar data represent the distance from the center of the ego vehicle.
    :param show_lidarseg_legend: Whether to display the legend for the lidarseg labels in the frame.
    :param filter_lidarseg_labels: Only show lidar points which belong to the given list of classes. If None
        or the list is empty, all classes will be displayed.
    :param lidarseg_preds_bin_path: A path to the .bin file which contains the user's lidar segmentation
                                    predictions for the sample.
    :param verbose: Whether to display the image after it is rendered.
    :param show_panoptic: When set to True, the lidar data is colored with the panoptic labels. When set
        to False, the colors of the lidar data represent the distance from the center of the ego vehicle.
        If show_lidarseg is True, show_panoptic will be set to False.
    """
    lidiar_render(sample_toekn, pred_data, out_path=out_path,
                  out_name=out_name, traj_use_perstep_offset=traj_use_perstep_offset)


def parse_args():
    """解析推理结果文件和可视化输出目录。"""
    parser = argparse.ArgumentParser(description='Visualize VAD predictions')
    parser.add_argument('--result-path', help='inference result file path')
    parser.add_argument('--save-path', help='the dir to save visualization results')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    # 推理结果含 results（检测/运动）、map_results（地图）和
    # plan_results（各导航命令对应的自车规划轨迹）。
    args = parse_args()
    inference_result_path = args.result_path
    out_path = args.save_path
    bevformer_results = mmcv.load(inference_result_path)
    sample_token_list = list(bevformer_results['results'].keys())

    nusc = NuScenes(version='v1.0-trainval', dataroot='./data/nuscenes', verbose=True)
    
    # 输出视频由左侧六相机拼图和右侧 BEV 图组成。
    imgs = []
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video_path = osp.join(out_path, 'vis.mp4')
    video = cv2.VideoWriter(video_path, fourcc, 10, (2933, 800), True)
    # 每个 sample_token 生成一帧视频。
    for id in tqdm(range(len(sample_token_list))):
        mmcv.mkdir_or_exist(out_path)
        # 先生成临时 BEV 图片，读入内存后删除。
        render_sample_data(sample_token_list[id],
                           pred_data=bevformer_results,
                           out_path=out_path)
        pred_path = osp.join(out_path, 'bev_pred.png')
        pred_img = cv2.imread(pred_path)
        os.remove(pred_path)

        sample_token = sample_token_list[id]
        sample = nusc.get('sample', sample_token)
        # sample = data['results'][sample_token_list[0]][0]
        # 前三路和后三路相机最终分别拼成一行。
        cams = [
            'CAM_FRONT_LEFT',
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_BACK_LEFT',
            'CAM_BACK',
            'CAM_BACK_RIGHT',
        ]

        cam_imgs = []
        # 逐相机读取原图，并筛选落在画面内的预测框。
        for cam in cams:
            sample_data_token = sample['data'][cam]
            sd_record = nusc.get('sample_data', sample_data_token)
            sensor_modality = sd_record['sensor_modality']
            if sensor_modality in ['lidar', 'radar']:
                assert False
            elif sensor_modality == 'camera':
                boxes = [Box(record['translation'], record['size'], Quaternion(record['rotation']),
                            name=record['detection_name'], token='predicted') for record in
                        bevformer_results['results'][sample_token]]
                data_path, boxes_pred, camera_intrinsic = get_predicted_data(sample_data_token,
                                                                            box_vis_level=BoxVisibility.ANY,
                                                                            pred_anns=boxes)
                _, boxes_gt, _ = nusc.get_sample_data(sample_data_token, box_vis_level=BoxVisibility.ANY)

                data = Image.open(data_path)
 
                # Show image.
                _, ax = plt.subplots(1, 1, figsize=(6, 12))
                ax.imshow(data)

                # 仅在前视相机上投影并绘制自车规划轨迹。
                if cam == 'CAM_FRONT':
                    lidar_sd_record =  nusc.get('sample_data', sample['data']['LIDAR_TOP'])
                    lidar_cs_record = nusc.get('calibrated_sensor', lidar_sd_record['calibrated_sensor_token'])
                    lidar_pose_record = nusc.get('ego_pose', lidar_sd_record['ego_pose_token'])

                    # 累加逐步位移，再补 z=-1、w=1 形成齐次坐标 [x,y,z,1]。
                    plan_cmd = np.argmax(bevformer_results['plan_results'][sample_token][1][0,0,0])
                    plan_traj = bevformer_results['plan_results'][sample_token][0][plan_cmd]
                    plan_traj[abs(plan_traj) < 0.01] = 0.0
                    plan_traj = plan_traj.cumsum(axis=0)

                    plan_traj = np.concatenate((
                        plan_traj[:, [0]],
                        plan_traj[:, [1]],
                        -1.0*np.ones((plan_traj.shape[0], 1)),
                        np.ones((plan_traj.shape[0], 1)),
                    ), axis=1)
                    # add the start point in lcf
                    plan_traj = np.concatenate((np.zeros((1, plan_traj.shape[1])), plan_traj), axis=0)
                    # plan_traj[0, :2] = 2*plan_traj[1, :2] - plan_traj[2, :2]
                    plan_traj[0, 0] = 0.3
                    plan_traj[0, 2] = -1.0
                    plan_traj[0, 3] = 1.0

                    l2e_r = lidar_cs_record['rotation']
                    l2e_t = lidar_cs_record['translation']
                    e2g_r = lidar_pose_record['rotation']
                    e2g_t = lidar_pose_record['translation']
                    l2e_r_mat = Quaternion(l2e_r).rotation_matrix
                    e2g_r_mat = Quaternion(e2g_r).rotation_matrix
                    s2l_r, s2l_t = obtain_sensor2top(nusc, sample_data_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, cam)
                    # 组合外参和内参得到 lidar->image 投影矩阵。
                    lidar2cam_r = np.linalg.inv(s2l_r)
                    lidar2cam_t = s2l_t @ lidar2cam_r.T
                    lidar2cam_rt = np.eye(4)
                    lidar2cam_rt[:3, :3] = lidar2cam_r.T
                    lidar2cam_rt[3, :3] = -lidar2cam_t
                    viewpad = np.eye(4)
                    viewpad[:camera_intrinsic.shape[0], :camera_intrinsic.shape[1]] = camera_intrinsic
                    lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                    plan_traj = lidar2img_rt @ plan_traj.T
                    plan_traj = plan_traj[0:2, ...] / np.maximum(
                        plan_traj[2:3, ...], np.ones_like(plan_traj[2:3, ...]) * 1e-5)
                    plan_traj = plan_traj.T
                    plan_traj = np.stack((plan_traj[:-1], plan_traj[1:]), axis=1)

                    plan_vecs = None
                    for i in range(plan_traj.shape[0]):
                        plan_vec_i = plan_traj[i]
                        x_linspace = np.linspace(plan_vec_i[0, 0], plan_vec_i[1, 0], 51)
                        y_linspace = np.linspace(plan_vec_i[0, 1], plan_vec_i[1, 1], 51)
                        xy = np.stack((x_linspace, y_linspace), axis=1)
                        xy = np.stack((xy[:-1], xy[1:]), axis=1)
                        if plan_vecs is None:
                            plan_vecs = xy
                        else:
                            plan_vecs = np.concatenate((plan_vecs, xy), axis=0)

                    cmap = 'winter'
                    y = np.sin(np.linspace(1/2*np.pi, 3/2*np.pi, 301))
                    colors = color_map(y[:-1], cmap)
                    line_segments = LineCollection(plan_vecs, colors=colors, linewidths=2, linestyles='solid', cmap=cmap)
                    ax.add_collection(line_segments)

                ax.set_xlim(0, data.size[0])
                ax.set_ylim(data.size[1], 0)
                ax.axis('off')
                if out_path is not None:
                    savepath = osp.join(out_path, f'{cam}_PRED')
                    plt.savefig(savepath, bbox_inches='tight', dpi=200, pad_inches=0.0)
                plt.close()

                # Load boxes and image.
                data_path = osp.join(out_path, f'{cam}_PRED.png')
                cam_img = cv2.imread(data_path)
                lw = 6
                tf = max(lw - 3, 1)
                w, h = cv2.getTextSize(cam, 0, fontScale=lw / 6, thickness=tf)[0]  # text width, height
                # color=(0, 0, 0)
                txt_color=(255, 255, 255)
                cv2.putText(cam_img,
                            cam, (10, h + 10),
                            0,
                            lw / 6,
                            txt_color,
                            thickness=tf,
                            lineType=cv2.LINE_AA)
                cam_imgs.append(cam_img)
            else:
                raise ValueError("Error: Unknown sensor modality!")

        plan_cmd = np.argmax(bevformer_results['plan_results'][sample_token][1][0,0,0])
        cmd_list = ['Turn Right', 'Turn Left', 'Go Straight']
        plan_cmd_str = cmd_list[plan_cmd]
        pred_img = cv2.copyMakeBorder(pred_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, None, value = 0)
        # font
        font = cv2.FONT_HERSHEY_SIMPLEX
        # fontScale
        fontScale = 1
        # Line thickness of 2 px
        thickness = 3
        # org
        org = (20, 40)      
        # Blue color in BGR
        color = (0, 0, 0)
        # Using cv2.putText() method
        pred_img = cv2.putText(pred_img, 'BEV', org, font, 
                        fontScale, color, thickness, cv2.LINE_AA)
        pred_img = cv2.putText(pred_img, plan_cmd_str, (20, 770), font, 
                        fontScale, color, thickness, cv2.LINE_AA)
        
        # 六相机图按 2x3 拼接，再与 BEV 图横向合并。
        sample_img = pred_img
        cam_img_top = cv2.hconcat([cam_imgs[0], cam_imgs[1], cam_imgs[2]])
        cam_img_down = cv2.hconcat([cam_imgs[3], cam_imgs[4], cam_imgs[5]])
        cam_img = cv2.vconcat([cam_img_top, cam_img_down])
        size = (2133, 800)
        cam_img = cv2.resize(cam_img, size)
        vis_img = cv2.hconcat([cam_img, sample_img])

        video.write(vis_img)
    
    # 所有帧写完后关闭视频编码器。
    video.release()
    cv2.destroyAllWindows()
