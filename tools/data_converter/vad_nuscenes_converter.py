import os
import math
import copy
import argparse
from os import path as osp
from collections import OrderedDict
from typing import List, Tuple, Union

import mmcv
import numpy as np
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from shapely.geometry import MultiPoint, box
from mmdet3d.datasets import NuScenesDataset
from nuscenes.utils.geometry_utils import view_points
from mmdet3d.core.bbox.box_np_ops import points_cam2img
from nuscenes.utils.geometry_utils import transform_matrix


nus_categories = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                  'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                  'barrier')

nus_attributes = ('cycle.with_rider', 'cycle.without_rider',
                  'pedestrian.moving', 'pedestrian.standing',
                  'pedestrian.sitting_lying_down', 'vehicle.moving',
                  'vehicle.parked', 'vehicle.stopped', 'None')

ego_width, ego_length = 1.85, 4.084

# *1. 基础数学工具：四元数/欧拉角转换，以及按时间戳匹配最近的 CAN bus 消息。
def quart_to_rpy(qua):
    x, y, z, w = qua
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(2 * (w * y - x * z))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (z * z + y * y))
    return roll, pitch, yaw

def locate_message(utimes, utime):
    i = np.searchsorted(utimes, utime)
    if i == len(utimes) or (i > 0 and utime - utimes[i-1] < utimes[i] - utime):
        i -= 1
    return i


def create_nuscenes_infos(root_path,
                          out_path,
                          can_bus_root_path,
                          info_prefix,
                          version='v1.0-trainval',
                          max_sweeps=10):
    """Create info file of nuscene dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str): Version of the data.
            Default: 'v1.0-trainval'
        max_sweeps (int): Max number of sweeps.
            Default: 10
    """
    # *==============================================================#
    # *1. 初始化 nuScenes 主数据库和 CAN bus expansion 数据接口。
    from nuscenes.nuscenes import NuScenes
    from nuscenes.can_bus.can_bus_api import NuScenesCanBus
    print(version, root_path)
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    nusc_can_bus = NuScenesCanBus(dataroot=can_bus_root_path)
    # *==============================================================#
    # *2. 根据数据版本取得官方 train/val/test 场景名称划分。
    from nuscenes.utils import splits
    available_vers = ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    assert version in available_vers
    if version == 'v1.0-trainval':
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == 'v1.0-test':
        train_scenes = splits.test
        val_scenes = []
    elif version == 'v1.0-mini':
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise ValueError('unknown')

    # *==============================================================#
    # *3. 检查本地 LiDAR 文件，只保留数据实际存在的场景，并把场景名称转换成 scene token。
    available_scenes = get_available_scenes(nusc)
    available_scene_names = [s['name'] for s in available_scenes]
    train_scenes = list(
        filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in train_scenes
    ])
    val_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in val_scenes
    ])

    test = 'test' in version
    if test:
        print('test scene: {}'.format(len(train_scenes)))
    else:
        print('train scene: {}, val scene: {}'.format(
            len(train_scenes), len(val_scenes)))
    # *==============================================================#
    # *4. 遍历全部 key frame，生成相机/点云标定、目标标注、轨迹和规划标签等逐帧 info。
    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(
        nusc, nusc_can_bus, train_scenes, val_scenes, test, max_sweeps=max_sweeps)
    # *==============================================================#
    # *5. 按版本写出 temporal_train/val/test.pkl；顶层格式为 {'infos': list, 'metadata': dict}。
    metadata = dict(version=version)
    if test:
        print('test sample: {}'.format(len(train_nusc_infos)))
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(out_path,
                             '{}_infos_temporal_test.pkl'.format(info_prefix))
        mmcv.dump(data, info_path)
    else:
        print('train sample: {}, val sample: {}'.format(
            len(train_nusc_infos), len(val_nusc_infos)))
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(out_path,
                             '{}_infos_temporal_train.pkl'.format(info_prefix))
        mmcv.dump(data, info_path)
        data['infos'] = val_nusc_infos
        info_val_path = osp.join(out_path,
                                 '{}_infos_temporal_val.pkl'.format(info_prefix))
        mmcv.dump(data, info_val_path)


def get_available_scenes(nusc):
    """Get available scenes from the input nuscenes class.

    Given the raw data, get the information of available scenes for
    further info generation.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.

    Returns:
        available_scenes (list[dict]): List of basic information for the
            available scenes.
    """
    # *1. 遍历 scene，并定位每个场景首个 sample 的 LIDAR_TOP 数据。
    available_scenes = []
    print('total scene num: {}'.format(len(nusc.scene)))
    for scene in nusc.scene:
        scene_token = scene['token']
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        has_more_frames = True
        scene_not_exist = False
        # *2. 检查对应点云文件是否真实存在；缺文件的 scene 不参与后续 pkl 生成。
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token'])
            lidar_path = str(lidar_path)
            if os.getcwd() in lidar_path:
                # path from lyftdataset is absolute path
                lidar_path = lidar_path.split(f'{os.getcwd()}/')[-1]
                # relative path
            if not mmcv.is_filepath(lidar_path):
                scene_not_exist = True
                break
            else:
                break
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print('exist scene num: {}'.format(len(available_scenes)))
    return available_scenes


def _get_can_bus_info(nusc, nusc_can_bus, sample):
    # *1. 使用 sample 所属 scene 名称读取该场景的 CAN bus pose 消息序列。
    scene_name = nusc.get('scene', sample['scene_token'])['name']
    sample_timestamp = sample['timestamp']
    try:
        pose_list = nusc_can_bus.get_messages(scene_name, 'pose')
    except:
        return np.zeros(18)  # server scenes do not have can bus information.
    # *2. 在有序消息中寻找时间上最接近当前 sample 的 pose；若 CAN bus 不可用则返回全零特征。
    can_bus = []
    # during each scene, the first timestamp of can_bus may be large than the first sample's timestamp
    last_pose = pose_list[0]
    for i, pose in enumerate(pose_list):
        if pose['utime'] > sample_timestamp:
            break
        last_pose = pose
    # *3. 按 BEVFormer/VAD 约定把位置、四元数和运动状态打包，并预留两个航向角槽位，共 18 维。
    _ = last_pose.pop('utime')  # useless
    pos = last_pose.pop('pos')
    rotation = last_pose.pop('orientation')
    can_bus.extend(pos)
    can_bus.extend(rotation)
    for key in last_pose.keys():
        can_bus.extend(pose[key])  # 16 elements
    can_bus.extend([0., 0.])
    return np.array(can_bus)


def _fill_trainval_infos(nusc,
                         nusc_can_bus,
                         train_scenes,
                         val_scenes,
                         test=False,
                         max_sweeps=10,
                         fut_ts=6,
                         his_ts=2):
    """Generate the train/val infos from the raw data.

    Args:
        nusc (:obj:`NuScenes`): Dataset class in the nuScenes dataset.
        train_scenes (list[str]): Basic information of training scenes.
        val_scenes (list[str]): Basic information of validation scenes.
        test (bool): Whether use the test mode. In the test mode, no
            annotations can be accessed. Default: False.
        max_sweeps (int): Max number of sweeps. Default: 10.

    Returns:
        tuple[list[dict]]: Information of training set and validation set
            that will be saved to the info file.
    """
    # *==============================================================#
    # *1. 初始化输出容器，并建立 nuScenes 原始类别名称到整数类别 ID 的映射。
    train_nusc_infos = []
    val_nusc_infos = []
    frame_idx = 0
    cat2idx = {}
    for idx, dic in enumerate(nusc.category):
        cat2idx[dic['name']] = idx
    # *==============================================================#
    # *2. 逐个遍历 nuScenes key frame；每个 sample 最终对应 pkl 中的一条 info。
    for sample in mmcv.track_iter_progress(nusc.sample):
        # *2.1 读取当前帧的地图区域、LIDAR_TOP 标定、ego pose，以及相邻帧 ego pose。
        map_location = nusc.get('log', nusc.get('scene', sample['scene_token'])['log_token'])['location']
        lidar_token = sample['data']['LIDAR_TOP']
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        cs_record = nusc.get('calibrated_sensor',
                             sd_rec['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])

        # *============================================================#
        #* sample['prev'] 和 sample['next'] 保存的是同一 scene 中相邻 nuScenes key frame 的
        #* sample token，而不是列表下标：prev 指向上一关键帧，next 指向下一关键帧。
        #* 场景首帧没有上一帧，因此 prev == ''；场景末帧没有下一帧，因此 next == ''。
        #* 这里按 sample token 查询相邻 sample，再取其 LIDAR_TOP sample_data，最后利用
        #* sample_data['ego_pose_token'] 取得该 LiDAR 采集时刻的自车位置和朝向。
        #* 这些相邻 key frame ego pose 后续用于估算自车速度和 yaw_rate；nuScenes key frame
        #* 通常间隔约 0.5 s。注意这里的 sample.prev/next 不同于后面用于收集高频 LiDAR
        #* 历史 sweep 的 sample_data.prev：前者连接关键帧，后者连接同一传感器的数据帧。
        if sample['prev'] != '':
            #* 非场景首帧：上一 key frame sample -> 上一帧 LIDAR_TOP -> 上一帧 ego pose。
            sample_prev = nusc.get('sample', sample['prev'])
            sd_rec_prev = nusc.get('sample_data', sample_prev['data']['LIDAR_TOP'])
            pose_record_prev = nusc.get('ego_pose', sd_rec_prev['ego_pose_token'])
        else:
            #* 场景首帧无 previous，后续运动状态估算会改用当前帧与下一帧。
            pose_record_prev = None
        if sample['next'] != '':
            #* 非场景末帧：下一 key frame sample -> 下一帧 LIDAR_TOP -> 下一帧 ego pose。
            sample_next = nusc.get('sample', sample['next'])
            sd_rec_next = nusc.get('sample_data', sample_next['data']['LIDAR_TOP'])
            pose_record_next = nusc.get('ego_pose', sd_rec_next['ego_pose_token'])
        else:
            #* 场景末帧无 next；未来轨迹生成时会据此停止并将后续 mask 置为无效。
            pose_record_next = None

        # *2.2 读取当前 LiDAR 文件路径、3D GT box，并匹配当前时刻的 CAN bus 状态。
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)

        mmcv.check_file_exist(lidar_path)
        can_bus = _get_can_bus_info(nusc, nusc_can_bus, sample)

        # *==============================================================#
        #* fut_valid_flag 是 VAD 额外生成的“完整未来是否可用”标志：从当前 sample 沿 next
        #* 连续检查 fut_ts（默认 6）个 key frame。只要场景在预测范围内提前结束，就置 False。
        #* 注意它是整段轨迹级别的 bool；逐时间步是否有效由后面的 gt_ego_fut_masks 表示。
        fut_valid_flag = True
        test_sample = copy.deepcopy(sample)
        for i in range(fut_ts):
            if test_sample['next'] != '':
                test_sample = nusc.get('sample', test_sample['next'])
            else:
                fut_valid_flag = False
        # *==============================================================#
        # *3. 建立当前帧 info 的基础部分：token、时序关系、标定、位姿、时间戳和地图区域。
        info = {
            'lidar_path': lidar_path,
            'token': sample['token'],
            'prev': sample['prev'],
            'next': sample['next'],
            'can_bus': can_bus,
            'frame_idx': frame_idx,  # temporal related info
            'sweeps': [],
            'cams': dict(),
            'scene_token': sample['scene_token'],  # temporal related info
            'lidar2ego_translation': cs_record['translation'],
            'lidar2ego_rotation': cs_record['rotation'],
            'ego2global_translation': pose_record['translation'],
            'ego2global_rotation': pose_record['rotation'],
            'timestamp': sample['timestamp'],
            'fut_valid_flag': fut_valid_flag,
            'map_location': map_location
        }
        # *==============================================================#
        # *4. 更新场景内帧编号：遇到场景末帧后清零，否则递增。
        if sample['next'] == '':
            frame_idx = 0
        else:
            frame_idx += 1

        l2e_r = info['lidar2ego_rotation']
        l2e_t = info['lidar2ego_translation']
        e2g_r = info['ego2global_rotation']
        e2g_t = info['ego2global_translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix
        # *==============================================================#
        # *5. 收集六路相机信息，并计算每个相机到当前 LIDAR_TOP 的外参及相机内参。
        camera_types = [
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_FRONT_LEFT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT',
        ]
        for cam in camera_types:
            cam_token = sample['data'][cam]
            cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token)
            cam_info = obtain_sensor2top(nusc, cam_token, l2e_t, l2e_r_mat,
                                         e2g_t, e2g_r_mat, cam)
            cam_info.update(cam_intrinsic=cam_intrinsic)
            info['cams'].update({cam: cam_info})
        # *==============================================================#
        # *6. 沿 LIDAR_TOP sample_data.prev 收集最多 max_sweeps 个历史点云及其到当前帧的变换。
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        sweeps = []
        while len(sweeps) < max_sweeps:
            if not sd_rec['prev'] == '':
                sweep = obtain_sensor2top(nusc, sd_rec['prev'], l2e_t,
                                          l2e_r_mat, e2g_t, e2g_r_mat, 'lidar')
                sweeps.append(sweep)
                sd_rec = nusc.get('sample_data', sd_rec['prev'])
            else:
                break
        info['sweeps'] = sweeps

        # *==============================================================#
        # *7. train/val 才生成监督标注；test 无公开 GT，因此只保留前面的传感器和时序信息。
        if not test:
            # *==============================================================#
            # *7.1 读取当前 sample 的全部 3D annotation，生成 box、类别、速度和有效性标志。
            annotations = [
                nusc.get('sample_annotation', token)
                for token in sample['anns']
            ]
            locs = np.array([b.center for b in boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
            rots = np.array([b.orientation.yaw_pitch_roll[0]
                             for b in boxes]).reshape(-1, 1)
            velocity = np.array(
                [nusc.box_velocity(token)[:2] for token in sample['anns']])
            valid_flag = np.array(
                [(anno['num_lidar_pts'] + anno['num_radar_pts']) > 0
                 for anno in annotations],
                dtype=bool).reshape(-1)

            # *==============================================================#
            # *7.2 将目标速度从 global 坐标系旋转到当前 LiDAR 坐标系。
            for i in range(len(boxes)):
                velo = np.array([*velocity[i], 0.0])
                velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
                    l2e_r_mat).T
                velocity[i] = velo[:2]

            names = [b.name for b in boxes]
            for i in range(len(names)):
                if names[i] in NuScenesDataset.NameMapping:
                    names[i] = NuScenesDataset.NameMapping[names[i]]
            names = np.array(names)

            # *==============================================================#
            # *7.3 将 nuScenes Box 整理成 SECOND/mmdet3d 的 7 维 LiDAR box 表示。
            #*
            #* 转换前，前面三组数组分别是：
            #*   locs: (N, 3)，每个 box 在当前 LIDAR_TOP 坐标系中的中心 [x, y, z]；
            #*   dims: (N, 3)，nuScenes Box.wlh 给出的尺寸 [width, length, height]；
            #*   rots: (N, 1)，nuScenes Box 的 yaw（orientation.yaw_pitch_roll[0]），单位 rad。
            #*
            #* 转换后，每行 gt_boxes 为：
            #*   [x, y, z, width, length, height, yaw_mmdet]，形状为 (N, 7)。
            #* 这里 locs 和 dims 直接拼接，没有交换尺寸列；关键转换发生在 yaw：
            #*
            #*   yaw_mmdet = -yaw_nuscenes - pi/2
            #*
            #* 其中负号用于对齐两套 box 对旋转正方向的定义，额外的 -pi/2 用于对齐
            #* nuScenes Box.wlh 与 SECOND/mmdet3d 对“零航向及长宽轴”的定义。二者共同保证
            #* 转换后的 3D box 在 LiDAR 坐标系中仍覆盖同一物理物体，而不是把物体真的旋转。
            #* 例如 nuScenes yaw=0 时，这里得到 mmdet yaw=-pi/2；这是同一 box 的不同参数化。
            gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)
            #* boxes 与 annotations 都由当前 sample 的 GT 产生，数量和顺序必须一一对应；
            #* 后续才能用同一个下标拼接类别、速度、有效标志和未来轨迹。
            assert len(gt_boxes) == len(
                annotations), f'{len(gt_boxes)}, {len(annotations)}'

            # *==============================================================#
            # *8. 沿每个实例 annotation.next 追踪未来运动，生成周围目标的轨迹、mask、yaw 和低层特征。
            #* nuScenes 不需要用几何距离做跨帧匹配：同一个物体在不同 key frame 中的
            #* sample_annotation 已经通过 prev/next 串成实例链。因此从当前 anno 开始不断读取
            #* cur_anno['next']，就能取得这个物体在下一帧、下两帧……的真实标注。
            #* 这里追踪的是当前帧实际存在的 num_box 个目标，每个目标最多向未来追踪
            #* fut_ts（默认 6）个 key frame，约对应未来 3 秒。
            #*
            # *8.1 为所有目标预分配未来监督数组。
            #*   gt_fut_trajs: (N, 6, 2)，相邻未来时刻的中心位移 [dx, dy]；
            #*   gt_fut_yaw:   (N, 6)，相邻未来时刻的 yaw 变化量；
            #*   gt_fut_masks: (N, 6)，该目标在对应未来时刻是否仍有有效 annotation；
            #*   gt_fut_goal:  (N,)，根据整段未来运动方向生成的离散目标类别。
            #* 数组先初始化为 0，所以实例提前消失或追踪链结束后，剩余轨迹/mask/yaw 自然保持 0。
            num_box = len(boxes)
            gt_fut_trajs = np.zeros((num_box, fut_ts, 2)) # (N 6 2)
            gt_fut_yaw = np.zeros((num_box, fut_ts)) # (N 6)
            gt_fut_masks = np.zeros((num_box, fut_ts)) # (N)
            #* gt_boxes 中 yaw_mmdet = -yaw_nuscenes-pi/2，这里执行逆关系恢复当前 box 的
            #* nuScenes/LiDAR yaw，作为 agent_lcf_feat 中的朝向特征。
            gt_boxes_yaw = -(gt_boxes[:,6] + np.pi / 2)
            #* 当前帧每个目标的 9 维低层特征：
            #* [x, y, yaw, vx, vy, width, length, height, category_id]。
            agent_lcf_feat = np.zeros((num_box, 9))
            gt_fut_goal = np.zeros((num_box))
            for i, anno in enumerate(annotations):
                #* boxes[i] 与 annotations[i] 一一对应。cur_box/cur_anno 表示追踪链的“当前节点”；
                #* 首次进入循环时是 t=0 当前帧，之后每轮更新为刚取得的未来帧节点。
                cur_box = boxes[i]
                cur_anno = anno
                #* agent_lcf_feat 只描述 t=0 的目标状态，不会随下面的未来追踪循环更新。
                agent_lcf_feat[i, 0:2] = cur_box.center[:2]
                agent_lcf_feat[i, 2] = gt_boxes_yaw[i]
                agent_lcf_feat[i, 3:5] = velocity[i]
                agent_lcf_feat[i, 5:8] = anno['size']  # [width, length, height]
                agent_lcf_feat[i, 8] = (
                    cat2idx[anno['category_name']]
                    if anno['category_name'] in cat2idx.keys() else -1)

                #* 内层循环只追踪外层第 i 个物体，不是遍历未来帧中的所有目标。
                #* cur_anno 从该物体 t0 时刻的 annotation 开始，每轮沿 annotation.next
                #* 前进到同一 instance 的下一条关键帧标注。默认 fut_ts=6、关键帧约间隔 0.5 s：
                #*   j=0：检查 t0 -> t1（未来约 0.5 s），保存 position(t1)-position(t0)；
                #*   j=1：检查 t1 -> t2（未来约 1.0 s），保存 position(t2)-position(t1)；
                #*   ...
                #*   j=5：检查 t5 -> t6（未来约 3.0 s），保存 position(t6)-position(t5)。
                #* 因此一次完整循环得到该物体未来 6 步的“逐步位移”，而不是 6 个绝对位置。
                #* cur_anno['next'] 是同一物体的下一条 annotation token；它不同于
                #* sample['next']（后者表示整个场景的下一关键帧 sample token）。
                for j in range(fut_ts):
                    #* next 非空说明同一物体还有下一时刻 GT，可继续生成第 j 步轨迹监督；
                    #* next 为空则说明实例链已经结束，剩余未来步无法提供有效监督。
                    if cur_anno['next'] != '':
                        #* annotation.next 指向同一 instance 在下一个有标注 key frame 中的 annotation，
                        #* 因此这里无需使用 box IoU、距离或 tracking ID 再做一次数据关联。
                        anno_next = nusc.get('sample_annotation', cur_anno['next'])
                        #* annotation 的 translation/rotation 原生表达在 global 坐标系；先据此构造
                        #* 下一时刻的 global Box。
                        box_next = Box(
                            anno_next['translation'], anno_next['size'], Quaternion(anno_next['rotation'])
                        )
                        #* global ->“当前 t=0 帧”ego。这里始终使用外层当前 sample 的 pose_record，
                        #* 而不是未来帧 ego pose，使所有未来 box 都落在同一个固定参考坐标系中。
                        box_next.translate(-np.array(pose_record['translation']))
                        box_next.rotate(Quaternion(pose_record['rotation']).inverse)
                        #* 当前帧 ego -> 当前帧 LIDAR_TOP。至此，box_next 与 cur_box 均表达在
                        #* 当前 t=0 LiDAR 坐标系，二者中心可以直接相减。
                        box_next.translate(-np.array(cs_record['translation']))
                        box_next.rotate(Quaternion(cs_record['rotation']).inverse)
                        #* 保存一步增量，而非相对 t=0 的累计位移：
                        #*   j=0: position(t1)-position(t0)
                        #*   j=1: position(t2)-position(t1)，依此类推。
                        gt_fut_trajs[i, j] = box_next.center[:2] - cur_box.center[:2]
                        #* 能取得下一 annotation，说明第 j 个未来监督有效。
                        gt_fut_masks[i, j] = 1
                        #* 同样保存相邻时刻 yaw 增量 yaw(t+j+1)-yaw(t+j)。
                        _, _, box_yaw = quart_to_rpy([cur_box.orientation.x, cur_box.orientation.y,
                                                      cur_box.orientation.z, cur_box.orientation.w])
                        _, _, box_yaw_next = quart_to_rpy([box_next.orientation.x, box_next.orientation.y,
                                                           box_next.orientation.z, box_next.orientation.w])
                        gt_fut_yaw[i, j] = box_yaw_next - box_yaw
                        #* 沿实例链向前移动一步；下一轮将继续寻找 t+j+2。
                        cur_anno = anno_next
                        cur_box = box_next
                    else:
                        #* next == '' 表示该实例已没有后续标注（场景结束或目标不再存在）。
                        #* 剩余轨迹显式补 0；mask/yaw 初始化时已经是 0，无需再次赋值。
                        gt_fut_trajs[i, j:] = 0
                        break
                # *==============================================================#
                # *8.2 将逐步位移累加成相对轨迹，再把整段运动方向量化为 goal 类别。
                #* cumsum 后 gt_fut_coords[k] 表示从 t=0 累积到第 k+1 个未来时刻的位置偏移。
                gt_fut_coords = np.cumsum(gt_fut_trajs[i], axis=-2)
                #* 当前实现用“最后累计点 - 第一个累计点”计算方向，相当于 t1 到最后有效/补齐
                #* 时刻的位移，而不是严格的 t0 到末时刻位移。
                coord_diff = gt_fut_coords[-1] - gt_fut_coords[0]
                #* 注意：代码使用 coord_diff.max() < 1.0 判静止，并非位移范数，也没有取绝对值；
                #* 这里保留原实现，仅说明其实际判定规则。
                if coord_diff.max() < 1.0:  # static
                    gt_fut_goal[i] = 9
                else:
                    #* atan2 得到运动方向角，加 pi 将范围平移到约 [0, 2*pi]；再以 pi/4
                    #* 为间隔整除，量化为八个 45° 方向 bin。极端边界值可能得到类别 8，
                    #* 静止类别固定为 9，这与原代码注释中的 0-8 范围保持一致。
                    box_mot_yaw = np.arctan2(coord_diff[1], coord_diff[0]) + np.pi
                    gt_fut_goal[i] = box_mot_yaw // (np.pi / 4)  # 0-8: goal direction class

            # *9. 生成自车历史轨迹 gt_ego_his_trajs。
            #*==================== 自车历史轨迹 gt_ego_his_trajs ====================#
            #* 原始来源：各历史 sample 的 ego_pose 与 LIDAR_TOP calibrated_sensor。
            #* get_global_sensor_pose() 返回该帧 LiDAR 原点在 global 坐标系中的 4x4 位姿；
            #* 此处先保存当前帧以及之前 his_ts（默认 2）帧的 global 三维位置，共 his_ts+1 个点。
            #* 若场景开头缺少历史帧，则使用最早已知帧的位移差向前线性外推，保持固定长度。
            ego_his_trajs = np.zeros((his_ts+1, 3))
            ego_his_trajs_diff = np.zeros((his_ts+1, 3))
            sample_cur = sample
            for i in range(his_ts, -1, -1):
                if sample_cur is not None:
                    pose_mat = get_global_sensor_pose(sample_cur, nusc, inverse=False)
                    ego_his_trajs[i] = pose_mat[:3, 3]
                    has_prev = sample_cur['prev'] != ''
                    has_next = sample_cur['next'] != ''
                    if has_next:
                        sample_next = nusc.get('sample', sample_cur['next'])
                        pose_mat_next = get_global_sensor_pose(sample_next, nusc, inverse=False)
                        ego_his_trajs_diff[i] = pose_mat_next[:3, 3] - ego_his_trajs[i]
                    sample_cur = nusc.get('sample', sample_cur['prev']) if has_prev else None
                else:
                    ego_his_trajs[i] = ego_his_trajs[i+1] - ego_his_trajs_diff[i+1]
                    ego_his_trajs_diff[i] = ego_his_trajs_diff[i+1]

            #* 将所有历史 global 位置统一变换到“当前帧”的 ego 坐标系：先减当前 ego
            #* 在 global 中的平移，再乘当前 ego2global 旋转的逆矩阵。
            ego_his_trajs = ego_his_trajs - np.array(pose_record['translation'])
            rot_mat = Quaternion(pose_record['rotation']).inverse.rotation_matrix
            ego_his_trajs = np.dot(rot_mat, ego_his_trajs.T).T
            #* 再从当前 ego 坐标系变换到当前 LIDAR_TOP 坐标系。
            ego_his_trajs = ego_his_trajs - np.array(cs_record['translation'])
            rot_mat = Quaternion(cs_record['rotation']).inverse.rotation_matrix
            ego_his_trajs = np.dot(rot_mat, ego_his_trajs.T).T
            #* 相邻位置作差，将 his_ts+1 个绝对位置转成 his_ts 个逐步位移；最终只保存 xy，
            #* 所以 gt_ego_his_trajs 默认形状为 (2, 2)，每行表示一段 (dx, dy)，单位 m。
            ego_his_trajs = ego_his_trajs[1:] - ego_his_trajs[:-1]

            # *10. 生成自车未来轨迹 gt_ego_fut_trajs 及逐步有效掩码 gt_ego_fut_masks。
            #*==================== 自车未来轨迹及掩码 ====================#
            #* 从当前 sample 沿 next 读取未来 fut_ts（默认 6）帧。为计算相邻位移，需要包括
            #* 当前时刻在内的 fut_ts+1 个位置，因此初始数组形状为 (7, 3)。
            ego_fut_trajs = np.zeros((fut_ts+1, 3))
            ego_fut_masks = np.zeros((fut_ts+1))
            sample_cur = sample
            for i in range(fut_ts+1):
                pose_mat = get_global_sensor_pose(sample_cur, nusc, inverse=False)
                ego_fut_trajs[i] = pose_mat[:3, 3]
                ego_fut_masks[i] = 1
                if sample_cur['next'] == '':
                    #* 场景提前结束时，用最后一个有效位置填充剩余位置，之后作差会得到零位移；
                    #* 对应的 mask 保持 0，训练时可忽略这些补齐时间步。
                    ego_fut_trajs[i+1:] = ego_fut_trajs[i]
                    break
                else:
                    sample_cur = nusc.get('sample', sample_cur['next'])
            #* 与历史轨迹相同：把所有未来 global 位置统一表达在当前帧 ego 坐标系中。
            ego_fut_trajs = ego_fut_trajs - np.array(pose_record['translation'])
            rot_mat = Quaternion(pose_record['rotation']).inverse.rotation_matrix
            ego_fut_trajs = np.dot(rot_mat, ego_fut_trajs.T).T
            #* 当前帧 ego -> 当前帧 LIDAR_TOP，得到以当前 LiDAR 为原点的未来位置序列。
            ego_fut_trajs = ego_fut_trajs - np.array(cs_record['translation'])
            rot_mat = Quaternion(cs_record['rotation']).inverse.rotation_matrix
            ego_fut_trajs = np.dot(rot_mat, ego_fut_trajs.T).T

            # *11. 根据 GT 未来终点横向偏移生成三分类驾驶指令 gt_ego_fut_cmd。
            #*==================== 驾驶指令 gt_ego_fut_cmd ====================#
            #* 这是 VAD 从 GT 未来轨迹派生的伪标签，并非 nuScenes 直接提供的导航命令。
            #* 判定发生在逐步差分之前：ego_fut_trajs[-1][0] 是最后一个未来位置相对当前
            #* LiDAR 原点的 x 偏移。按本项目约定，以 +/-2 m 为阈值生成 3 维 one-hot：
            #*   x >=  2 m -> [1, 0, 0]（Turn Right，右转）
            #*   x <= -2 m -> [0, 1, 0]（Turn Left，左转）
            #*   其余       -> [0, 0, 1]（Go Straight，直行）
            if ego_fut_trajs[-1][0] >= 2:
                command = np.array([1, 0, 0])  # Turn Right
            elif ego_fut_trajs[-1][0] <= -2:
                command = np.array([0, 1, 0])  # Turn Left
            else:
                command = np.array([0, 0, 1])  # Go Straight
            #* 将 7 个“相对当前帧的未来位置”转换为 6 个相邻时间步增量；保存 xy 后，
            #* gt_ego_fut_trajs 默认形状为 (6, 2)。若需要相对当前帧的累计轨迹，可沿时间维 cumsum。
            ego_fut_trajs = ego_fut_trajs[1:] - ego_fut_trajs[:-1]

            # *12. 融合 ego pose、CAN bus 和车辆尺寸，生成自车低层状态 gt_ego_lcf_feat。
            #*==================== 自车低层状态 gt_ego_lcf_feat ====================#
            #* VAD 将 ego pose、CAN bus 和固定车辆尺寸组合成 9 维特征：
            #* [vx, vy, ax, ay, yaw_rate, length, width, speed, curvature]。
            #* 其中 vx/vy、yaw_rate 由相邻 0.5 s key frame 的 ego pose 估算；ax/ay 取 can_bus[7:9]；
            #* length/width 是本转换器定义的自车尺寸；speed/curvature 优先取 CAN bus，缺失时回退估算。
            ego_lcf_feat = np.zeros(9)
            #* 优先用上一帧计算速度与 yaw_rate；场景首帧没有 previous 时改用下一帧。
            _, _, ego_yaw = quart_to_rpy(pose_record['rotation'])
            ego_pos = np.array(pose_record['translation'])
            if pose_record_prev is not None:
                _, _, ego_yaw_prev = quart_to_rpy(pose_record_prev['rotation'])
                ego_pos_prev = np.array(pose_record_prev['translation'])
            if pose_record_next is not None:
                _, _, ego_yaw_next = quart_to_rpy(pose_record_next['rotation'])
                ego_pos_next = np.array(pose_record_next['translation'])
            assert (pose_record_prev is not None) or (pose_record_next is not None), 'prev token and next token all empty'
            if pose_record_prev is not None:
                ego_w = (ego_yaw - ego_yaw_prev) / 0.5
                ego_v = np.linalg.norm(ego_pos[:2] - ego_pos_prev[:2]) / 0.5
                ego_vx, ego_vy = ego_v * math.cos(ego_yaw + np.pi/2), ego_v * math.sin(ego_yaw + np.pi/2)
            else:
                ego_w = (ego_yaw_next - ego_yaw) / 0.5
                ego_v = np.linalg.norm(ego_pos_next[:2] - ego_pos[:2]) / 0.5
                ego_vx, ego_vy = ego_v * math.cos(ego_yaw + np.pi/2), ego_v * math.sin(ego_yaw + np.pi/2)

            ref_scene = nusc.get("scene", sample['scene_token'])
            try:
                pose_msgs = nusc_can_bus.get_messages(ref_scene['name'],'pose')
                steer_msgs = nusc_can_bus.get_messages(ref_scene['name'], 'steeranglefeedback')
                pose_uts = [msg['utime'] for msg in pose_msgs]
                steer_uts = [msg['utime'] for msg in steer_msgs]
                ref_utime = sample['timestamp']
                pose_index = locate_message(pose_uts, ref_utime)
                pose_data = pose_msgs[pose_index]
                steer_index = locate_message(steer_uts, ref_utime)
                steer_data = steer_msgs[steer_index]
                #* CAN bus 的纵向速度，单位 m/s。
                v0 = pose_data["vel"][0]  # [0] means longitudinal velocity  m/s
                #* 用转向反馈和轴距 2.588 m 近似曲率；代码约定正值表示左转。
                steering = steer_data["value"]
                #* 新加坡为左侧通行，统一符号约定时需要翻转 steering。
                flip_flag = True if map_location.startswith('singapore') else False
                if flip_flag:
                    steering *= -1
                Kappa = 2 * steering / 2.588
            except:
                #* 当前 scene 没有可用 CAN bus 时，以前后轨迹位移估算速度，并将曲率设为 0。
                delta_x = ego_his_trajs[-1, 0] + ego_fut_trajs[0, 0]
                delta_y = ego_his_trajs[-1, 1] + ego_fut_trajs[0, 1]
                v0 = np.sqrt(delta_x**2 + delta_y**2)
                Kappa = 0

            ego_lcf_feat[:2] = np.array([ego_vx, ego_vy]) #can_bus[13:15]
            ego_lcf_feat[2:4] = can_bus[7:9]
            ego_lcf_feat[4] = ego_w #can_bus[12]
            ego_lcf_feat[5:7] = np.array([ego_length, ego_width])
            ego_lcf_feat[7] = v0
            ego_lcf_feat[8] = Kappa

            # *13. 将目标检测/运动标签和自车规划标签统一写入当前帧 info。
            info['gt_boxes'] = gt_boxes
            info['gt_names'] = names
            info['gt_velocity'] = velocity.reshape(-1, 2)
            info['num_lidar_pts'] = np.array(
                [a['num_lidar_pts'] for a in annotations])
            info['num_radar_pts'] = np.array(
                [a['num_radar_pts'] for a in annotations])
            info['valid_flag'] = valid_flag
            info['gt_agent_fut_trajs'] = gt_fut_trajs.reshape(-1, fut_ts*2).astype(np.float32)
            info['gt_agent_fut_masks'] = gt_fut_masks.reshape(-1, fut_ts).astype(np.float32)
            info['gt_agent_lcf_feat'] = agent_lcf_feat.astype(np.float32)
            info['gt_agent_fut_yaw'] = gt_fut_yaw.astype(np.float32)
            info['gt_agent_fut_goal'] = gt_fut_goal.astype(np.float32)
            #* 以下字段均由 VAD 在转换阶段生成并写入元数据 pkl，而不是 nuScenes 原生字段。
            info['gt_ego_his_trajs'] = ego_his_trajs[:, :2].astype(np.float32)  # (his_ts, 2)，历史逐步位移
            info['gt_ego_fut_trajs'] = ego_fut_trajs[:, :2].astype(np.float32)  # (fut_ts, 2)，未来逐步位移
            info['gt_ego_fut_masks'] = ego_fut_masks[1:].astype(np.float32)  # (fut_ts,)，去除当前时刻 mask
            info['gt_ego_fut_cmd'] = command.astype(np.float32)  # (3,)，[右转, 左转, 直行] one-hot
            info['gt_ego_lcf_feat'] = ego_lcf_feat.astype(np.float32)  # (9,)，自车低层运动/控制特征

        # *14. 按 scene token 将完整 info 放入 train 或 val 列表，防止同一场景跨数据划分。
        if sample['scene_token'] in train_scenes:
            train_nusc_infos.append(info)
        else:
            val_nusc_infos.append(info)

    return train_nusc_infos, val_nusc_infos

def get_global_sensor_pose(rec, nusc, inverse=False):
    # *1. 读取指定 sample 的 LIDAR_TOP 标定和采集时刻 ego pose。
    lidar_sample_data = nusc.get('sample_data', rec['data']['LIDAR_TOP'])

    sd_ep = nusc.get("ego_pose", lidar_sample_data["ego_pose_token"])
    sd_cs = nusc.get("calibrated_sensor", lidar_sample_data["calibrated_sensor_token"])
    # *2. 按需组合 global_from_ego @ ego_from_sensor，或计算其逆向变换。
    if inverse is False:
        global_from_ego = transform_matrix(sd_ep["translation"], Quaternion(sd_ep["rotation"]), inverse=False)
        ego_from_sensor = transform_matrix(sd_cs["translation"], Quaternion(sd_cs["rotation"]), inverse=False)
        pose = global_from_ego.dot(ego_from_sensor)
        # translation equivalent writing
        # pose_translation = np.array(sd_cs["translation"])
        # rot_mat = Quaternion(sd_ep['rotation']).rotation_matrix
        # pose_translation = np.dot(rot_mat, pose_translation)
        # # pose_translation = pose[:3, 3]
        # pose_translation = pose_translation + np.array(sd_ep["translation"])
    else:
        sensor_from_ego = transform_matrix(sd_cs["translation"], Quaternion(sd_cs["rotation"]), inverse=True)
        ego_from_global = transform_matrix(sd_ep["translation"], Quaternion(sd_ep["rotation"]), inverse=True)
        pose = sensor_from_ego.dot(ego_from_global)
    return pose

def obtain_sensor2top(nusc,
                      sensor_token,
                      l2e_t,
                      l2e_r_mat,
                      e2g_t,
                      e2g_r_mat,
                      sensor_type='lidar'):
    """Obtain the info with RT matric from general sensor to Top LiDAR.

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
    # *1. 读取来源传感器帧的文件、标定参数和采集时刻 ego pose。
    sd_rec = nusc.get('sample_data', sensor_token)
    cs_record = nusc.get('calibrated_sensor',
                         sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    data_path = str(nusc.get_sample_data_path(sd_rec['token']))
    if os.getcwd() in data_path:  # path from lyftdataset is absolute path
        data_path = data_path.split(f'{os.getcwd()}/')[-1]  # relative path
    # *2. 整理该相机/历史 LiDAR 帧自身的基础元数据。
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

    # *3. 组合 sensor -> ego -> global -> 当前 ego -> 当前 LIDAR_TOP，得到统一坐标变换。
    # *3.1 点变换约定为 points @ R.T + T，供相机标定和历史点云融合共同使用。
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                  ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    sweep['sensor2lidar_rotation'] = R.T  # points @ R.T + T
    sweep['sensor2lidar_translation'] = T
    return sweep


def export_2d_annotation(root_path, info_path, version, mono3d=False):
    """Export 2d annotation from the info file and raw data.

    Args:
        root_path (str): Root path of the raw data.
        info_path (str): Path of the info file.
        version (str): Dataset version.
        mono3d (bool): Whether to export mono3d annotation. Default: False.
    """
    # *1. 定义六路相机，加载前面生成的 info pkl 和 nuScenes 数据库。
    camera_types = [
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_FRONT_LEFT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_BACK_RIGHT',
    ]
    nusc_infos = mmcv.load(info_path)['infos']
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    # info_2d_list = []
    cat2Ids = [
        dict(id=nus_categories.index(cat_name), name=cat_name)
        for cat_name in nus_categories
    ]
    coco_ann_id = 0
    coco_2d_dict = dict(annotations=[], images=[], categories=cat2Ids)
    # *2. 遍历每帧、每个相机，将可见 3D box 投影到图像并整理为 COCO annotation。
    for info in mmcv.track_iter_progress(nusc_infos):
        for cam in camera_types:
            cam_info = info['cams'][cam]
            coco_infos = get_2d_boxes(
                nusc,
                cam_info['sample_data_token'],
                visibilities=['', '1', '2', '3', '4'],
                mono3d=mono3d)
            (height, width, _) = mmcv.imread(cam_info['data_path']).shape
            coco_2d_dict['images'].append(
                dict(
                    file_name=cam_info['data_path'].split('data/nuscenes/')
                    [-1],
                    id=cam_info['sample_data_token'],
                    token=info['token'],
                    cam2ego_rotation=cam_info['sensor2ego_rotation'],
                    cam2ego_translation=cam_info['sensor2ego_translation'],
                    ego2global_rotation=info['ego2global_rotation'],
                    ego2global_translation=info['ego2global_translation'],
                    cam_intrinsic=cam_info['cam_intrinsic'],
                    width=width,
                    height=height))
            for coco_info in coco_infos:
                if coco_info is None:
                    continue
                # add an empty key for coco format
                coco_info['segmentation'] = []
                coco_info['id'] = coco_ann_id
                coco_2d_dict['annotations'].append(coco_info)
                coco_ann_id += 1
    # *3. 根据是否包含单目 3D 字段确定文件名，并写出 COCO JSON。
    if mono3d:
        json_prefix = f'{info_path[:-4]}_mono3d'
    else:
        json_prefix = f'{info_path[:-4]}'
    mmcv.dump(coco_2d_dict, f'{json_prefix}.coco.json')


def get_2d_boxes(nusc,
                 sample_data_token: str,
                 visibilities: List[str],
                 mono3d=True):
    """Get the 2D annotation records for a given `sample_data_token`.

    Args:
        sample_data_token (str): Sample data token belonging to a camera \
            keyframe.
        visibilities (list[str]): Visibility filter.
        mono3d (bool): Whether to get boxes with mono3d annotation.

    Return:
        list[dict]: List of 2D annotation record that belongs to the input
            `sample_data_token`.
    """

    # *1. 读取相机 sample_data、所属 sample、相机标定和采集时刻 ego pose。
    sd_rec = nusc.get('sample_data', sample_data_token)

    assert sd_rec[
        'sensor_modality'] == 'camera', 'Error: get_2d_boxes only works' \
        ' for camera sample_data!'
    if not sd_rec['is_key_frame']:
        raise ValueError(
            'The 2D re-projections are available only for keyframes.')

    s_rec = nusc.get('sample', sd_rec['sample_token'])

    # Get the calibrated sensor and ego pose
    # record to get the transformation matrices.
    cs_rec = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_rec = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    camera_intrinsic = np.array(cs_rec['camera_intrinsic'])

    # *2. 读取当前 sample 的 3D annotation，并按 visibility_token 过滤。
    ann_recs = [
        nusc.get('sample_annotation', token) for token in s_rec['anns']
    ]
    ann_recs = [
        ann_rec for ann_rec in ann_recs
        if (ann_rec['visibility_token'] in visibilities)
    ]

    repro_recs = []

    # *3. 逐目标执行 global -> ego -> camera 变换，再投影到图像平面并裁剪到画布。
    for ann_rec in ann_recs:
        # Augment sample_annotation with token information.
        ann_rec['sample_annotation_token'] = ann_rec['token']
        ann_rec['sample_data_token'] = sample_data_token

        # Get the box in global coordinates.
        box = nusc.get_box(ann_rec['token'])

        # Move them to the ego-pose frame.
        box.translate(-np.array(pose_rec['translation']))
        box.rotate(Quaternion(pose_rec['rotation']).inverse)

        # Move them to the calibrated sensor frame.
        box.translate(-np.array(cs_rec['translation']))
        box.rotate(Quaternion(cs_rec['rotation']).inverse)

        # Filter out the corners that are not in front of the calibrated
        # sensor.
        corners_3d = box.corners()
        in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
        corners_3d = corners_3d[:, in_front]

        # Project 3d box to 2d.
        corner_coords = view_points(corners_3d, camera_intrinsic,
                                    True).T[:, :2].tolist()

        # Keep only corners that fall within the image.
        final_coords = post_process_coords(corner_coords)

        # Skip if the convex hull of the re-projected corners
        # does not intersect the image canvas.
        if final_coords is None:
            continue
        else:
            min_x, min_y, max_x, max_y = final_coords

        # Generate dictionary record to be included in the .json file.
        repro_rec = generate_record(ann_rec, min_x, min_y, max_x, max_y,
                                    sample_data_token, sd_rec['filename'])

        # *4. mono3d 模式额外写入相机坐标系 3D box、速度、投影中心深度和属性类别。
        if mono3d and (repro_rec is not None):
            loc = box.center.tolist()

            dim = box.wlh
            dim[[0, 1, 2]] = dim[[1, 2, 0]]  # convert wlh to our lhw
            dim = dim.tolist()

            rot = box.orientation.yaw_pitch_roll[0]
            rot = [-rot]  # convert the rot to our cam coordinate

            global_velo2d = nusc.box_velocity(box.token)[:2]
            global_velo3d = np.array([*global_velo2d, 0.0])
            e2g_r_mat = Quaternion(pose_rec['rotation']).rotation_matrix
            c2e_r_mat = Quaternion(cs_rec['rotation']).rotation_matrix
            cam_velo3d = global_velo3d @ np.linalg.inv(
                e2g_r_mat).T @ np.linalg.inv(c2e_r_mat).T
            velo = cam_velo3d[0::2].tolist()

            repro_rec['bbox_cam3d'] = loc + dim + rot
            repro_rec['velo_cam3d'] = velo

            center3d = np.array(loc).reshape([1, 3])
            center2d = points_cam2img(
                center3d, camera_intrinsic, with_depth=True)
            repro_rec['center2d'] = center2d.squeeze().tolist()
            # normalized center2D + depth
            # if samples with depth < 0 will be removed
            if repro_rec['center2d'][2] <= 0:
                continue

            ann_token = nusc.get('sample_annotation',
                                 box.token)['attribute_tokens']
            if len(ann_token) == 0:
                attr_name = 'None'
            else:
                attr_name = nusc.get('attribute', ann_token[0])['name']
            attr_id = nus_attributes.index(attr_name)
            repro_rec['attribute_name'] = attr_name
            repro_rec['attribute_id'] = attr_id

        repro_recs.append(repro_rec)

    return repro_recs


def post_process_coords(
    corner_coords: List, imsize: Tuple[int, int] = (1600, 900)
) -> Union[Tuple[float, float, float, float], None]:
    """Get the intersection of the convex hull of the reprojected bbox corners
    and the image canvas, return None if no intersection.

    Args:
        corner_coords (list[int]): Corner coordinates of reprojected
            bounding box.
        imsize (tuple[int]): Size of the image canvas.

    Return:
        tuple [float]: Intersection of the convex hull of the 2D box
            corners and the image canvas.
    """
    # *1. 由投影角点生成凸包，并与图像画布求交，去除画面外区域。
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array(
            [coord for coord in img_intersection.exterior.coords])

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        # *2. 用交集区域的轴对齐外接矩形作为最终 2D bbox。
        return min_x, min_y, max_x, max_y
    else:
        return None


def generate_record(ann_rec: dict, x1: float, y1: float, x2: float, y2: float,
                    sample_data_token: str, filename: str) -> OrderedDict:
    """Generate one 2D annotation record given various informations on top of
    the 2D bounding box coordinates.

    Args:
        ann_rec (dict): Original 3d annotation record.
        x1 (float): Minimum value of the x coordinate.
        y1 (float): Minimum value of the y coordinate.
        x2 (float): Maximum value of the x coordinate.
        y2 (float): Maximum value of the y coordinate.
        sample_data_token (str): Sample data token.
        filename (str):The corresponding image file where the annotation
            is present.

    Returns:
        dict: A sample 2D annotation record.
            - file_name (str): flie name
            - image_id (str): sample data token
            - area (float): 2d box area
            - category_name (str): category name
            - category_id (int): category id
            - bbox (list[float]): left x, top y, dx, dy of 2d box
            - iscrowd (int): whether the area is crowd
    """
    # *1. 保留后续可能用到的 nuScenes annotation 标识、可见性及点数等字段。
    repro_rec = OrderedDict()
    repro_rec['sample_data_token'] = sample_data_token
    coco_rec = dict()

    relevant_keys = [
        'attribute_tokens',
        'category_name',
        'instance_token',
        'next',
        'num_lidar_pts',
        'num_radar_pts',
        'prev',
        'sample_annotation_token',
        'sample_data_token',
        'visibility_token',
    ]

    for key, value in ann_rec.items():
        if key in relevant_keys:
            repro_rec[key] = value

    repro_rec['bbox_corners'] = [x1, y1, x2, y2]
    repro_rec['filename'] = filename

    coco_rec['file_name'] = filename
    coco_rec['image_id'] = sample_data_token
    coco_rec['area'] = (y2 - y1) * (x2 - x1)

    # *2. 将 nuScenes 细粒度类别映射到检测类别，并生成 COCO bbox/area/category 字段。
    if repro_rec['category_name'] not in NuScenesDataset.NameMapping:
        return None
    cat_name = NuScenesDataset.NameMapping[repro_rec['category_name']]
    coco_rec['category_name'] = cat_name
    coco_rec['category_id'] = nus_categories.index(cat_name)
    coco_rec['bbox'] = [x1, y1, x2 - x1, y2 - y1]
    coco_rec['iscrowd'] = 0

    return coco_rec


def nuscenes_data_prep(root_path,
                       can_bus_root_path,
                       info_prefix,
                       version,
                       dataset_name,
                       out_dir,
                       max_sweeps=10):
    """Prepare data related to nuScenes dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        dataset_name (str): The dataset class name.
        out_dir (str): Output directory of the groundtruth database info.
        max_sweeps (int): Number of input consecutive frames. Default: 10
    """
    # *1. 当前 VAD 数据准备入口只生成 temporal info pkl；2D annotation/GT database 需另行调用。
    create_nuscenes_infos(
        root_path, out_dir, can_bus_root_path, info_prefix, version=version, max_sweeps=max_sweeps)


parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument('dataset', metavar='kitti', help='name of the dataset')
parser.add_argument(
    '--root-path',
    type=str,
    default='./data/kitti',
    help='specify the root path of dataset')
parser.add_argument(
    '--canbus',
    type=str,
    default='./data',
    help='specify the root path of nuScenes canbus')
parser.add_argument(
    '--version',
    type=str,
    default='v1.0',
    required=False,
    help='specify the dataset version, no need for kitti')
parser.add_argument(
    '--max-sweeps',
    type=int,
    default=10,
    required=False,
    help='specify sweeps of lidar per example')
parser.add_argument(
    '--out-dir',
    type=str,
    default='./data/kitti',
    required='False',
    help='name of info pkl')
parser.add_argument('--extra-tag', type=str, default='kitti')
parser.add_argument(
    '--workers', type=int, default=4, help='number of threads to be used')
args = parser.parse_args()

if __name__ == '__main__':
    # *1. 完整版 nuScenes：依次为 trainval 与 test 生成独立 temporal pkl。
    if args.dataset == 'nuscenes' and args.version != 'v1.0-mini':
        train_version = f'{args.version}-trainval'
        nuscenes_data_prep(
            root_path=args.root_path,
            can_bus_root_path=args.canbus,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)
        test_version = f'{args.version}-test'
        nuscenes_data_prep(
            root_path=args.root_path,
            can_bus_root_path=args.canbus,
            info_prefix=args.extra_tag,
            version=test_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)
    # *2. mini 版 nuScenes：一次调用同时按官方 mini_train/mini_val 划分并写出两个 pkl。
    elif args.dataset == 'nuscenes' and args.version == 'v1.0-mini':
        train_version = f'{args.version}'
        nuscenes_data_prep(
            root_path=args.root_path,
            can_bus_root_path=args.canbus,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)
