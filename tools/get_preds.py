"""初始化 VAD 推理所需的数据和模型，后续用于保存感知、预测结果。"""
import os
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

warnings.filterwarnings('ignore', message=r'On January 1, 2023, MMCV will release v2\.0\.0.*')
warnings.filterwarnings('ignore', message=r'The arguments `.*` in BaseTransformerLayer has been deprecated.*')

import argparse
import copy
import importlib
import os.path as osp
import sys
import torch
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.collections import LineCollection

from mmcv import Config, DictAction
from mmcv.parallel import DataContainer, MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model


# 保证从任意目录执行脚本时都能导入 projects.mmdet3d_plugin。
sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

from projects.mmdet3d_plugin.datasets.builder import build_dataloader


def visualize_result(result, data, split, vis_root, score_thr, map_thr):
    """绘制六路图像、预测 BEV 和 GT BEV，仅用于快速调试。"""
    class_names = [
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
        'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']

    def unpack(value):
        while True:
            if isinstance(value, DataContainer):
                value = value.data
            elif isinstance(value, (list, tuple)) and len(value) == 1:
                value = value[0]
            else:
                return value

    def array(value):
        value = unpack(value)
        if hasattr(value, 'detach'):
            value = value.detach().cpu()
        return value.numpy() if hasattr(value, 'numpy') else np.asarray(value)

    def draw_traj(ax, points, cmap_name, width, alpha=1.0):
        points = np.asarray(points)
        if len(points) < 2:
            return
        segments = np.stack([points[:-1], points[1:]], axis=1)
        colors = cm.get_cmap(cmap_name)(np.linspace(0.15, 0.95, len(segments)))
        colors[:, 3] *= alpha
        ax.add_collection(LineCollection(
            segments, colors=colors, linewidths=width, zorder=5))

    def draw_box(ax, corners, color, width=1.2):
        # 直接使用 mmdet3d 计算的底面角点，避免手算时混淆 w/l 和 yaw 约定。
        bottom = corners[[0, 3, 7, 4, 0], :2]
        ax.plot(bottom[:, 0], bottom[:, 1], color=color,
                linewidth=width, zorder=3)

    meta = unpack(data['img_metas'])
    pred = result[0]['pts_bbox']
    pred_boxes_obj = pred['boxes_3d']
    boxes = array(pred_boxes_obj.tensor)
    pred_corners = (array(pred_boxes_obj.corners) if len(pred_boxes_obj)
                    else np.empty((0, 8, 3), dtype=np.float32))
    scores = array(pred['scores_3d'])
    labels = array(pred['labels_3d']).astype(np.int64)
    trajs = array(pred['trajs_3d'])
    map_scores = array(pred['map_scores_3d'])
    map_labels = array(pred['map_labels_3d']).astype(np.int64)
    map_pts = array(pred['map_pts_3d'])
    ego_preds = np.squeeze(array(pred['ego_fut_preds']))
    ego_cmd = np.squeeze(array(pred['ego_fut_cmd'])).reshape(-1)

    gt_boxes_obj = unpack(data['gt_bboxes_3d'])
    gt_boxes = array(gt_boxes_obj.tensor)
    gt_corners = (array(gt_boxes_obj.corners) if len(gt_boxes_obj)
                  else np.empty((0, 8, 3), dtype=np.float32))
    gt_labels = array(data['gt_labels_3d']).astype(np.int64)
    gt_attr = array(data['gt_attr_labels'])
    gt_offsets = gt_attr[:, :12].reshape(-1, 6, 2)
    gt_masks = gt_attr[:, 12:18]
    ego_gt = np.squeeze(array(data['ego_fut_trajs'])).reshape(-1, 2)
    fut_valid = bool(np.squeeze(array(data['fut_valid_flag'])))

    gt_map_obj = unpack(data.get('map_gt_bboxes_3d'))
    gt_map_label_obj = unpack(data.get('map_gt_labels_3d'))
    if gt_map_obj is not None and hasattr(gt_map_obj, 'fixed_num_sampled_points'):
        gt_map_pts = array(gt_map_obj.fixed_num_sampled_points)
        gt_map_labels = array(gt_map_label_obj).astype(np.int64)
    else:
        gt_map_pts = np.empty((0, 0, 2), dtype=np.float32)
        gt_map_labels = np.empty((0,), dtype=np.int64)

    scene_token = str(meta['scene_token'])
    frame_idx = int(meta['frame_idx'])
    sample_idx = str(meta['sample_idx'])
    scene_dir = osp.join(vis_root, split, scene_token)
    os.makedirs(scene_dir, exist_ok=True)

    fig = plt.figure(figsize=(16, 4.8))
    grid = fig.add_gridspec(1, 3, width_ratios=[5.4, 1, 1], wspace=0.04)
    camera_grid = grid[0].subgridspec(2, 3, wspace=0.01, hspace=0.01)
    camera_axes = [fig.add_subplot(camera_grid[r, c])
                   for r in range(2) for c in range(3)]
    pred_ax = fig.add_subplot(grid[1])
    gt_ax = fig.add_subplot(grid[2], sharex=pred_ax, sharey=pred_ax)

    camera_names = [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    for ax, name, index in zip(camera_axes, camera_names, [2, 0, 1, 4, 3, 5]):
        ax.imshow(plt.imread(meta['filename'][index]))
        ax.text(0.01, 0.97, name, transform=ax.transAxes, va='top',
                color='white', fontsize=7,
                bbox=dict(facecolor='black', alpha=0.45, edgecolor='none'))
        ax.axis('off')

    pred_map_colors = ['darkorange', 'goldenrod', 'tomato']
    for pts, score, label in zip(map_pts, map_scores, map_labels):
        if score >= map_thr:
            pts = np.asarray(pts).reshape(-1, 2)
            color = pred_map_colors[int(label) % 3]
            pred_ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=1)
            pred_ax.scatter(pts[:, 0], pts[:, 1], color=color, s=2)

    for box, corners, score, label, modes in zip(
            boxes, pred_corners, scores, labels, trajs):
        if score < score_thr:
            continue
        draw_box(pred_ax, corners, 'tomato')
        name = class_names[int(label)] if 0 <= int(label) < len(class_names) else str(label)
        pred_ax.text(box[0], box[1], f'{name} {score:.2f}',
                     color='darkred', fontsize=6)
        for offsets in np.asarray(modes).reshape(-1, 6, 2):
            points = np.cumsum(offsets, axis=0) + box[:2]
            draw_traj(pred_ax, np.vstack([box[:2], points]), 'autumn', 1, 0.65)

    gt_map_colors = ['cornflowerblue', 'royalblue', 'slategrey']
    for pts, label in zip(gt_map_pts, gt_map_labels):
        pts = np.asarray(pts).reshape(-1, 2)
        color = gt_map_colors[int(label) % 3]
        gt_ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=1)
        gt_ax.scatter(pts[:, 0], pts[:, 1], color=color, s=2)

    for box, corners, label, offsets, mask in zip(
            gt_boxes, gt_corners, gt_labels, gt_offsets, gt_masks):
        draw_box(gt_ax, corners, 'dodgerblue', 1.4)
        name = class_names[int(label)] if 0 <= int(label) < len(class_names) else str(label)
        gt_ax.text(box[0], box[1], name, color='navy', fontsize=6)
        valid_steps = int(mask.sum())
        if valid_steps:
            points = np.cumsum(offsets[:valid_steps], axis=0) + box[:2]
            draw_traj(gt_ax, np.vstack([box[:2], points]), 'summer', 1.6)

    ego_box = np.array([[-0.9, -2], [-0.9, 2], [0.9, 2],
                        [0.9, -2], [-0.9, -2]])
    for ax in [pred_ax, gt_ax]:
        ax.plot(ego_box[:, 0], ego_box[:, 1], color='mediumseagreen', linewidth=1.2)
        ax.plot([0, 0], [0, 2], color='mediumseagreen', linewidth=1.2)
        ax.set(xlim=(-15, 15), ylim=(-30, 30), xlabel='x / m', ylabel='y / m')
        ax.set_aspect('equal')
        ax.grid(color='lightgray', linewidth=0.4, alpha=0.5)

    cmd_idx = int(np.argmax(ego_cmd))
    ego_pred = ego_preds if ego_preds.ndim == 2 else ego_preds[cmd_idx]
    draw_traj(pred_ax, np.vstack([np.zeros(2), np.cumsum(ego_pred, axis=0)]),
              'plasma', 2.2)
    draw_traj(gt_ax, np.vstack([np.zeros(2), np.cumsum(ego_gt, axis=0)]),
              'winter', 2.2)
    pred_ax.set_title(
        f'Prediction\nobjects {int((scores >= score_thr).sum())}/{len(scores)}, '
        f'maps {int((map_scores >= map_thr).sum())}/{len(map_scores)}', fontsize=8)
    gt_ax.set_title(
        f'Ground truth\nobjects {len(gt_boxes)}, maps {len(gt_map_pts)}, '
        f'ego valid {fut_valid}', fontsize=8)
    fig.suptitle(f'Sample Token: {sample_idx}    |    Frame Number: {frame_idx}',
                 fontsize=8)
    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.04, top=0.88)
    fig.savefig(osp.join(scene_dir, f'frame_{frame_idx:03d}.png'),
                bbox_inches='tight', dpi=100)
    plt.close(fig)


def print_metrics(split, metrics):
    """按 VAD 的数据集评测公式汇总前 N 帧指标。"""
    def value(x):
        return float(x.item()) if hasattr(x, 'item') else float(x)

    total = {}
    for metric in metrics:
        for key, val in metric.items():
            total[key] = total.get(key, 0.0) + value(val)

    def divide(a, b):
        return a / b if b else float('nan')

    print(f'\n-------------- {split} Motion Prediction --------------')
    for cls in ['car', 'pedestrian']:
        print(
            f'{cls}: '
            f'EPA={divide(total["hit_"+cls] - 0.5 * total["fp_"+cls], total["gt_"+cls]):.4f}, '
            f'ADE={divide(total["ADE_"+cls], total["cnt_ade_"+cls]):.4f}, '
            f'FDE={divide(total["FDE_"+cls], total["cnt_fde_"+cls]):.4f}, '
            f'MR={divide(total["MR_"+cls], total["cnt_fde_"+cls]):.4f}')

    valid_num = sum(value(metric['fut_valid_flag']) for metric in metrics)
    print(f'-------------- {split} Planning ({int(valid_num)} valid) --------------')
    for key in metrics[0]:
        if key.startswith('plan_'):
            print(f'{key}: {divide(total[key], valid_num):.4f}')


def parse_args():
    parser = argparse.ArgumentParser(description='初始化 VAD 预测结果提取流程')
    parser.add_argument('config', help='配置文件路径')
    parser.add_argument('checkpoint', help='模型权重路径')
    parser.add_argument(
        '--cfg-options', nargs='+', action=DictAction,
        help='覆盖配置项，例如 data.workers_per_gpu=2')
    return parser.parse_args()


def main():
    args = parse_args()
    # 可视化已移到本脚本，避免模型内部重复绘图。
    # os.environ.pop('VAD_DEBUG_VIS_DIR', None)
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    # 导入配置中注册的 VAD 数据集、pipeline 和模型模块。
    if cfg.get('custom_imports'):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg.custom_imports)
    if cfg.get('plugin', False):
        plugin_dir = cfg.get('plugin_dir', osp.dirname(args.config))
        importlib.import_module(plugin_dir.rstrip('/').replace('/', '.'))

    # 训练集也使用验证 pipeline，以逐帧、无随机增强的方式生成预测结果。
    train_cfg = copy.deepcopy(cfg.data.val)
    train_cfg.ann_file = cfg.data.train.ann_file
    train_cfg.test_mode = True
    val_cfg = copy.deepcopy(cfg.data.val)
    val_cfg.test_mode = True
    # samples_per_gpu 属于 dataloader，不能传给 Dataset 构造函数。
    train_cfg.pop('samples_per_gpu', None)
    val_cfg.pop('samples_per_gpu', None)

    train_dataset = build_dataset(train_cfg)
    val_dataset = build_dataset(val_cfg)
    loader_kwargs = dict(
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        num_gpus=1,
        dist=False,
        shuffle=False,
    )
    train_loader = build_dataloader(train_dataset, **loader_kwargs)
    val_loader = build_dataloader(val_dataset, **loader_kwargs)

    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.CLASSES = checkpoint.get('meta', {}).get('CLASSES', val_dataset.CLASSES)
    model = MMDataParallel(model.cuda(), device_ids=[0])
    model.eval()

    print(f'训练集：{len(train_dataset)} 帧，{len(train_loader)} 个 batch')
    print(f'验证集：{len(val_dataset)} 帧，{len(val_loader)} 个 batch')
    print('模型与权重加载完成。')

    # 调试参数直接固定在脚本中。
    N = 10
    SCORE_THR = 0.3
    MAP_THR = 0.3
    VIS_ROOT = 'out/pred_vis'
    os.makedirs(osp.join(VIS_ROOT, 'train'), exist_ok=True)
    os.makedirs(osp.join(VIS_ROOT, 'val'), exist_ok=True)

    train_metrics = []
    model.module.prev_frame_info['scene_token'] = None
    model.module.prev_frame_info['prev_bev'] = None
    for i, data in enumerate(train_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        visualize_result(result, data, 'train', VIS_ROOT, SCORE_THR, MAP_THR)
        train_metrics.append(result[0]['metric_results'])
        print(f'\rtrain: {i + 1}/{min(N, len(train_loader))}', end='', flush=True)
        if i + 1 >= N:
            break
    print_metrics('Train', train_metrics)

    val_metrics = []
    model.module.prev_frame_info['scene_token'] = None
    model.module.prev_frame_info['prev_bev'] = None
    for i, data in enumerate(val_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        visualize_result(result, data, 'val', VIS_ROOT, SCORE_THR, MAP_THR)
        val_metrics.append(result[0]['metric_results'])
        print(f'\rval: {i + 1}/{min(N, len(val_loader))}', end='', flush=True)
        if i + 1 >= N:
            break
    print_metrics('Val', val_metrics)
    print(f'\n调试可视化已保存到：{VIS_ROOT}')


if __name__ == '__main__':
    main()
