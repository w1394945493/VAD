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
    #*-----------------------------------------------------#
    #* 1. 定义类别、单样本解包与绘图工具
    # result 为模型返回的样本列表；data 为 batch_size=1 的原输入；split 区分 train/val。
    # score_thr/map_thr 只控制画图筛选，不改变预测结果或评测阈值。
    class_names = [
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
        'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']

    def unpack(value):
        # 逐层去掉 DataContainer 和单元素列表包装，保留 Tensor 本身的维度。
        while True:
            if isinstance(value, DataContainer):
                value = value.data
            elif isinstance(value, (list, tuple)) and len(value) == 1:
                value = value[0]
            else:
                return value

    def array(value):
        # 转为 NumPy 供绘图读取；可能与 CPU Tensor 共享内存，不应原地修改返回数组。
        value = unpack(value)
        if hasattr(value, 'detach'):
            value = value.detach().cpu()
        return value.numpy() if hasattr(value, 'numpy') else np.asarray(value)

    def draw_traj(ax, points, cmap_name, width, alpha=1.0):
        # 将相邻轨迹点组成线段，按时间顺序着渐变色；不足两点时无需画线。
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
        bottom = corners[[0, 3, 7, 4, 0], :2]  # 四个底面顶点投影到 BEV，重复首点闭合。
        ax.plot(bottom[:, 0], bottom[:, 1], color=color,
                linewidth=width, zorder=3)

    #*-----------------------------------------------------#
    #* 2. 提取未经运动评测阈值过滤的预测结果
    # N/Nmap 为目标/地图实例数，M 为目标轨迹模态数，P 为地图采样点数。
    # 当前调试代码按未来 T=6 步编写，每步 0.5s；目标通常 M=6，自车有 3 个指令分支。
    meta = unpack(data['img_metas'])  # 当前帧的 token、帧号和六路图像路径。
    pred = result[0]['pts_bbox']
    pred_boxes_obj = pred['boxes_3d']
    boxes = array(pred_boxes_obj.tensor)  # [N,9]：中心、尺寸、框编码 yaw、vx、vy。
    pred_corners = (array(pred_boxes_obj.corners) if len(pred_boxes_obj)
                    else np.empty((0, 8, 3), dtype=np.float32))
    scores = array(pred['scores_3d'])  # [N]：检测置信度。
    labels = array(pred['labels_3d']).astype(np.int64)  # [N]：类别 ID。
    trajs = array(pred['trajs_3d'])  # [N,M,12]：各模态未来 6 步的 (dx,dy) 增量。
    map_scores = array(pred['map_scores_3d'])  # [Nmap]：地图实例置信度。
    map_labels = array(pred['map_labels_3d']).astype(np.int64)  # [Nmap]：地图类别 ID。
    map_pts = array(pred['map_pts_3d'])  # [Nmap,P,2]：已在 BEV 坐标系的折线点。
    ego_preds = np.squeeze(array(pred['ego_fut_preds']))  # 通常 [3,6,2]：每个指令一条增量轨迹。
    ego_cmd = np.squeeze(array(pred['ego_fut_cmd'])).reshape(-1)  # [3]：右转/左转/直行 one-hot。

    #*-----------------------------------------------------#
    #* 3. 提取 GT 目标、未来轨迹、有效掩码与地图实例
    # GT 框应保持原始框编码；规划评测必须在副本上转换 yaw，避免污染这里的角点。
    gt_boxes_obj = unpack(data['gt_bboxes_3d'])
    gt_boxes = array(gt_boxes_obj.tensor)
    gt_corners = (array(gt_boxes_obj.corners) if len(gt_boxes_obj)
                  else np.empty((0, 8, 3), dtype=np.float32))
    gt_labels = array(data['gt_labels_3d']).astype(np.int64)
    gt_attr = array(data['gt_attr_labels'])
    gt_offsets = gt_attr[:, :12].reshape(-1, 6, 2)  # [Ng,6,2]：周围目标未来逐步位移。
    gt_masks = gt_attr[:, 12:18]  # [Ng,6]：未来步是否有有效标注。
    ego_gt = np.squeeze(array(data['ego_fut_trajs'])).reshape(-1, 2)  # [6,2]：自车 GT 增量轨迹。
    fut_valid = bool(np.squeeze(array(data['fut_valid_flag'])))  # 是否有完整未来标注供规划评测。

    gt_map_obj = unpack(data.get('map_gt_bboxes_3d'))
    gt_map_label_obj = unpack(data.get('map_gt_labels_3d'))
    if gt_map_obj is not None and hasattr(gt_map_obj, 'fixed_num_sampled_points'):
        gt_map_pts = array(gt_map_obj.fixed_num_sampled_points)
        gt_map_labels = array(gt_map_label_obj).astype(np.int64)
    else:
        # 未提供地图 GT 时保留空数组，仍可绘制其余内容。
        gt_map_pts = np.empty((0, 0, 2), dtype=np.float32)
        gt_map_labels = np.empty((0,), dtype=np.int64)

    #*-----------------------------------------------------#
    #* 4. 创建输出目录：根目录/train或val/scene_token/frame_xxx.png
    scene_token = str(meta['scene_token'])
    frame_idx = int(meta['frame_idx'])
    sample_idx = str(meta['sample_idx'])
    scene_dir = osp.join(vis_root, split, scene_token)
    os.makedirs(scene_dir, exist_ok=True)

    #*-----------------------------------------------------#
    #* 5. 创建紧凑画布：左侧 2×3 环视图，中间预测 BEV，右侧 GT BEV
    fig = plt.figure(figsize=(16, 4.8))
    grid = fig.add_gridspec(1, 3, width_ratios=[5.4, 1, 1], wspace=0.04)
    camera_grid = grid[0].subgridspec(2, 3, wspace=0.01, hspace=0.01)
    camera_axes = [fig.add_subplot(camera_grid[r, c])
                   for r in range(2) for c in range(3)]
    pred_ax = fig.add_subplot(grid[1])
    gt_ax = fig.add_subplot(grid[2], sharex=pred_ax, sharey=pred_ax)

    # 原相机顺序 FRONT/FRONT_RIGHT/FRONT_LEFT/BACK/BACK_LEFT/BACK_RIGHT，按左右布局重排。
    # 从 filename 读取原图，相机名叠加在图内，不展示归一化或 padding 后的网络输入。
    camera_names = [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    for ax, name, index in zip(camera_axes, camera_names, [2, 0, 1, 4, 3, 5]):
        ax.imshow(plt.imread(meta['filename'][index]))
        ax.text(0.01, 0.97, name, transform=ax.transAxes, va='top',
                color='white', fontsize=7,
                bbox=dict(facecolor='black', alpha=0.45, edgecolor='none'))
        ax.axis('off')

    #*-----------------------------------------------------#
    #* 6. 绘制预测地图、目标框/类别/分数及全部目标轨迹模态
    # 地图按类别使用暖色，与 GT 的蓝灰色区分；只绘制超过各自阈值的实例。
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
            # 增量累加后加当前目标中心，再补上起点；每个目标画全部模态，故可见多条轨迹。
            points = np.cumsum(offsets, axis=0) + box[:2]
            draw_traj(pred_ax, np.vstack([box[:2], points]), 'autumn', 1, 0.65)

    #*-----------------------------------------------------#
    #* 7. 绘制 GT 地图、目标框/类别及有效未来轨迹
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
        valid_steps = int(mask.sum())  # 沿用原实现：假定有效未来标注为连续前缀。
        if valid_steps:
            points = np.cumsum(offsets[:valid_steps], axis=0) + box[:2]
            draw_traj(gt_ax, np.vstack([box[:2], points]), 'summer', 1.6)

    #*-----------------------------------------------------#
    #* 8. 绘制自车示意轮廓、指令对应的预测轨迹和 GT 轨迹
    # 绿色框为沿用原可视化的 1.8×4m 示意框，不是碰撞评测使用的精确 footprint。
    ego_box = np.array([[-0.9, -2], [-0.9, 2], [0.9, 2],
                        [0.9, -2], [-0.9, -2]])
    for ax in [pred_ax, gt_ax]:
        ax.plot(ego_box[:, 0], ego_box[:, 1], color='mediumseagreen', linewidth=1.2)
        ax.plot([0, 0], [0, 2], color='mediumseagreen', linewidth=1.2)
        ax.set(xlim=(-15, 15), ylim=(-30, 30), xlabel='x / m', ylabel='y / m')
        ax.set_aspect('equal')
        ax.grid(color='lightgray', linewidth=0.4, alpha=0.5)

    # 3 个指令各预测 1 条轨迹，只画当前指令对应分支；6 步增量累加并补原点，共 7 个点。
    # 自车预测用 plasma 紫红渐变，GT 用 winter 蓝绿渐变；两侧使用相同米制范围和等比例坐标。
    cmd_idx = int(np.argmax(ego_cmd))
    ego_pred = ego_preds if ego_preds.ndim == 2 else ego_preds[cmd_idx]
    draw_traj(pred_ax, np.vstack([np.zeros(2), np.cumsum(ego_pred, axis=0)]),
              'plasma', 2.2)
    draw_traj(gt_ax, np.vstack([np.zeros(2), np.cumsum(ego_gt, axis=0)]),
              'winter', 2.2)
    #*-----------------------------------------------------#
    #* 9. 添加实例数量、未来有效性、帧标识，并保存组合图
    # 预测数量为阈值筛选后/筛选前；关闭画布释放资源，避免逐帧绘图累积内存。
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

    #*----------------- 调试输出指标说明 -----------------#
    # Motion Prediction：周围目标运动预测，car 合并 car/truck/bus/trailer，pedestrian 为行人。
    # EPA=(hit-0.5*fp)/gt：综合检测与轨迹命中的指标；范围 (-∞,1]，越大越好，理想值 1，误检多时可为负。
    #   gt 为未来完整的 GT 数；hit 为已匹配且 minFDE<=2m 的目标数；fp 为未匹配预测数。
    # ADE：每个匹配目标先取各模态的平均位移误差最小值，再跨目标平均；单位 m，范围 [0,+∞)，越小越好。
    # FDE：每个匹配目标取各模态的终点误差最小值，再跨目标平均；单位 m，范围 [0,+∞)，越小越好。
    #   ADE 可用部分有效未来步；FDE 仅用未来完整的匹配目标；两者选中的最佳模态可以不同。
    # MR：未来完整的匹配目标中 minFDE>2m 的比例；范围 [0,1]，越小越好，不包含未匹配的漏检目标。

    # Planning：自车规划；标题中的 valid 为未来标注完整的帧数，范围 [0,实际推理帧数]，是统计量而非质量指标。
    # plan_L2_1s/2s/3s：前 1/2/3 秒内预测自车轨迹与 GT 的平均位置误差；单位 m，范围 [0,+∞)，越小越好。
    # plan_obj_col_1s/2s/3s：预测轨迹点落入周围目标 GT 未来占用区域的比例；范围 [0,1]，越小越好。
    # plan_obj_box_col_1s/2s/3s：考虑自车矩形 footprint 的碰撞比例；范围 [0,1]，越小越好，对应论文车身碰撞指标。
    #   两类碰撞均对前 2/4/6 个时间步平均，再对有效帧平均；不是仅检查终点，也不是整段是否曾碰撞。
    #   GT 自车框也碰撞的时间步会屏蔽预测碰撞计数；因此 0 只表示当前评测规则下没有统计到碰撞。
    #   MR/碰撞输出为比例：0.01=1%；L2/ADE/FDE 输出为米。只有 EPA 越大越好，其余质量指标越小越好。
    # 分母为 0 时输出 nan（无可评测样本）；这里只统计前 N 帧，不能直接代表整个数据集表现。
    #*---------------------------------------------------#
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
    VIS_ROOT = 'out/vad_pred_vis'
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
