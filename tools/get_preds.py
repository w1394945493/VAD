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

from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model


# 保证从任意目录执行脚本时都能导入 projects.mmdet3d_plugin。
sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

from projects.mmdet3d_plugin.datasets.builder import build_dataloader


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

    # 调试阶段只跑前 N 帧；保持数据顺序，以正确复用同一场景的历史 BEV。
    N = 10

    train_metrics = []
    model.module.prev_frame_info['scene_token'] = None
    model.module.prev_frame_info['prev_bev'] = None
    for i, data in enumerate(train_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
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
        val_metrics.append(result[0]['metric_results'])
        print(f'\rval: {i + 1}/{min(N, len(val_loader))}', end='', flush=True)
        if i + 1 >= N:
            break
    print_metrics('Val', val_metrics)


if __name__ == '__main__':
    main()
