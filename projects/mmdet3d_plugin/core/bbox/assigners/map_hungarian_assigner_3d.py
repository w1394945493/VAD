import torch
import torch.nn.functional as F

from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.assigners import AssignResult
from mmdet.core.bbox.assigners import BaseAssigner
from mmdet.core.bbox.match_costs import build_match_cost
from mmdet.models.utils.transformer import inverse_sigmoid
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
from projects.mmdet3d_plugin.VAD.utils.map_utils import (
    normalize_2d_bbox, normalize_2d_pts, denormalize_2d_bbox
)

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None

@BBOX_ASSIGNERS.register_module()
class MapHungarianAssigner3D(BaseAssigner):
    """Computes one-to-one matching between predictions and ground truth.
    This class computes an assignment between the targets and the predictions
    based on the costs. The costs are a weighted sum of four components:
    classification, bbox L1, bbox IoU and ordered-point costs. The
    targets don't include the no_object, so generally there are more
    predictions than targets. After the one-to-one matching, the un-matched
    are treated as backgrounds. Thus each query prediction will be assigned
    with `0` or a positive integer indicating the ground truth index:
    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt
    Args:
        cls_weight (int | float, optional): The scale factor for classification
            cost. Default 1.0.
        bbox_weight (int | float, optional): The scale factor for regression
            L1 cost. Default 1.0.
        iou_weight (int | float, optional): The scale factor for regression
            iou cost. Default 1.0.
        iou_calculator (dict | optional): The config for the iou calculation.
            Default type `BboxOverlaps2D`.
        iou_mode (str | optional): "iou" (intersection over union), "iof"
                (intersection over foreground), or "giou" (generalized
                intersection over union). Default "giou".
    """

    def __init__(self,
                 cls_cost=dict(type='ClassificationCost', weight=1.),
                 reg_cost=dict(type='BBoxL1Cost', weight=1.0),
                 iou_cost=dict(type='IoUCost', weight=0.0),
                 pts_cost=dict(type='ChamferDistance',loss_src_weight=1.0,loss_dst_weight=1.0),
                 pc_range=None):
        self.cls_cost = build_match_cost(cls_cost)
        self.reg_cost = build_match_cost(reg_cost)
        self.iou_cost = build_match_cost(iou_cost)
        self.pts_cost = build_match_cost(pts_cost)
        self.pc_range = pc_range

    def assign(self,
               bbox_pred,
               cls_pred,
               pts_pred,
               gt_bboxes,
               gt_labels,
               gt_pts,
               gt_bboxes_ignore=None,
               eps=1e-7):
        """Computes one-to-one matching based on the weighted costs.
        This method assign each query prediction to a ground truth or
        background. The `assigned_gt_inds` with -1 means don't care,
        0 means negative sample, and positive number is the index (1-based)
        of assigned gt.
        The assignment is done in the following steps, the order matters.
        1. assign every prediction to -1
        2. compute the weighted costs
        3. do Hungarian matching on CPU based on the costs
        4. assign all to 0 (background) first, then for each matched pair
           between predictions and gts, treat this prediction as foreground
           and assign the corresponding gt index (plus 1) to it.
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_bboxes (Tensor): Ground truth boxes with unnormalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`. Default None.
            eps (int | float, optional): A value added to the denominator for
                numerical stability. Default 1e-7.
        Returns:
            tuple:
                - AssignResult：每个预测 query 匹配到的 GT 编号；0 表示背景，
                  正整数 n 表示第 n-1 个 GT。
                - order_index [V,G]：每个“预测-GT”组合代价最小的 GT 点序编号。
        """
        #*==================== 地图预测与 GT 的 Hungarian 匹配 ====================#
        #* 完整训练逻辑（候选点序生成 -> 匹配 -> 构造 target -> 计算 loss）：
        #*
        #* 1) 数据集为每个 GT 地图实例生成 S 种等价有序点表示：
        #*    开放折线通常包含正序/反序，闭合折线包含不同起点的循环移位点序，
        #*    得到 gt_pts=[G,S,P_gt,2]。这些候选表示同一个 GT 几何实例，
        #*    不是 S 个不同 GT，也不是要求网络输出 S 条预测。
        #* 2) 模型固定输出 V 个地图 query；每个 query 预测类别、包围框和一条有序折线，
        #*    分别为 cls_pred=[V,C]、bbox_pred=[V,4]、pts_pred=[V,P_pred,2]。
        #* 3) 若 P_pred != P_gt，先沿点序维将预测折线线性插值为 P_gt 个点。
        #* 4) 每个预测 query 与每个 GT 的全部 S 个候选点序计算点代价，得到
        #*    pts_cost_ordered=[V,G,S]；在 S 维取最小值，得到 pts_cost=[V,G]，
        #*    同时用 order_index=[V,G] 记录每个“预测-GT”组合的最佳候选编号。
        #* 5) 将点代价与分类、包围框 L1、IoU 代价相加，形成总成本矩阵 cost=[V,G]。
        #* 6) Hungarian 在 cost 上寻找全局总代价最小的一对一预测-GT配对；匹配 query
        #*    是正样本，未匹配 query 是背景。注意：候选点序最小化发生在 Hungarian 之前，
        #*    因此最佳候选的点代价会直接影响实例配对结果。
        #* 7) 本函数返回 AssignResult 和 order_index。随后 VADHead._map_get_target_single()
        #*    对每个匹配对 (pred_i,gt_j) 读取 order_index[i,j]，从 gt_pts 中取出该 GT
        #*    对当前预测最合适的点序，写入 pts_targets；同时构造类别和包围框 targets。
        #* 8) VADHead.map_loss_single() 最后使用这些 targets 计算分类、框 L1、IoU、
        #*    PtsL1Loss 和 PtsDirCosLoss；此时匹配与最佳点序选择都已经完成。
        #*
        #* 简写：GT候选点序 [G,S,P,2] -> 全组合点代价 [V,G,S]
        #*      -> 候选维取最小 [V,G] -> 融合各项代价 -> Hungarian 一对一匹配
        #*      -> 取匹配 GT 的最佳点序 -> 构造 targets -> 计算各项 loss。
        #*
        #* V=num_query（固定数量的地图预测），G=num_gt（当前帧真实地图实例数），
        #* S=num_orders（同一 GT 的等价点序数），P=每种点序的采样点数。
        #* 先对每个 [预测,GT] 组合从 S 个点序中选出最小点代价，再把该代价与
        #* 分类、包围框 L1、IoU 代价相加，最后在 [V,G] 代价矩阵上做一对一匹配。
        assert gt_bboxes_ignore is None, \
            'Only case when gt_bboxes_ignore is None is supported.'
        assert bbox_pred.shape[-1] == 4, \
            'Only support bbox pred shape is 4 dims'
        num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)

        # 1. 初始化为 -1（ignore/尚未分配）；完成匹配后，未匹配 query 会被置为背景 0。
        assigned_gt_inds = bbox_pred.new_full((num_bboxes, ),
                                              -1,
                                              dtype=torch.long)
        assigned_labels = bbox_pred.new_full((num_bboxes, ),
                                             -1,
                                             dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels), None

        #*==================== 1. 构造各项 Pred-GT 匹配代价 ====================#
        # 分类代价：每个预测属于每个 GT 类别的代价，输出 [V,G]。
        cls_cost = self.cls_cost(cls_pred, gt_labels)  # [V,G]
        # 预测框是 [0,1] 归一化格式，先把米制 GT 框归一化到同一尺度。
        normalized_gt_bboxes = normalize_2d_bbox(gt_bboxes, self.pc_range)  # [G,4]
        # normalized_gt_bboxes = gt_bboxes
        # 包围框 L1 代价：逐一比较 V 个预测框与 G 个 GT 框，输出 [V,G]。
        reg_cost = self.reg_cost(
            bbox_pred[:, :4], normalized_gt_bboxes[:, :4])  # [V,G]

        # gt_pts=[G,S,P_gt,2]：每个 GT 含 S 种等价点序；2=(x,y)。
        _, num_orders, num_pts_per_gtline, num_coords = gt_pts.shape
        normalized_gt_pts = normalize_2d_pts(gt_pts, self.pc_range)  # [G,S,P_gt,2]
        num_pts_per_predline = pts_pred.size(1)
        # 若预测点数 P_pred 与 GT 点数 P_gt 不同，沿点序维重采样预测折线。
        if num_pts_per_predline != num_pts_per_gtline:
            pts_pred_interpolated = F.interpolate(
                pts_pred.permute(0, 2, 1), size=num_pts_per_gtline,
                mode='linear', align_corners=True)  # [V,2,P_gt]
            pts_pred_interpolated = \
                pts_pred_interpolated.permute(0, 2, 1).contiguous()  # [V,P_gt,2]
        else:
            pts_pred_interpolated = pts_pred  # [V,P_gt,2]

        #* 每个预测都与每个 GT 的所有 S 种候选点序计算点集匹配代价。
        # self.pts_cost 的展平输出随后恢复为 [V,G,S]。
        pts_cost_ordered = self.pts_cost(pts_pred_interpolated, normalized_gt_pts)
        pts_cost_ordered = pts_cost_ordered.view(
            num_bboxes, num_gts, num_orders)  # [V,G,S]
        # 在 S 维取最小值：pts_cost 是参与 Hungarian 的最佳点序代价；
        # order_index 保存最佳候选编号，匹配完成后据此构造 pts_targets。
        pts_cost, order_index = torch.min(pts_cost_ordered, dim=2)  # 均为 [V,G]

        # IoU 代价在米制坐标下计算，因此先反归一化预测框。
        bboxes = denormalize_2d_bbox(bbox_pred, self.pc_range)  # [V,4]
        iou_cost = self.iou_cost(bboxes, gt_bboxes)  # [V,G]
        # 各 cost 对象内部已包含配置中的权重；相加得到最终 [V,G] 代价矩阵。
        cost = cls_cost + reg_cost + iou_cost + pts_cost  # [V,G]

        #*==================== 2. Hungarian 实例级一对一匹配 ====================#
        # scipy 在 CPU 上求使总代价最小的预测-GT 配对；一个预测和一个 GT 最多使用一次。
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        # matched_row_inds=预测 query 编号；matched_col_inds=与其匹配的 GT 编号。
        matched_row_inds = torch.from_numpy(matched_row_inds).to(
            bbox_pred.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(
            bbox_pred.device)

        #*==================== 3. 记录匹配结果 ====================#
        # 先将所有 query 设为背景 0，再把匹配成功者设为对应 GT 的 1-based 编号。
        assigned_gt_inds[:] = 0
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        # AssignResult 决定正/负样本；order_index 供 VADHead 为正样本选出最佳 GT 点序。
        return AssignResult(
            num_gts, assigned_gt_inds, None,
            labels=assigned_labels), order_index
