import torch
import numpy as np
import torch.nn as nn
from mmcv.cnn import xavier_init
from mmcv.utils import ext_loader
from torch.nn.init import normal_
from mmcv.runner.base_module import BaseModule
from mmdet.models.utils.builder import TRANSFORMER
from torchvision.transforms.functional import rotate
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence

from projects.mmdet3d_plugin.VAD.modules.decoder import CustomMSDeformableAttention
from projects.mmdet3d_plugin.VAD.modules.temporal_self_attention import TemporalSelfAttention
from projects.mmdet3d_plugin.VAD.modules.spatial_cross_attention import MSDeformableAttention3D


ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])

def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.
    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class MapDetectionTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR3D transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default:
            `LN`.
    """

    def __init__(self, *args, return_intermediate=False, **kwargs):
        super(MapDetectionTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.fp16_enabled = False

    def forward(self,
                query,
                *args,
                reference_points=None,
                reg_branches=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `Detr3DTransformerDecoder`.
        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            reg_branch: (obj:`nn.ModuleList`): Used for
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        #* Map Decoder 逐层更新地图 Query，并用回归结果迭代细化其 BEV 参考点。
        output = query  # [V*P, B, D]
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            # Deformable Attention 要求显式的 feature-level 维度。
            reference_points_input = reference_points[..., :2].unsqueeze(
                2)  # BS NUM_QUERY NUM_LEVEL 2
            # 当前层从 BEV value 中采样与参考点相关的特征并更新 Map Query。
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                key_padding_mask=key_padding_mask,
                **kwargs)
            output = output.permute(1, 0, 2)

            if reg_branches is not None:
                # 使用当前层回归偏移修正参考点；inverse_sigmoid 后相加可在
                # logit 空间完成残差更新，再用 sigmoid 映射回归一化 BEV 坐标。
                tmp = reg_branches[lid](output)

                assert reference_points.shape[-1] == 2

                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points[..., :2] = tmp[
                    ..., :2] + inverse_sigmoid(reference_points[..., :2])
                # new_reference_points[..., 2:3] = tmp[
                #     ..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])

                new_reference_points = new_reference_points.sigmoid()

                # detach 阻断跨层参考点坐标的梯度链，采用 DETR 式逐层迭代优化。
                reference_points = new_reference_points.detach()

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return output, reference_points


@TRANSFORMER.register_module()
class VADPerceptionTransformer(BaseModule):
    """Implements the Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 two_stage_num_proposals=300,
                 encoder=None,
                 decoder=None,
                 map_decoder=None,
                 embed_dims=256,
                 rotate_prev_bev=True,
                 use_shift=True,
                 use_can_bus=True,
                 can_bus_norm=True,
                 use_cams_embeds=True,
                 rotate_center=[100, 100],
                 map_num_vec=50,
                 map_num_pts_per_vec=10,
                 **kwargs):
        super(VADPerceptionTransformer, self).__init__(**kwargs)
        self.encoder = build_transformer_layer_sequence(encoder)
        if decoder is not None:
            self.decoder = build_transformer_layer_sequence(decoder)
        else:
            self.decoder = None
        if map_decoder is not None:
            self.map_decoder = build_transformer_layer_sequence(map_decoder)
        else:
            self.map_decoder = None

        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.fp16_enabled = False
        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.can_bus_norm = can_bus_norm
        self.use_cams_embeds = use_cams_embeds
        self.two_stage_num_proposals = two_stage_num_proposals
        self.rotate_center = rotate_center
        self.map_num_vec = map_num_vec
        self.map_num_pts_per_vec = map_num_pts_per_vec
        self.init_layers()

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        # 不同 FPN 层和不同相机共享同一特征空间，分别添加可学习标识以区分来源。
        self.level_embeds = nn.Parameter(torch.Tensor(
            self.num_feature_levels, self.embed_dims))
        self.cams_embeds = nn.Parameter(
            torch.Tensor(self.num_cams, self.embed_dims))
        #* 从 Query 的位置编码生成归一化参考点：Agent 使用 3D 点，Map 使用 BEV 2D 点。
        self.reference_points = nn.Linear(self.embed_dims, 3)
        self.map_reference_points = nn.Linear(self.embed_dims, 2)
        # 将位移、转角、速度等 18 维 can_bus 自车状态编码到 D 维 BEV 特征空间。
        self.can_bus_mlp = nn.Sequential(
            nn.Linear(18, self.embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims // 2, self.embed_dims),
            nn.ReLU(inplace=True),
        )
        if self.can_bus_norm:
            self.can_bus_mlp.add_module('norm', nn.LayerNorm(self.embed_dims))

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention3D) or isinstance(m, TemporalSelfAttention) \
                    or isinstance(m, CustomMSDeformableAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        normal_(self.level_embeds)
        normal_(self.cams_embeds)
        xavier_init(self.reference_points, distribution='uniform', bias=0.)
        xavier_init(self.map_reference_points, distribution='uniform', bias=0.)
        xavier_init(self.can_bus_mlp, distribution='uniform', bias=0.)

    # TODO apply fp16 to this module cause grad_norm NAN
    # @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'prev_bev', 'bev_pos'))
    def get_bev_features(
            self,
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=[0.512, 0.512],
            bev_pos=None,
            prev_bev=None,
            **kwargs):
        """
        obtain bev features.
        """

        #*==================== 1. 初始化 BEV Query 与位置编码 ====================#
        bs = mlvl_feats[0].size(0)
        # 为 batch 中每个样本复制同一组可学习 BEV Query：[HW,D] -> [HW,B,D]。
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)
        # BEV 位置编码展平：[B,D,H,W] -> [HW,B,D]。
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)

        #*==================== 2. 根据自车运动计算历史 BEV 平移量 ====================#
        # can_bus[0:2] 是相对上一帧的全局位移，can_bus[-2] 是当前 ego 朝向。
        delta_x = np.array([each['can_bus'][0]
                           for each in kwargs['img_metas']])
        delta_y = np.array([each['can_bus'][1]
                           for each in kwargs['img_metas']])
        ego_angle = np.array(
            [each['can_bus'][-2] / np.pi * 180 for each in kwargs['img_metas']])
        grid_length_y = grid_length[0]
        grid_length_x = grid_length[1]
        translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
        translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
        bev_angle = ego_angle - translation_angle
        shift_y = translation_length * \
            np.cos(bev_angle / 180 * np.pi) / grid_length_y / bev_h
        shift_x = translation_length * \
            np.sin(bev_angle / 180 * np.pi) / grid_length_x / bev_w
        shift_y = shift_y * self.use_shift
        shift_x = shift_x * self.use_shift
        shift = bev_queries.new_tensor(
            [shift_x, shift_y]).permute(1, 0)  # xy, bs -> bs, xy

        #*==================== 3. 旋转对齐上一帧 BEV ====================#
        if prev_bev is not None:
            # 统一 prev_bev 为 [HW,B,D]，以便送入 Temporal Self-Attention。
            if prev_bev.shape[1] == bev_h * bev_w:
                prev_bev = prev_bev.permute(1, 0, 2)
            if self.rotate_prev_bev:
                for i in range(bs):
                    # 按自车相对旋转角旋转历史 BEV，使其与当前帧坐标方向对齐。
                    rotation_angle = kwargs['img_metas'][i]['can_bus'][-1]
                    tmp_prev_bev = prev_bev[:, i].reshape(
                        bev_h, bev_w, -1).permute(2, 0, 1)
                    tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle,
                                          center=self.rotate_center)
                    tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(
                        bev_h * bev_w, 1, -1)
                    prev_bev[:, i] = tmp_prev_bev[:, 0]

        #*==================== 4. 将自车状态注入 BEV Query ====================#
        can_bus = bev_queries.new_tensor(
            [each['can_bus'] for each in kwargs['img_metas']])  # [:, :]
        can_bus = self.can_bus_mlp(can_bus)[None, :, :]
        bev_queries = bev_queries + can_bus * self.use_can_bus

        #*==================== 5. 展平并拼接多尺度、多相机图像特征 ====================#
        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            # [B,N_cam,D,H,W] -> [N_cam,B,HW,D]。
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            if self.use_cams_embeds:
                # 注入相机身份，使网络能够区分同一位置来自哪个摄像头。
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            # 注入 FPN 层级身份，使网络能够区分不同分辨率的特征。
            feat = feat + self.level_embeds[None,
                                            None, lvl:lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=bev_pos.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat_flatten.permute(
            0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)

        #*==================== 6. BEV Encoder：图像到 BEV + 时序融合 ====================#
        # Encoder 内部通常包含：
        # Temporal Self-Attention：结合 prev_bev 和 shift 聚合历史信息；
        # Spatial Cross-Attention：BEV Query 投影到各相机视角，从图像特征采样信息。
        bev_embed = self.encoder(
            bev_queries,
            feat_flatten,
            feat_flatten,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            prev_bev=prev_bev,
            shift=shift,
            **kwargs
        )

        return bev_embed  # [B, bev_h*bev_w, D]

    # TODO apply fp16 to this module cause grad_norm NAN
    # @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'object_query_embed', 'prev_bev', 'bev_pos'))
    def forward(self,
                mlvl_feats,
                bev_queries,
                object_query_embed,
                map_query_embed,
                bev_h,
                bev_w,
                grid_length=[0.512, 0.512],
                bev_pos=None,
                reg_branches=None,
                cls_branches=None,
                map_reg_branches=None,
                map_cls_branches=None,                
                prev_bev=None,            
                **kwargs):
        """Forward function for `Detr3DTransformer`.
        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, num_cams, embed_dims, h, w].
            bev_queries (Tensor): (bev_h*bev_w, c)
            bev_pos (Tensor): (bs, embed_dims, bev_h, bev_w)
            object_query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when `with_box_refine` is True. Default to None.
        Returns:
            tuple[Tensor]: BEV、Agent 和 Map 三路输出：
                - bev_embed: [bev_h*bev_w, B, D]；
                - inter_states: [Ld, A, B, D]；
                - init_reference_out: [B, A, 3]；
                - inter_references_out: [Ld, B, A, 3]；
                - map_inter_states: [Lm, V*P, B, D]；
                - map_init_reference_out: [B, V*P, 2]；
                - map_inter_references_out: [Lm, B, V*P, 2]。
        """

        #*==================== 1. 生成当前帧时空 BEV 特征 ====================#
        bev_embed = self.get_bev_features(
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=grid_length,
            bev_pos=bev_pos,
            prev_bev=prev_bev,
            **kwargs)  # bev_embed shape: bs, bev_h*bev_w, embed_dims

        #*==================== 2. 拆分 Agent Query 并生成 3D 参考点 ====================#
        bs = mlvl_feats[0].size(0)
        # 输入的 [A,2D] 沿通道拆成位置编码 query_pos 和内容 query，各为 [A,D]。
        query_pos, query = torch.split(
            object_query_embed, self.embed_dims, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_pos)
        # Agent 初始参考点为归一化 (x,y,z)，shape=[B,A,3]。
        reference_points = reference_points.sigmoid()
        init_reference_out = reference_points

        #*==================== 3. 拆分 Map Query 并生成 2D 参考点 ====================#
        #* Map Query 采用“实例 × 点”的形式：共 V 个地图实例槽位，每个实例用 P 个有序点表示。
        # 输入 map_query_embed=[V*P,2D]；每行对应“某个地图实例的某个采样点”。
        # 前 D 维作为位置编码 map_query_pos，后 D 维作为内容特征 map_query。
        map_query_pos, map_query = torch.split(
            map_query_embed, self.embed_dims, dim=1)
        # 为 batch 复制查询：[V*P,D] -> [B,V*P,D]。
        map_query_pos = map_query_pos.unsqueeze(0).expand(bs, -1, -1)
        map_query = map_query.unsqueeze(0).expand(bs, -1, -1)
        map_reference_points = self.map_reference_points(map_query_pos)
        # Map 初始参考点位于归一化 BEV 平面，shape=[B,V*P,2]。
        map_reference_points = map_reference_points.sigmoid()
        map_init_reference_out = map_reference_points        

        # Decoder 使用 query-first 布局 [Q,B,D]；BEV 作为其 value/memory。
        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        map_query = map_query.permute(1, 0, 2)
        map_query_pos = map_query_pos.permute(1, 0, 2)
        bev_embed = bev_embed.permute(1, 0, 2)

        #*==================== 4. Agent Decoder：从 BEV 解码动态目标 ====================#
        #* self.decoder 的实际类型由配置指定为 DetectionTransformerDecoder，实现在
        #* projects/mmdet3d_plugin/VAD/modules/decoder.py。
        # 每层执行 Agent Query self-attention、对 BEV 的 deformable cross-attention 和 FFN：
        # Agent 间先交换信息，再以 reference_points 为采样中心读取 BEV，最后更新 Query。
        if self.decoder is not None:
            inter_states, inter_references = self.decoder(
                query=query,              # Agent 内容查询，[A,B,D]，是 decoder 要持续更新的状态
                key=None,                 # 本实现的 deformable cross-attention 不单独传 key
                value=bev_embed,          # 可供 Agent Query 读取的 BEV memory，[HW,B,D]
                query_pos=query_pos,      # Agent Query 的可学习位置编码，[A,B,D]
                reference_points=reference_points,  # Agent 初始采样中心 (x,y,z)，[B,A,3]
                reg_branches=reg_branches,  # 各层 3D 框回归头，用预测偏移迭代细化参考点
                cls_branches=cls_branches,  # two-stage 模式下传入的各层分类头；当前配置通常为 None
                # Decoder 的 value 只有一个 BEV 特征层，空间大小为 bev_h × bev_w。
                spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
                level_start_index=torch.tensor([0], device=query.device),  # 唯一 BEV 层从索引 0 开始
                **kwargs)
            #* inter_states：每层更新后的 Agent Query，[Ld,A,B,D]；之后用于分类和 3D 框回归。
            #* inter_references：每层更新后的归一化参考点 (x,y,z)，[Ld,B,A,3]；
            #* 下一层围绕新参考点从 BEV 采样，从而逐层修正目标位置。
            inter_references_out = inter_references
        else:
            # 没有配置 decoder 时，不做 BEV 信息读取；仅增加“层”维度保持接口一致。
            # inter_states=[1,A,B,D]；inter_references_out=[1,B,A,3]。
            inter_states = query.unsqueeze(0)
            inter_references_out = reference_points.unsqueeze(0)

        #*==================== 5. Map Decoder：从 BEV 解码矢量地图 ====================#
        #* self.map_decoder 的实际类型是本文件开头定义的 MapDetectionTransformerDecoder。
        # 每个 Map Query 当前表示一个地图实例中的一个采样点，因此 Query 总数为 V*P：
        # V 是地图实例候选数，P 是每个实例的固定采样点数。
        # 每层先让 Map Queries 通过 self-attention 交换结构信息，再以二维参考点为中心
        # 对 BEV 做 deformable cross-attention，最终学习车道线、道路边界等矢量地图特征。
        if self.map_decoder is not None:
            map_inter_states, map_inter_references = self.map_decoder(
                query=map_query,          # Map 点内容查询，[V*P,B,D]，是 decoder 持续更新的状态
                key=None,                 # deformable cross-attention 不单独传 key
                value=bev_embed,          # Map Query 读取的 BEV memory，[HW,B,D]
                query_pos=map_query_pos,  # Map 点查询的可学习位置编码，[V*P,B,D]
                reference_points=map_reference_points,  # 初始二维采样中心 (x,y)，[B,V*P,2]
                reg_branches=map_reg_branches,  # 各层地图点回归头，用预测偏移细化二维参考点
                cls_branches=map_cls_branches,  # 地图分类头接口；类别最终在 VADHead 中统一计算
                # Decoder 的 value 仅包含一个 bev_h × bev_w 的 BEV 特征层。
                spatial_shapes=torch.tensor([[bev_h, bev_w]], device=map_query.device),
                level_start_index=torch.tensor([0], device=map_query.device),  # 唯一 BEV 层从 0 开始
                **kwargs)
            #* map_inter_states：每层更新后的逐点 Map Query，[Lm,V*P,B,D]：
            # Lm = Map Decoder 层数；V = 地图实例候选数；P = 每个实例的有序点数；
            # B = batch size；D = 每个地图点 Query 的 embedding 维度。
            # 因此 map_inter_states[l, v*P+p, b] 表示：第 l 层中，第 b 个样本的
            # 第 v 个地图实例里第 p 个有序点对应的 D 维特征。
            #* map_inter_references：每层更新后的归一化二维参考点，[Lm,B,V*P,2]；
            #* 下一层围绕新参考点读取 BEV，从而逐层修正矢量点位置。
            # VADHead 随后把 V*P 还原成 [V,P]：点特征均值用于实例分类，
            # P 个二维点共同组成一个车道线或道路边界实例。
            map_inter_references_out = map_inter_references
        else:
            # 未配置 Map Decoder 时不读取 BEV，只增加“层”维度以保持返回接口一致。
            # map_inter_states=[1,V*P,B,D]；map_inter_references_out=[1,B,V*P,2]。
            map_inter_states = map_query.unsqueeze(0)
            map_inter_references_out = map_reference_points.unsqueeze(0)

        #*==================== 6. 返回 BEV、Agent 与 Map 三路结果 ====================#
        # 返回顺序必须与 VADHead.forward() 中的 outputs 解包顺序严格一致。
        return (
            bev_embed,              # BEV memory，[HW,B,D]；HW=bev_h*bev_w，B=批大小，D=特征维度
            inter_states,           # Agent特征，[Ld,A,B,D]；Ld=Agent解码层数，A=Agent Query数，B=批大小，D=特征维度
            init_reference_out,     # Agent初始点(x,y,z)，[B,A,3]；B=批大小，A=Agent Query数
            inter_references_out,   # Agent更新点，[Ld,B,A,3]；Ld=Agent解码层数，B=批大小，A=Agent Query数
            map_inter_states,       # Map点特征，[Lm,V*P,B,D]；Lm=Map解码层数，V=实例数，P=每实例点数，B=批大小，D=特征维度
            map_init_reference_out, # Map初始点(x,y)，[B,V*P,2]；B=批大小，V=地图实例数，P=每实例点数
            map_inter_references_out,  # Map更新点，[Lm,B,V*P,2]；Lm=Map解码层数，B=批大小，V=实例数，P=每实例点数
        )


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class CustomTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR3D transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default: `LN`.
    """

    def __init__(self, *args, return_intermediate=False, **kwargs):
        super(CustomTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.fp16_enabled = False

    def forward(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                key_padding_mask=None,
                *args,
                **kwargs):
        """Forward function for `Detr3DTransformerDecoder`.
        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        #* 通用 Query Decoder：按配置执行 self-attention / cross-attention / FFN。
        # 在 VAD 中可用于运动交互、Ego-Agent 交互和 Ego-Map 交互等模块。
        intermediate = []
        for lid, layer in enumerate(self.layers):
            # query 是被更新对象；key/value 是其需要读取的上下文信息。
            query = layer(
                query=query,
                key=key,
                value=value,
                query_pos=query_pos,
                key_pos=key_pos,
                attn_masks=attn_masks,
                key_padding_mask=key_padding_mask,
                *args,
                **kwargs)

            if self.return_intermediate:
                # 需要深监督或分析中间层时，保存每一层的 Query 表示。
                intermediate.append(query)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return query
