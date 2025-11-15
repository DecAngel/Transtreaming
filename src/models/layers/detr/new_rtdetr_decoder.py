from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init

from .detr_utils import get_contrastive_denoising_training_group, inverse_sigmoid
from .rtdetrv2_decoder import RTDETRTransformerv2, TransformerDecoder, TransformerDecoderLayer, MLP, MSDeformableAttention
from src.utils.pylogger import RankedLogger
from src.utils.inspection import inspect


logger = RankedLogger(__name__, rank_zero_only=True)

TypeDecoder = Literal["default", "velocity", "perception", "spin", "double"]
TypeQuery = Literal["default", "mixed"]
TypeEmbedding = Literal["default", "temporal"]
TypeOffset = Literal["default", "velocity"]


# shapes:
# Fp: (B, T, L_v, C)
# queries: (B, L_q, C)
# idx_q: (B, L_q)  # integer spatial index into L_v
# t_ref_q: (B, L_q)  # integer ref frame index for each query (or scalar)

class TemporalDeformableAttention(nn.Module):
    def __init__(self, C, num_heads=8, K=4):
        super().__init__()
        self.num_heads = num_heads
        self.K = K
        self.head_dim = C // num_heads
        self.offset_predictor = nn.Linear(C, num_heads * K)  # predict Δt (raw)
        self.attn_weight_predictor = nn.Linear(C, num_heads * K)  # raw logits for K samples
        self.value_proj = nn.Linear(C, C)
        self.output_proj = nn.Linear(C, C)
        # init offsets small
        nn.init.constant_(self.offset_predictor.bias, 0.0)
        # optionally scale weights
        self.scale = self.head_dim ** -0.5

    def forward(self, Fp, queries, idx_q, t_ref_q):
        B, T, L_v, C = Fp.shape
        _, L_q, _ = queries.shape
        # project values
        V = self.value_proj(Fp)  # (B, T, L_v, C)

        # predict offsets and attn logits
        offsets = self.offset_predictor(queries)  # (B, L_q, H*K)
        attn_logits = self.attn_weight_predictor(queries)  # (B, L_q, H*K)
        offsets = offsets.view(B, L_q, self.num_heads, self.K)  # (B,Lq,H,K)
        attn_logits = attn_logits.view(B, L_q, self.num_heads, self.K)

        attn_weights = torch.softmax(attn_logits, dim=-1)  # over K

        # for each sample compute sampled features:
        # build t_sample = t_ref + offsets (broadcast)
        t_ref = t_ref_q.unsqueeze(2).unsqueeze(3)  # (B, L_q, 1, 1)
        t_sample = t_ref + offsets  # (B, L_q, H, K)
        # clamp to valid range [0, T-1]
        t_sample = t_sample.clamp(0.0, float(T - 1))

        # Now gather V at (batch, t_floor/ceil, idx_q)
        # idx_q: (B, L_q) -> expand to shape (B, L_q, H, K)
        idx_exp = idx_q.unsqueeze(2).unsqueeze(3).expand(B, L_q, self.num_heads, self.K)

        t0 = torch.floor(t_sample).long()  # (B,Lq,H,K)
        t1 = (t0 + 1).clamp(max=T-1)
        alpha = (t_sample - t0.float()).unsqueeze(-1)  # alpha shape (B,Lq,H,K,1)

        # gather features: V[b, t, idx, :] -> we need to index for all combos
        # reshape indexing to vectorized gather:
        # convert to linear index or use advanced indexing
        b_idx = torch.arange(B, device=Fp.device)[:, None, None, None]  # (B,1,1,1)
        b_idx = b_idx.expand(B, L_q, self.num_heads, self.K)

        v0 = V[b_idx, t0, idx_exp]  # (B,Lq,H,K,C)
        v1 = V[b_idx, t1, idx_exp]  # (B,Lq,H,K,C)
        sampled = (1 - alpha) * v0 + alpha * v1  # (B,Lq,H,K,C)

        # merge head dim: split last C into num_heads x head_dim if desired
        sampled = sampled.view(B, L_q, self.num_heads, self.K, self.head_dim)
        attn_weights = attn_weights.unsqueeze(-1)  # (B,Lq,H,K,1)
        head_out = (attn_weights * sampled).sum(dim=3)  # sum over K -> (B,Lq,H,head_dim)
        head_out = head_out.view(B, L_q, C)

        out = self.output_proj(head_out)  # (B,Lq,C)
        return out


class SpatialTemporalDeformableAttention(MSDeformableAttention):
    def __init__(
            self,
            embed_dim=256,
            num_heads=8,
            num_levels=4,
            num_points=4,
            method='default',
            offset_scale=0.5,

            type_offset: TypeOffset = "default",
    ):
        super().__init__(
            embed_dim,
            num_heads,
            num_levels,
            num_points,
            method=method,
            offset_scale=offset_scale,
        )

        self.type_offset = type_offset
        if self.type_offset == "velocity":
            self.sampling_offsets_2 = nn.Linear(embed_dim, self.total_points * 2)
            if method == 'discrete':
                for p in self.sampling_offsets_2.parameters():
                    p.requires_grad = False

            nn.init.constant_(self.sampling_offsets_2.weight, 0)
            nn.init.constant_(self.sampling_offsets_2.bias, 0)

    def forward(self,
                query: torch.Tensor,
                reference_points: torch.Tensor,
                value: torch.Tensor,
                value_spatial_shapes: list[tuple[int, int]],
                value_mask: torch.Tensor = None,

                future_time: torch.Tensor = None,):
        # query: B Lq C
        # ref: B Lq 1 nlevel 4
        # value: B Lv C
        # shape: list<nlevel>[tuple<2>[height,width]]
        # mask: B Lv
        # future_time: B TF
        bs, Len_q = query.shape[:2]
        Len_v = value.shape[1]
        _, TF = future_time.shape

        value = self.value_proj(value)
        if value_mask is not None:
            value = value * value_mask.to(value.dtype).unsqueeze(-1)

        value = value.reshape(bs, Len_v, self.num_heads, self.head_dim)

        sampling_offsets: torch.Tensor = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.reshape(bs, Len_q, self.num_heads, sum(self.num_points_list), 2)

        if self.type_offset == "velocity":
            assert future_time is not None, 'future_time cannot be None if use offset_2'
            sampling_offsets_2: torch.Tensor = self.sampling_offsets_2(query)
            sampling_offsets_2 = sampling_offsets_2.reshape(bs, TF, Len_q, self.num_heads, sum(self.num_points_list), 2)
            sampling_offsets_2 = sampling_offsets_2 * future_time[..., None, None, None, None]
            sampling_offsets_2 = sampling_offsets_2.flatten(0, 1)

            # B*TF Lq nhead npoint 2
            sampling_offsets = sampling_offsets + sampling_offsets_2

        attention_weights = self.attention_weights(query).reshape(bs, Len_q, self.num_heads, sum(self.num_points_list))
        attention_weights = F.softmax(attention_weights, dim=-1).reshape(bs, Len_q, self.num_heads,
                                                                         sum(self.num_points_list))

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.tensor(value_spatial_shapes)
            offset_normalizer = offset_normalizer.flip([1]).reshape(1, 1, 1, self.num_levels, 1, 2)
            sampling_locations = reference_points.reshape(bs, Len_q, 1, self.num_levels, 1,
                                                          2) + sampling_offsets / offset_normalizer
        elif reference_points.shape[-1] == 4:
            # reference_points [8, 480, None, 1,  4]
            # sampling_offsets [8, 480, 8,    12, 2]
            num_points_scale = self.num_points_scale.to(dtype=query.dtype).unsqueeze(-1)
            offset = sampling_offsets * num_points_scale * reference_points[:, :, None, :, 2:] * self.offset_scale
            sampling_locations = reference_points[:, :, None, :, :2] + offset
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".
                format(reference_points.shape[-1]))

        output = self.ms_deformable_attn_core(value, value_spatial_shapes, sampling_locations, attention_weights,
                                              self.num_points_list)

        output = self.output_proj(output)

        return output


class SpatialTemporalTransformerDecoderLayer(TransformerDecoderLayer):
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation='relu',
                 n_levels=4,
                 n_points=4,
                 cross_attn_method='default',

                 type_offset: TypeOffset = "default",
                 ):
        super().__init__(
            d_model,
            n_head,
            dim_feedforward,
            dropout,
            activation,
            n_levels,
            n_points,
            cross_attn_method
        )
        # ST attention
        self.cross_attn = SpatialTemporalDeformableAttention(
            d_model, n_head, n_levels, n_points, method=cross_attn_method,
            type_offset=type_offset,
        )

        # temporal attention
        # self.temp_attn = nn.MultiheadAttention(
        #     d_model,
        #     n_head,
        #     dropout=dropout,
        #     batch_first=True,
        # )
        # self.dropout5 = nn.Dropout(dropout)
        # self.norm5 = nn.LayerNorm(d_model)

    def forward(self,
                target,
                reference_points,
                memory,
                memory_spatial_shapes,
                attn_mask=None,
                memory_mask=None,
                query_pos_embed=None,

                future_time: torch.Tensor = None,
                # temp_memory=None,
                # future_pos_embed=None,
                # past_pos_embed=None,
                ):
        # target: B*TF Lq C
        # ref_point: B Lq 1 nlevel 4
        # memory: B Lv C
        # shape: list<nlevel>[tuple<2>[height,width]]
        # attn_mask: Lq Lv
        # memory_mask: B Lv
        # query_pos_embed: 1 Lq C

        # future_time: B TF
        # temp_memory B*TF
        # temp_pos_embed: 1 Lv C
        # change ref_point and memory to
        # logger.info(f'DecoderLayer: {inspect(locals())}')

        # self attention
        q = k = self.with_pos_embed(target, query_pos_embed)

        target2, _ = self.self_attn(q, k, value=target, attn_mask=attn_mask)
        target = target + self.dropout1(target2)
        target = self.norm1(target)

        # cross attention
        target2 = self.cross_attn(
            self.with_pos_embed(target, query_pos_embed),
            reference_points,
            memory,
            memory_spatial_shapes,
            memory_mask,
            future_time=future_time,
        )
        target = target + self.dropout2(target2)
        target = self.norm2(target)

        # temporal cross attention
        # if temp_memory is not None:
        # q = self.with_pos_embed(target, query_pos_embed)
        # k = self.with_pos_embed(temp_memory, query_pos_embed)
        # v = temp_memory
        # target2, _ = self.temp_attn(q, k, value=v)
        # target = target + self.dropout5(target2)
        # target = self.norm5(target)

        # ffn
        target2 = self.forward_ffn(target)
        target = target + self.dropout4(target2)
        target = self.norm3(target)

        return target


class SpatialTemporalTransformerDecoder(TransformerDecoder):
    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1,
                type_decoder: TypeDecoder = "default",):
        super().__init__(hidden_dim, decoder_layer, num_layers, eval_idx)
        self.type_decoder = type_decoder

    def forward(self,
                target,
                ref_points_unact,
                memory,
                memory_spatial_shapes,
                bbox_head,
                score_head,
                query_pos_head,
                attn_mask=None,
                memory_mask=None,

                future_time=None,
                temp_pos_embed=None,
                velocity_head=None,
                ):
        # target: B*TF Lq C
        # ref_point: B*TF Lq 4
        # memory: B*TF Lv C
        # shape: list<nlevel>[tuple<2>[height,width]]
        # attn_mask: Lq Lv
        # memory_mask: B Lv
        # query_pos_embed: 1 Lq C
        # temp_pos_embed: B*TF 1 1 C

        # future_time: B TF
        # logger.info(f'Decoder: {inspect(locals())}')

        dec_out_bboxes = []
        dec_out_logits = []
        ref_points_detach = F.sigmoid(ref_points_unact)

        output = target
        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach)
            # query_pos_embed: B*TF Lq C
            if temp_pos_embed is not None:
                query_pos_embed = query_pos_embed + temp_pos_embed

            output = layer(output, ref_points_input, memory, memory_spatial_shapes,
                           attn_mask=attn_mask, memory_mask=memory_mask, query_pos_embed=query_pos_embed,
                           future_time=future_time,)

            if velocity_head is not None:
                B, TF = future_time.shape
                if self.training and i > 0:
                    pos = inverse_sigmoid(ref_points)
                else:
                    pos = inverse_sigmoid(ref_points_detach)
                offset = bbox_head[i](output)
                inter_ref_bbox = F.sigmoid(pos + offset)
                # logger.info(f"Decoder layer {i}: {inspect(locals())}")

                velocity = velocity_head[i](output)
                if self.type_decoder == 'velocity':
                    # velocity:
                    vel_ref_bbox = pos[:, None] + offset[:, None] + velocity[:, None] * future_time[..., None, None]
                    vel_ref_bbox = F.sigmoid(vel_ref_bbox.flatten(0, 1))
                elif self.type_decoder == 'perception':
                    # perception
                    dx, dy, log_s = (velocity[:, None] * future_time[..., None, None]).unbind(-1)
                    vel_ref_bbox_origin = (pos[:, None] + offset[:, None]).tile([1, TF, 1, 1])
                    hw = vel_ref_bbox_origin[..., 2:] * torch.exp(log_s[..., None])
                    cx = (vel_ref_bbox_origin[..., 0] + dx).unsqueeze(-1)
                    cy = (vel_ref_bbox_origin[..., 1] + dy).unsqueeze(-1)
                    vel_ref_bbox = torch.cat([cx, cy, hw], dim=-1)
                    vel_ref_bbox = F.sigmoid(vel_ref_bbox.flatten(0, 1))
                elif self.type_decoder == 'spin':
                    # b t L
                    vx, vy, sx, sy = velocity[:, None].unbind(-1)
                    # cx, cy, w, h = inter_ref_bbox[:, None].tile([1, TF, 1, 1]).unbind(-1)  # 0-1
                    cx, cy, w, h = (pos + offset)[:, None].tile([1, TF, 1, 1]).unbind(-1)
                    t = future_time[..., None]

                    # cx1 = cx + (vx + torch.expm1(sx) * (cx - 0.5)) * t
                    # cy1 = cy + (vy + torch.expm1(sy) * (cy - 0.5)) * t
                    # w1 = w * torch.exp(sx * t)
                    # h1 = h * torch.exp(sy * t)
                    cx1 = cx + vx * t
                    cy1 = cy + vy * t
                    w1 = w * torch.exp(sx * t)
                    h1 = h * torch.exp(sy * t)

                    vel_ref_bbox = torch.stack([cx1, cy1, w1, h1], dim=-1)
                    # vel_ref_bbox = vel_ref_bbox.flatten(0, 1)
                    vel_ref_bbox = F.sigmoid(vel_ref_bbox.flatten(0, 1))

                    # logger.info(f"decoder output: {inspect(locals())}")
                elif self.type_decoder == "double":
                    cx, cy, w, h = (pos + offset)[:, None].tile([1, TF, 1, 1]).unbind(-1)
                    vx, vy, vw, vh, ax, ay, aw, ah = velocity[:, None].unbind(-1)
                    t = future_time[..., None]
                    t2 = t ** 2

                    cx1 = cx + vx * t + ax * t2
                    cy1 = cy + vy * t + ay * t2
                    w1 = w + vw * t + aw * t2
                    h1 = h + vh * t + ah * t2

                    vel_ref_bbox = torch.stack([cx1, cy1, w1, h1], dim=-1)
                    vel_ref_bbox = F.sigmoid(vel_ref_bbox.flatten(0, 1))
                else:
                    raise ValueError(f"velocity shape is incorrect: {velocity.shape}")
                # B*TF L 4
                score = torch.repeat_interleave(score_head[i](output), repeats=TF, dim=0)

                if self.training:
                    dec_out_logits.append(score)
                    dec_out_bboxes.append(vel_ref_bbox)
                elif i == self.eval_idx:
                    dec_out_logits.append(score)
                    dec_out_bboxes.append(vel_ref_bbox)
                    break

                ref_points = inter_ref_bbox
                ref_points_detach = inter_ref_bbox.detach()

            else:
                inter_ref_bbox = F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points_detach))

                if self.training:
                    dec_out_logits.append(score_head[i](output))
                    if i == 0:
                        dec_out_bboxes.append(inter_ref_bbox)
                    else:
                        dec_out_bboxes.append(F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points)))

                elif i == self.eval_idx:
                    dec_out_logits.append(score_head[i](output))
                    dec_out_bboxes.append(inter_ref_bbox)
                    break

                ref_points = inter_ref_bbox
                ref_points_detach = inter_ref_bbox.detach()

        return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits)


class AdaptiveRTDETRTransformer(RTDETRTransformerv2):
    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 feat_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 num_levels=3,
                 num_points=4,
                 nhead=8,
                 num_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 num_denoising=100,
                 label_noise_ratio=0.5,
                 box_noise_scale=1.0,
                 learn_query_content=False,
                 eval_spatial_size=None,
                 eval_idx=-1,
                 eps=1e-2,
                 aux_loss=True,
                 cross_attn_method='default',
                 query_select_method='default',

                 type_decoder: TypeDecoder = "default",
                 type_query: TypeQuery = "default",
                 type_embedding: TypeEmbedding = "default",
                 type_offset: TypeOffset = "default",
                 # use_temp_attention=False,
                 ):
        super().__init__(
            num_classes,
            hidden_dim,
            num_queries,
            feat_channels,
            feat_strides,
            num_levels,
            num_points,
            nhead,
            num_layers,
            dim_feedforward,
            dropout,
            activation,
            num_denoising,
            label_noise_ratio,
            box_noise_scale,
            (type_query == "mixed") or learn_query_content,
            eval_spatial_size,
            eval_idx,
            eps,
            aux_loss,
            cross_attn_method,
            query_select_method,
        )
        self.type_decoder = type_decoder
        self.type_query = type_query
        self.type_embedding = type_embedding
        self.type_offset = type_offset

        decoder_layer = SpatialTemporalTransformerDecoderLayer(
            hidden_dim,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            num_levels,
            num_points,
            cross_attn_method=cross_attn_method,
            type_offset=type_offset,
        )

        self.decoder = SpatialTemporalTransformerDecoder(
            hidden_dim,
            decoder_layer,
            num_layers,
            eval_idx,
            type_decoder=type_decoder,
        )

        if self.type_decoder == "velocity":
            self.enc_velocity_head = MLP(hidden_dim, hidden_dim, 4, 3)
            init.constant_(self.enc_velocity_head.layers[-1].weight, 0)
            init.constant_(self.enc_velocity_head.layers[-1].bias, 0)
            self.dec_velocity_head = nn.ModuleList([
                MLP(hidden_dim, hidden_dim, 4, 3) for _ in range(num_layers)
            ])
            for _reg in self.dec_velocity_head:
                init.constant_(_reg.layers[-1].weight, 0)
                init.constant_(_reg.layers[-1].bias, 0)
        elif self.type_decoder == "perception":
            self.enc_velocity_head = MLP(hidden_dim, hidden_dim, 3, 3)
            init.constant_(self.enc_velocity_head.layers[-1].weight, 0)
            init.constant_(self.enc_velocity_head.layers[-1].bias, 0)
            self.dec_velocity_head = nn.ModuleList([
                MLP(hidden_dim, hidden_dim, 3, 3) for _ in range(num_layers)
            ])
            for _reg in self.dec_velocity_head:
                init.constant_(_reg.layers[-1].weight, 0)
                init.constant_(_reg.layers[-1].bias, 0)
        elif self.type_decoder == 'spin':
            self.enc_velocity_head = MLP(hidden_dim, hidden_dim, 4, 3)
            init.constant_(self.enc_velocity_head.layers[-1].weight, 0)
            init.constant_(self.enc_velocity_head.layers[-1].bias, 0)
            self.dec_velocity_head = nn.ModuleList([
                MLP(hidden_dim, hidden_dim, 4, 3) for _ in range(num_layers)
            ])
            for _reg in self.dec_velocity_head:
                init.constant_(_reg.layers[-1].weight, 0)
                init.constant_(_reg.layers[-1].bias, 0)
        elif self.type_decoder == 'double':
            self.enc_velocity_head = MLP(hidden_dim, hidden_dim, 8, 3)
            init.constant_(self.enc_velocity_head.layers[-1].weight, 0)
            init.constant_(self.enc_velocity_head.layers[-1].bias, 0)
            self.dec_velocity_head = nn.ModuleList([
                MLP(hidden_dim, hidden_dim, 8, 3) for _ in range(num_layers)
            ])
            for _reg in self.dec_velocity_head:
                init.constant_(_reg.layers[-1].weight, 0)
                init.constant_(_reg.layers[-1].bias, 0)

        if self.type_embedding == "temporal":
            self.temp_pos_head = MLP(1, 2 * hidden_dim, hidden_dim, 2)
            init.xavier_uniform_(self.temp_pos_head.layers[0].weight)
            init.xavier_uniform_(self.temp_pos_head.layers[1].weight)


    def _get_decoder_input_ex(
            self,
            memory: torch.Tensor,
            spatial_shapes,
            denoising_logits=None,
            denoising_bbox_unact=None,

            future_time=None,
    ):
        # memory: B Lq C
        # shape: nlevel 2

        # prepare input for decoder
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        else:
            anchors = self.anchors
            valid_mask = self.valid_mask

        # memory = torch.where(valid_mask, memory, 0)
        # TODO fix type error for onnx export
        memory = valid_mask.to(memory.dtype) * memory

        output_memory: torch.Tensor = self.enc_output(memory)
        enc_outputs_logits: torch.Tensor = self.enc_score_head(output_memory)
        enc_outputs_coord_unact: torch.Tensor = self.enc_bbox_head(output_memory) + anchors
        enc_topk_bboxes_list, enc_topk_logits_list = [], []

        if self.type_decoder in ['velocity', 'perception', 'spin', 'double']:
            enc_outputs_vel_unact = self.enc_velocity_head(output_memory)
            enc_topk_memory, enc_topk_logits, enc_topk_bbox_unact, enc_topk_vel_unact = self._select_topk_ex(
                output_memory, enc_outputs_logits, enc_outputs_coord_unact, enc_outputs_vel_unact, self.num_queries
            )
        elif self.type_decoder == 'default':
            enc_topk_memory, enc_topk_logits, enc_topk_bbox_unact = \
                self._select_topk(output_memory, enc_outputs_logits, enc_outputs_coord_unact, self.num_queries)
        else:
            raise ValueError(f"Unknown type_decoder: {self.type_decoder}")

        if self.training:
            if self.type_decoder == "velocity":
                # B L 4 -> B T L 4
                B, TF = future_time.shape
                enc_topk_tmpbbox_unact = enc_topk_bbox_unact[:, None] + enc_topk_vel_unact[:, None] * future_time[..., None, None]
                enc_topk_tmpbbox_unact = enc_topk_tmpbbox_unact.flatten(0, 1)
                enc_topk_bboxes = F.sigmoid(enc_topk_tmpbbox_unact)
                enc_topk_bboxes_list.append(enc_topk_bboxes)
                enc_topk_logits_list.append(torch.repeat_interleave(enc_topk_logits, repeats=TF, dim=0))
                # logger.info(f"decoder input: {inspect(locals())}")
            elif self.type_decoder == "perception":
                # B L 4 -> B T L 4
                B, TF = future_time.shape
                enc_vxt, enc_vyt, enc_log_st = (enc_topk_vel_unact[:, None] * future_time[..., None, None]).unbind(-1)
                enc_topk_tmpbbox_unact_origin = enc_topk_bbox_unact[:, None].tile([1, TF, 1, 1])
                hw = enc_topk_tmpbbox_unact_origin[..., 2:] * torch.exp(enc_log_st[..., None])
                cx = enc_topk_tmpbbox_unact_origin[..., 0] + enc_vxt
                cy = enc_topk_tmpbbox_unact_origin[..., 1] + enc_vyt
                enc_topk_tmpbbox_unact = torch.cat([cx.unsqueeze(-1), cy.unsqueeze(-1), hw], dim=-1).flatten(0, 1)
                enc_topk_bboxes = F.sigmoid(enc_topk_tmpbbox_unact)
                enc_topk_bboxes_list.append(enc_topk_bboxes)
                enc_topk_logits_list.append(torch.repeat_interleave(enc_topk_logits, repeats=TF, dim=0))
                # logger.info(f"decoder input: {inspect(locals())}")
            elif self.type_decoder == 'spin':
                # b t L
                B, TF = future_time.shape
                vx, vy, sx, sy = enc_topk_vel_unact[:, None].unbind(-1)
                # cx, cy, w, h = F.sigmoid(enc_topk_vel_unact[:, None]).tile([1, TF, 1, 1]).unbind(-1)  # 0-1
                cx, cy, w, h = enc_topk_bbox_unact[:, None].tile([1, TF, 1, 1]).unbind(-1)
                t = future_time[..., None]

                # cx1 = cx + (vx + torch.expm1(sx) * (cx - 0.5)) * t
                # cy1 = cy + (vy + torch.expm1(sy) * (cy - 0.5)) * t
                # w1 = w * torch.exp(sx * t)
                # h1 = h * torch.exp(sy * t)
                cx1 = cx + vx * t
                cy1 = cy + vy * t
                w1 = w * torch.exp(sx * t)
                h1 = h * torch.exp(sy * t)

                enc_topk_tmpbbox_unact = torch.stack([cx1, cy1, w1, h1], dim=-1).flatten(0, 1)
                enc_topk_bboxes = F.sigmoid(enc_topk_tmpbbox_unact)
                enc_topk_bboxes_list.append(enc_topk_bboxes)
                enc_topk_logits_list.append(torch.repeat_interleave(enc_topk_logits, repeats=TF, dim=0))
                # logger.info(f"decoder input: {inspect(locals())}")
            elif self.type_decoder == 'double':
                # b t L
                B, TF = future_time.shape
                cx, cy, w, h = enc_topk_bbox_unact[:, None].tile([1, TF, 1, 1]).unbind(-1)
                vx, vy, vw, vh, ax, ay, aw, ah = enc_topk_vel_unact[:, None].unbind(-1)
                t = future_time[..., None]
                t2 = t ** 2

                cx1 = cx + vx * t + ax * t2
                cy1 = cy + vy * t + ay * t2
                w1 = w + vw * t + aw * t2
                h1 = h + vh * t + ah * t2

                enc_topk_tmpbbox_unact = torch.stack([cx1, cy1, w1, h1], dim=-1).flatten(0, 1)
                enc_topk_bboxes = F.sigmoid(enc_topk_tmpbbox_unact)
                enc_topk_bboxes_list.append(enc_topk_bboxes)
                enc_topk_logits_list.append(torch.repeat_interleave(enc_topk_logits, repeats=TF, dim=0))
                # logger.info(f"decoder input: {inspect(locals())}")
            elif self.type_decoder == "default":
                enc_topk_bboxes = F.sigmoid(enc_topk_bbox_unact)
                enc_topk_bboxes_list.append(enc_topk_bboxes)
                enc_topk_logits_list.append(enc_topk_logits)
            else:
                raise ValueError(f"Unknown type_decoder: {self.type_decoder}")

        if self.type_query == "mixed":
            # combining learn query and topk
            num_learnable = int(self.num_queries * 0.3)
            num_current = self.num_queries - num_learnable
            learn_content = self.tgt_embed.weight.unsqueeze(0).tile([memory.shape[0], 1, 1])[:, :num_learnable]
            topk_content = enc_topk_memory.detach()[:, :num_current]
            content = torch.cat([learn_content, topk_content], dim=1)
        elif self.type_query == "default":
            if self.learn_query_content:
                content = self.tgt_embed.weight.unsqueeze(0).tile([memory.shape[0], 1, 1])
            else:
                content = enc_topk_memory.detach()
        else:
            raise ValueError(f"Unknown type_query: {self.type_query}")

        enc_topk_bbox_unact = enc_topk_bbox_unact.detach()

        if denoising_bbox_unact is not None:
            enc_topk_bbox_unact = torch.concat([denoising_bbox_unact, enc_topk_bbox_unact], dim=1)
            content = torch.concat([denoising_logits, content], dim=1)

        return content, enc_topk_bbox_unact, enc_topk_bboxes_list, enc_topk_logits_list

    def _select_topk_ex(
            self,
            memory: torch.Tensor,
            outputs_logits: torch.Tensor,
            outputs_coords_unact: torch.Tensor,
            outputs_vel_unact: torch.Tensor,
            topk: int,
    ):
        if self.query_select_method == 'default':
            _, topk_ind = torch.topk(outputs_logits.max(-1).values, topk, dim=-1)

        elif self.query_select_method == 'one2many':
            _, topk_ind = torch.topk(outputs_logits.flatten(1), topk, dim=-1)
            topk_ind = topk_ind // self.num_classes

        elif self.query_select_method == 'agnostic':
            _, topk_ind = torch.topk(outputs_logits.squeeze(-1), topk, dim=-1)
            _, topk_ind = torch.topk(outputs_logits.squeeze(-1), topk, dim=-1)

        topk_ind: torch.Tensor

        topk_coords = outputs_coords_unact.gather(dim=1, \
                                                  index=topk_ind.unsqueeze(-1).repeat(1, 1,
                                                                                      outputs_coords_unact.shape[-1]))

        topk_logits = outputs_logits.gather(dim=1, \
                                            index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_logits.shape[-1]))

        topk_memory = memory.gather(dim=1, \
                                    index=topk_ind.unsqueeze(-1).repeat(1, 1, memory.shape[-1]))

        topk_vel = outputs_vel_unact.gather(
            dim=1,
            index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_vel_unact.shape[-1])
        )

        return topk_memory, topk_logits, topk_coords, topk_vel

    def forward(self, feats, targets=None, past_time=None, future_time=None):
        # feats: list<nlevel>[B 1 C H W]
        # targets: list<B*TF>[dict<boxes,labels>[nobj 4]]
        # past_time: B TP
        # future_time: B TF
        B, TP = past_time.size()
        _, TF = future_time.size()

        if self.type_decoder == 'default':
            # input projection and embedding
            feats = [f[:, -1].tile([TF, 1, 1, 1]) for f in feats]
            memory, spatial_shapes = self._get_encoder_input(feats)

            # prepare denoising training
            if self.training and self.num_denoising > 0:
                denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = \
                    get_contrastive_denoising_training_group(targets, \
                                                             self.num_classes,
                                                             self.num_queries,
                                                             self.denoising_class_embed,
                                                             num_denoising=self.num_denoising,
                                                             label_noise_ratio=self.label_noise_ratio,
                                                             box_noise_scale=self.box_noise_scale, )
            else:
                denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

            init_ref_contents, init_ref_points_unact, enc_topk_bboxes_list, enc_topk_logits_list = \
                self._get_decoder_input_ex(memory, spatial_shapes, denoising_logits, denoising_bbox_unact, future_time=future_time)
        elif self.type_decoder in ['velocity', 'perception', 'spin', 'double']:
            feats = [f[:, -1] for f in feats]
            memory, spatial_shapes = self._get_encoder_input(feats)

            # prepare denoising training
            if self.training and self.num_denoising > 0:
                denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = \
                    get_contrastive_denoising_training_group(
                        targets,
                        # [t for i, t in enumerate(targets) if i % TF == 0],
                        self.num_classes,
                        self.num_queries,
                        self.denoising_class_embed,
                        num_denoising=self.num_denoising,
                        label_noise_ratio=self.label_noise_ratio,
                        box_noise_scale=self.box_noise_scale,
                    )
            else:
                denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

            if denoising_logits is not None:
                denoising_logits = denoising_logits[::TF]
                denoising_bbox_unact = denoising_bbox_unact[::TF]

            init_ref_contents, init_ref_points_unact, enc_topk_bboxes_list, enc_topk_logits_list = \
                self._get_decoder_input_ex(memory, spatial_shapes, denoising_logits, denoising_bbox_unact, future_time=future_time)
        else:
            raise ValueError(f'Unknown type_decoder: {self.type_decoder}')

        # duplicate
        if self.type_embedding == 'temporal':
            temp_pos_embed = self.temp_pos_head(future_time.reshape(B*TF, 1))[:, None]
        else:
            temp_pos_embed = None

        init_ref_points_unact = init_ref_points_unact.to(dtype=init_ref_contents.dtype)

        # decoder
        # Layers B L (4)
        out_bboxes, out_logits = self.decoder(
            init_ref_contents,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
            future_time=future_time,
            temp_pos_embed=temp_pos_embed,
            velocity_head=self.dec_velocity_head if self.type_decoder in ["velocity", "perception", "spin", 'double'] else None,
        )

        if self.training and dn_meta is not None:
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta['dn_num_split'], dim=2)
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta['dn_num_split'], dim=2)

        out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1]}

        if self.training and self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(out_logits[:-1], out_bboxes[:-1])
            out['enc_aux_outputs'] = self._set_aux_loss(enc_topk_logits_list, enc_topk_bboxes_list)
            out['enc_meta'] = {'class_agnostic': self.query_select_method == 'agnostic'}

            if dn_meta is not None:
                out['dn_aux_outputs'] = self._set_aux_loss(dn_out_logits, dn_out_bboxes)
                out['dn_meta'] = dn_meta

        # logger.info(f"adaptive: {inspect(locals())}")
        return out


