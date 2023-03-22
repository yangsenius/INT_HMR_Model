import torch
import torch.nn as nn
import numpy as np
# , HybridEmbed, PatchEmbed, Block
from lib.models.vision_transformer import VisionTransformer, trunc_normal_, Block
from lib.utils.geometry import rot6d_to_rotmat, rotation_matrix_to_angle_axis
from lib.models.smpl import SMPL, SMPL_MODEL_DIR, SMPL_MEAN_PARAMS
from lib.models.spin import projection


class TokenPoseRot6d(VisionTransformer):
    def __init__(self, img_size=224, joints_num=24, pred_rot_dim=6, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4, qkv_bias=False, qk_scale=None, representation_size=None,
                 drop_rate=0, attn_drop_rate=0, drop_path_rate=0, hybrid_backbone=None,
                 token_init_mode='normal', proj_rot_mode='linear', use_joint2d_head=False,
                 norm_layer=nn.LayerNorm, st_mode='vanilla', contraint_token_delta=False, seq_length=16,
                 use_rot6d_to_token_head=False, mask_ratio=0.,
                 temporal_layers=3, temporal_num_heads=1, 
                 enable_temp_modeling=True, enable_temp_embedding=False):

        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                         num_classes=num_classes, embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                         mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                         representation_size=representation_size, drop_rate=drop_rate,
                         attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                         hybrid_backbone=hybrid_backbone, norm_layer=norm_layer, st_mode=st_mode)

        # joints tokens
        self.joint3d_tokens = nn.Parameter(torch.zeros(1, joints_num, embed_dim))
        self.shape_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cam_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.joints_num = joints_num

        self._init_tokens(mode=token_init_mode)

        self.return_tokens = contraint_token_delta

        self.joint3d_head = nn.Linear(embed_dim, pred_rot_dim)
        self.shape_head = nn.Linear(embed_dim, 10)
        self.cam_head = nn.Linear(embed_dim, 3)

        self.apply(self._init_weights)

        self.smpl = SMPL(
            SMPL_MODEL_DIR,
            create_transl=False,
            create_global_orient=False,
            create_body_pose=False,
            create_betas=False,
        )
        
        self.enable_temp_modeling = False
        self.reconstruct = False
        
        if enable_temp_modeling:
            self.enable_temp_modeling = enable_temp_modeling
            # stochastic depth decay rule
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
            self.temporal_transformer = nn.ModuleList([
                Block(dim=embed_dim, num_heads=temporal_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                    st_mode='vanilla') for i in range(temporal_layers)])
            
            self.enable_pos_embedding = False
            if enable_temp_embedding:
                self.enable_pos_embedding = True
                self.temporal_pos_embedding = nn.Parameter(torch.zeros(1, seq_length, embed_dim))
                trunc_normal_(self.temporal_pos_embedding, std=.02)

            self.mask_ratio = mask_ratio

            if mask_ratio > 0.:
                self.reconstruct = True
                self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        del self.head, self.pre_logits

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        # ascend: small is keep, large is remove
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        # ids_keep = ids_shuffle[:, :len_keep]
        # x_keep = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        x[mask.long(), :] = self.mask_token  # .expand(mask.shape[0], mask.shape[1], -1)
        x_masked = x

        return x_masked

    def _init_tokens(self, mode='normal'):
        if mode == 'normal':
            trunc_normal_(self.joint3d_tokens, std=.02)
            trunc_normal_(self.shape_token, std=.02)
            trunc_normal_(self.cam_token, std=.02)
        else:
            print("zero initialize tokens")
            pass

    def forward_features(self, x, seqlen=1):
        B = x.shape[0]   # (NT, 3, H, W)

        x = self.patch_embed(x)  # (NT, 14*14, 2048)  (bs, seq, embedding_size)

        joint3d_tokens = self.joint3d_tokens.expand(B, -1, -1)
        shape_token = self.shape_token.expand(B, -1, -1)
        cam_token = self.cam_token.expand(B, -1, -1)

        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embed[:, :, :]

        x = torch.cat([joint3d_tokens, shape_token, cam_token, x], dim=1)   
        # [NT, HW+24+1+1， embedding_size]

        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x, seqlen)

        x = self.norm(x)
        joint3d_tokens = x[:, :self.joints_num]
        shape_token = x[:, self.joints_num]
        cam_token = x[:, self.joints_num + 1]

        return joint3d_tokens, shape_token, cam_token

    def temporal_modeling(self, joint3d_tokens, seq_length, mask_ratio=0.):
        # joint3d_tokens [B, N, C]
        B, N, C = joint3d_tokens.shape
        joint3d_tokens = joint3d_tokens.reshape(-1,seq_length, N, C).permute(0, 2, 1, 3)
        joint3d_tokens_temporal = joint3d_tokens.reshape(-1, seq_length, C)
        
        # [bs*N, seq_length, C]
        if self.enable_pos_embedding:
            if self.temporal_pos_embedding.shape[1] !=seq_length:
                
                temporal_pos_embedding = torch.nn.functional.interpolate(
                    self.temporal_pos_embedding.data.permute(0,2,1),
                    size=seq_length,
                    mode='linear'
                ).permute(0, 2, 1)
                self.temporal_pos_embedding = torch.nn.Parameter(temporal_pos_embedding)
            joint3d_tokens_temporal += self.temporal_pos_embedding[:, :seq_length, :]

        if self.training and mask_ratio > 0.:
            joint3d_tokens_temporal_masked = self.random_masking(
                joint3d_tokens_temporal, mask_ratio)
        else:
            joint3d_tokens_temporal_masked = joint3d_tokens_temporal

        for blk in self.temporal_transformer:
            joint3d_tokens_temporal_masked = blk(joint3d_tokens_temporal_masked)

        pred_joint3d_tokens_temporal = joint3d_tokens_temporal_masked.reshape(
            -1, N, seq_length, C).permute(0, 2, 1, 3)
        pred_joint3d_tokens_temporal = pred_joint3d_tokens_temporal.reshape(B, N, C)
        return pred_joint3d_tokens_temporal

    def forward(self, x, J_regressor=None, **kwargs):

        batch_size, seqlen = x.shape[:2]
        x = x.reshape(-1, x.shape[-3], x.shape[-2], x.shape[-1])  # (NT, 3, H, W)

        joint3d_tokens, shape_token, cam_token = self.forward_features(x, seqlen)
        
        if self.enable_temp_modeling:
            joint3d_tokens_before = joint3d_tokens.clone().detach_()
            joint3d_tokens = self.temporal_modeling(
                joint3d_tokens, seqlen, mask_ratio=self.mask_ratio)

        # # [bs*seq_length, N, embed_dim]
        pred_joints_rot6d = self.joint3d_head(joint3d_tokens)  # [b, 24, 6]
        pred_shape = self.shape_head(shape_token)
        pred_cam = self.cam_head(cam_token)

        output = {}

        # mse loss
        if self.reconstruct and self.training:
            reconstruct_loss = (joint3d_tokens - joint3d_tokens_before)**2
            reconstruct_loss = reconstruct_loss.mean()
            output['reconstruct_loss'] = reconstruct_loss

        nt = pred_joints_rot6d.shape[0]
        pred_rotmat = rot6d_to_rotmat(pred_joints_rot6d).reshape(nt, -1, 3, 3)

        pred_output = self.smpl(
            betas=pred_shape,
            body_pose=pred_rotmat[:, 1:],
            global_orient=pred_rotmat[:, 0].unsqueeze(1),
            pose2rot=False
        )

        pred_vertices = pred_output.vertices[:nt]
        pred_joints = pred_output.joints[:nt]

        if J_regressor is not None:
            J_regressor_batch = J_regressor[None, :].expand(
                pred_vertices.shape[0], -1, -1).to(pred_vertices.device)
            pred_joints = torch.matmul(J_regressor_batch, pred_vertices)

        pred_keypoints_2d = projection(pred_joints, pred_cam)

        pose = rotation_matrix_to_angle_axis(
            pred_rotmat.reshape(-1, 3, 3)).reshape(nt, -1)

        output['theta'] = torch.cat([pred_cam, pose, pred_shape], dim=1)
        output['verts'] = pred_vertices
        output['kp_2d'] = pred_keypoints_2d
        output['kp_3d'] = pred_joints
        output['rotmat'] = pred_rotmat

        output['theta'] = output['theta'].reshape(batch_size, seqlen, -1)
        output['verts'] = output['verts'].reshape(batch_size, seqlen, -1, 3)
        output['kp_2d'] = output['kp_2d'].reshape(batch_size, seqlen, -1, 2)
        output['kp_3d'] = output['kp_3d'].reshape(batch_size, seqlen, -1, 3)
        output['rotmat'] = output['rotmat'].reshape(batch_size, seqlen, -1, 3, 3)

        return output


from lib.models.resnetv2 import ResNetV2
import torch.utils.model_zoo as model_zoo
from lib.models.vision_transformer import _conv_filter, model_urls
from functools import partial


def Token3d(num_blocks, num_heads, st_mode, pretrained=True, proj_rot_mode='linear',
            use_joint2d_head=False, contraint_token_delta=False,
            use_rot6d_to_token_head=False, mask_ratio=0.,
            temporal_layers=3, temporal_num_heads=1,
            enable_temp_modeling=True, enable_temp_embedding=False,
            **kwargs):
    """ Hybrid model with a R50 and a Vit of custom layers .
    """
    # create a ResNetV2 w/o pre-activation, that uses StdConv and GroupNorm and has 3 stages, no head
    backbone = ResNetV2(
        layers=(3, 4, 9), num_classes=0, global_pool='', in_chans=kwargs.get('in_chans', 3),
        preact=False, stem_type='same')
    model = TokenPoseRot6d(
        patch_size=16, embed_dim=768, depth=num_blocks, num_heads=num_heads,
        hybrid_backbone=backbone, mlp_ratio=4, qkv_bias=True,
        representation_size=768, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        st_mode=st_mode, proj_rot_mode=proj_rot_mode,
        use_joint2d_head=use_joint2d_head,
        contraint_token_delta=contraint_token_delta,
        use_rot6d_to_token_head=use_rot6d_to_token_head,
        mask_ratio=mask_ratio,
        temporal_layers=temporal_layers,
        temporal_num_heads=temporal_num_heads,
        enable_temp_modeling=enable_temp_modeling, 
        enable_temp_embedding=enable_temp_embedding,
        **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url(
            model_urls['vit_base_resnet50_224_in21k'], progress=False, map_location='cpu')
        state_dict = _conv_filter(state_dict)
        del state_dict['head.weight']
        del state_dict['head.bias']
        model.load_state_dict(state_dict, strict=False)
    return model