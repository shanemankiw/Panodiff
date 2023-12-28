import einops
import torch
import torch as th
import torch.nn as nn
from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)

from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, \
    AttentionBlock
from ldm.models.diffusion.ddpm import LatentDiffusion, LatentInpaintDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config, default
from ldm.models.diffusion.ddim import DDIMSampler, PanoSampler
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
import importlib
from RotationModels.utils.compute_utils import *
from RotationModels.utils.loss_utils import *
from RotationModels.utils.pano_utils import sample_from_patch, deform_a_little
# for debug
import cv2
# for debug
import cv2


class ControlledUnetModel(UNetModel):
    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)

        if control is not None:
            h += control.pop()

        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.out(h)


class ControlNet(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            hint_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,  # custom transformer support
            transformer_depth=1,  # custom transformer support
            context_dim=None,  # custom transformer support
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=True,
            disable_self_attentions=None,
            num_attention_blocks=None,
            disable_middle_self_attn=False,
            use_linear_in_transformer=False,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(
                map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
        )

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, hint, timesteps, context, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        guided_hint = self.input_hint_block(hint, emb, context)

        outs = []

        h = x.type(self.dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, emb, context)
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return outs


class RotaNet(nn.Module):
    def __init__(self, **args):
        super(RotaNet, self).__init__()
        self.args = args

        encoder_lib = importlib.import_module(args["encoder"]["type"])
        self.encoder = encoder_lib.ImageEncoder(args["encoder"])
        # print("Encoder:")
        # print(self.encoder)

        dn_lib = importlib.import_module(args["rotationnet"]["type"])
        self.rotation_net = dn_lib.RotationNet(args["rotationnet"])
        # print("rotationnet:")
        # print(self.rotation_net)

        dn_lib_y = importlib.import_module(args["rotationnet_y"]["type"])
        self.rotation_net_y = dn_lib_y.RotationNet(args["rotationnet_y"])
        # print("rotationnet_y:")
        # print(self.rotation_net_y)

        dn_lib_z = importlib.import_module(args["rotationnet_z"]["type"])
        self.rotation_net_z = dn_lib_z.RotationNet(args["rotationnet_z"])
        # print("rotationnet_z:")
        # print(self.rotation_net_z)

        self.classification = args["classification"]
        self.pairwise_type = args["pairwise_type"]
        self.rotation_parameterization = args["rotation_parameterization"]

    def forward(self, data_full):
        img1 = data_full['img1']
        img2 = data_full['img2']

        image_feature_map1 = self.encoder(img1)
        image_feature_map2 = self.encoder(img2)

        # pairwise operation
        if self.pairwise_type == "concat":
            pairwise_feature = torch.cat([image_feature_map1, image_feature_map2], dim=1)
        elif self.pairwise_type == "cost_volume":
            pairwise_feature = compute_correlation_volume_pairwise(image_feature_map1, image_feature_map2, num_levels=1)
        elif self.pairwise_type == "correlation_volume":
            pairwise_feature = compute_correlation_volume_pairwise(image_feature_map1, image_feature_map2, num_levels=4)

        output = {}
        if not self.classification:
            out_rmat, _out_rotation = self.rotation_net(pairwise_feature)
            output["rmat"] = out_rmat
        else:
            _, out_rotation_x = self.rotation_net(pairwise_feature)
            _, out_rotation_y = self.rotation_net_y(pairwise_feature)
            _, out_rotation_z = self.rotation_net_z(pairwise_feature)
            output["rot_x"] = out_rotation_x
            output["rot_y"] = out_rotation_y
            output["rot_z"] = out_rotation_z

        return output

    def compute_loss(self, data_full, output):
        rotation_x1 = data_full['rotation_x1']
        rotation_y1 = data_full['rotation_y1']
        rotation_x2 = data_full['rotation_x2']
        rotation_y2 = data_full['rotation_y2']

        batch_size = data_full['img1'].size(0)
        gt_rmat = compute_gt_rmat(rotation_x1, rotation_y1, rotation_x2, rotation_y2, batch_size)
        if self.rotation_parameterization:
            angle_x, angle_y, angle_z = compute_angle(rotation_x1, rotation_x2, rotation_y1, rotation_y2)
        else:
            angle_x, angle_y, angle_z = compute_euler_angles_from_rotation_matrices(gt_rmat)

        # loss type
        if not self.classification:
            # regression loss
            res1 = rotation_loss_reg(output["rmat"], gt_rmat)
            loss = res1['loss']
        else:
            # classification loss
            _, rotation_x = torch.topk(output["rot_x"], 1, dim=-1)
            _, rotation_y = torch.topk(output["rot_y"], 1, dim=-1)
            _, rotation_z = torch.topk(output["rot_z"], 1, dim=-1)
            loss_x = rotation_loss_class(output["rot_x"], angle_x)
            loss_y = rotation_loss_class(output["rot_y"], angle_y)
            loss_z = rotation_loss_class(output["rot_z"], angle_z)

            loss = loss_x + loss_y + loss_z

        return loss


class ControlLDM(LatentDiffusion):

    def __init__(self, control_stage_config, control_key, only_mid_control, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        control = batch[self.control_key]
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)
        control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()
        return x, dict(c_crossattn=[c], c_concat=[control])

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond['c_crossattn'], 1)

        if cond['c_concat'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None,
                                  only_mid_control=self.only_mid_control)
        else:
            control = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond_txt)
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control,
                                  only_mid_control=self.only_mid_control)

        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        c_cat, c = c["c_concat"][0][:N], c["c_crossattn"][0][:N]
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z)
        log["control"] = c_cat * 2.0 - 1.0
        log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N)
            uc_cat = c_cat  # torch.zeros_like(c_cat)
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
            samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             )
            # trick: make ends meet

            # trick: make ends meet
            
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()


class ControlInpaintLDM(LatentDiffusion):
    def __init__(
            self,
            control_stage_config,
            concat_keys=("mask", "hint"),
            masked_image_key="hint",
            control_key=None,
            only_mid_control=False,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.masked_image_key = masked_image_key
        assert self.masked_image_key in concat_keys
        self.concat_keys = concat_keys

        # ControlNet part
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13

    @torch.no_grad()
    def get_input(
            self, batch, k, bs=None, *args, **kwargs):

        # note: restricted to non-trainable encoders currently
        assert (
            not self.cond_stage_trainable
        ), "trainable cond stages not yet supported for inpainting"
        # NOTE(wjh)
        # This x, z is actually [z, c]. see get_input function in LatentDiffusion.
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        # control = batch[self.control_key]

        assert exists(self.concat_keys)
        c_cat = list()
        for ck in self.concat_keys:
            cc = (
                rearrange(batch[ck], "b h w c -> b c h w")
                .to(memory_format=torch.contiguous_format)
                .float()
            )
            if bs is not None:
                cc = cc[:bs]
                cc = cc.to(self.device)
            '''bchw = z.shape
            if ck == self.masked_image_key:
                cc = self.get_first_stage_encoding(self.encode_first_stage(cc))
            else:
                cc = torch.nn.functional.interpolate(cc, size=bchw[-2:])'''
            c_cat.append(cc)
        c_cat = torch.cat(c_cat, dim=1)
        # NOTE(wjh): c_cat is the control latent, and c_crossattn is just the
        all_conds = {"c_concat": [c_cat], "c_crossattn": [c]}

        return x, all_conds

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond['c_crossattn'], 1)

        if cond['c_concat'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None,
                                  only_mid_control=self.only_mid_control)
        else:
            control = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond_txt)
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            '''# HACK(wjh) adding mask into propagating? 
            mask = torch.cat(cond['c_concat'], 1)[:, :1]'''
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control,
                                  only_mid_control=self.only_mid_control)

        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        c_cat, c = c["c_concat"][0][:N], c["c_crossattn"][0][:N]
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z)
        log["control"] = c_cat[:, -3:] * 2.0 - 1.0  # NOTE(wjh): only output rgb layers.
        _mask = c_cat[:, :1]
        log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            # NOTE(wjh):
            # used in condition guided sample:
            # model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)

            uc_cross = self.get_unconditional_conditioning(N)
            uc_cat = c_cat  # torch.zeros_like(c_cat)
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
            samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg * (1 - _mask) + log[
                "control"] * _mask

        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        # NOTE(wjh)
        # sample the whole 50 steps, from 0 to 50
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()


class ControlInpaintLDM_ft(LatentInpaintDiffusion):

    def __init__(
            self,
            control_stage_config,
            concat_keys=("mask", "hint"),
            masked_image_key="hint",
            control_key=None,
            only_mid_control=False,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.masked_image_key = masked_image_key
        assert self.masked_image_key in concat_keys
        self.concat_keys = concat_keys

        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13

    @torch.no_grad()
    def get_input(
            self, batch, k, bs=None, return_first_stage_outputs=False, *args, **kwargs):

        # note: restricted to non-trainable encoders currently
        assert (
            not self.cond_stage_trainable
        ), "trainable cond stages not yet supported for inpainting"
        # NOTE(wjh)
        z, all_conds, x, xrec, xc = super().get_input(
            batch,
            self.first_stage_key,
            return_first_stage_outputs=True,
        )

        assert exists(self.concat_keys)
        c_control = list()
        for ck in self.concat_keys:
            cc = (
                rearrange(batch[ck], "b h w c -> b c h w")
                .to(memory_format=torch.contiguous_format)
                .float()
            )
            if bs is not None:
                cc = cc[:bs]
                cc = cc.to(self.device)
            c_control.append(cc)
        c_control = torch.cat(c_control, dim=1)

        all_conds['control_input'] = [c_control]

        if return_first_stage_outputs:
            return z, all_conds, x, xrec, xc

        return z, all_conds

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):

        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond['c_crossattn'], 1)
        xc = torch.cat([x_noisy] + cond['c_concat'], dim=1)

        if 'control_input' not in cond:
            eps = diffusion_model(x=xc, timesteps=t, context=cond_txt, control=None,
                                  only_mid_control=self.only_mid_control)
        else:
            control = self.control_model(x=xc, hint=torch.cat(cond['control_input'], 1), timesteps=t, context=cond_txt)
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            eps = diffusion_model(x=xc, timesteps=t, context=cond_txt, control=control,
                                  only_mid_control=self.only_mid_control)

        return eps

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["control_input"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)

        c_cat, c_control, c = c["c_concat"][0][:N], c["control_input"][0][:N], c["c_crossattn"][0][:N]
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z)
        log["control"] = c_control[:, -3:] * 2.0 - 1.0  # NOTE(wjh): only output rgb layers.
        _mask = c_control[:, :1]
        log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            samples, z_denoise_row = self.sample_log(
                cond={"c_concat": [c_cat], "control_input": [c_control], "c_crossattn": [c]},
                batch_size=N, ddim=use_ddim,
                ddim_steps=ddim_steps, eta=ddim_eta, mask=_mask)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            # NOTE(wjh):
            # used in condition guided sample:
            # model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)

            uc_cross = self.get_unconditional_conditioning(N)
            uc_control = c_control  # torch.zeros_like(c_cat)
            uc_full = {"control_input": [uc_control], "c_crossattn": [uc_cross], "c_concat": [c_cat]}
            # uc_full = {"c_crossattn": [uc_cross], "c_concat": [c_cat]}
            samples_cfg, _ = self.sample_log(
                # cond={"c_concat": [c_cat], "c_crossattn": [c]},
                cond={"c_concat": [c_cat], "c_crossattn": [c], "control_input": [c_control]},
                batch_size=N, ddim=use_ddim,
                ddim_steps=ddim_steps, eta=ddim_eta,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=uc_full,
                # mask=c_cat[:, :1],
                # x0=z[:N]
            )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            # x_samples_cfg = torch.clip(x_samples_cfg, min=-1, max=1)
            log[
                f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg  # x_samples_cfg *  _mask + log["control"] * (1 - _mask)
            # log["pred_x0"] = self.decode_first_stage(_['pred_x0'][0])

        return log

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()


class Rota_Inpaint(LatentInpaintDiffusion):
    def __init__(
            self,
            control_stage_config,
            rotation_stage_config,
            concat_keys=("mask", "hint"),
            masked_image_key="hint",
            control_key=None,
            only_mid_control=False,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.masked_image_key = masked_image_key
        assert self.masked_image_key in concat_keys
        self.concat_keys = concat_keys

        self.control_model = instantiate_from_config(control_stage_config)
        self.rotation_model = instantiate_from_config(rotation_stage_config)
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13
        self.learning_rate_rotation = rotation_stage_config.params.lr

    # @torch.no_grad()
    def get_input(
            self, batch, k, bs=None, return_first_stage_outputs=False, *args, **kwargs):

        if 'hint' not in batch:  # In test, we don't need to sample this everytime
            rot_output = self.rotation_model(batch)
            # HACK
            if self.rotation_supervise:
                batch['loss_rot'] = self.rotation_model.compute_loss(batch, rot_output)

            # integrate rotation into this.
            weighted_sum = lambda weights, indices: (F.softmax(weights, dim=1) * indices).sum(dim=1)
            # clip_angle = lambda angle: angle - 360 if angle > 180 else angle

            weights, rotation_x = torch.topk(rot_output["rot_x"], 16, dim=-1)
            angle_x = weighted_sum(weights, rotation_x) - 180
            angle_x[torch.where(angle_x > 180)] -= 360.0
            angle_x[torch.where(angle_x < -180)] += 360.0

            weights, rotation_y = torch.topk(rot_output["rot_y"], 16, dim=-1)
            angle_y1 = weighted_sum(weights, rotation_y) - 180
            angle_y1[torch.where(angle_y1 > 180)] -= 360.0
            angle_y1[torch.where(angle_y1 < -180)] += 360.0

            weights, rotation_z = torch.topk(rot_output["rot_z"], 16, dim=-1)
            angle_y2 = weighted_sum(weights, rotation_z) - 180
            angle_y2[torch.where(angle_y2 > 180)] -= 360.0
            angle_y2[torch.where(angle_y2 < -180)] += 360.0
            # Done (wjh)
            # 1[Done] We need to rotate the groundtruth input to align with the first image?
            # 2[Done] The output from RotationNet seems wrong. Think of a new way to do this.
            # 3[Done] adjust sample from patch to batchsize x 3 x height x width

            # patch1, patch2 = inverse_normalize(batch['img1']), inverse_normalize(batch['img2'])
            # patch1, patch2 = batch['patch1'].permute([0, 3, 1, 2]), batch['patch2'].permute([0, 3, 1, 2])
            if self.down_scale == 2:
                patch1, patch2 = inverse_normalize(batch['img1']), inverse_normalize(batch['img2'])
            elif self.down_scale == 1:
                patch1, patch2 = batch['img1_original'].permute(0, 3, 1, 2), batch['img2_original'].permute(0, 3, 1, 2)
            else:
                raise ValueError

            if self.use_pred_rots:
                # fix gt
                pano1, mask1 = sample_from_patch(patch1,
                                                 theta=torch.zeros_like(angle_x),
                                                 phi=angle_y1,

                                                 pano_height=batch['pano'].shape[1],
                                                 pano_width=batch['pano'].shape[2])

                pano2, mask2 = sample_from_patch(patch2,
                                                 theta=-angle_x,
                                                 phi=angle_y2,
                                                 pano_height=batch['pano'].shape[1],
                                                 pano_width=batch['pano'].shape[2])
            else:
                if self.use_gt_rots:
                    # fix image 1
                    '''pano1, mask1 = sample_from_patch(patch1, 
                                                    theta=torch.zeros_like(angle_x),
                                                    phi=(batch['rotation_y1']*180/torch.pi).to(angle_x.dtype),#angle_y1,
                                                    pano_height=batch['pano'].shape[1], 
                                                    pano_width=batch['pano'].shape[2])
    
                    delta_theta_gt = ((batch['rotation_x2'] - batch['rotation_x1']) * 180 / torch.pi).to(angle_x.dtype)
                    delta_theta_gt[torch.where(delta_theta_gt>180)] -= 360.0
                    delta_theta_gt[torch.where(delta_theta_gt<-180)] += 360.0
    
                    pano2, mask2 = sample_from_patch(patch2, 
                                                    theta=-(delta_theta_gt),
                                                    phi=(batch['rotation_y2']*180/torch.pi).to(angle_x.dtype),
                                                    pano_height=batch['pano'].shape[1], 
                                                    pano_width=batch['pano'].shape[2])'''
                    # fix gt
                    pano1, mask1 = sample_from_patch(patch1,
                                                     theta=-(batch['rotation_x1'] * 180 / torch.pi).to(angle_x.dtype),
                                                     phi=(batch['rotation_y1'] * 180 / torch.pi).to(angle_x.dtype),
                                                     # angle_y1,
                                                     pano_height=batch['pano'].shape[1],
                                                     pano_width=batch['pano'].shape[2])

                    delta_theta_gt = ((batch['rotation_x2'] - batch['rotation_x1']) * 180 / torch.pi).to(angle_x.dtype)
                    delta_theta_gt[torch.where(delta_theta_gt > 180)] -= 360.0
                    delta_theta_gt[torch.where(delta_theta_gt < -180)] += 360.0

                    pano2, mask2 = sample_from_patch(patch2,
                                                     theta=-(batch['rotation_x2'] * 180 / torch.pi).to(angle_x.dtype),
                                                     phi=(batch['rotation_y2'] * 180 / torch.pi).to(angle_x.dtype),
                                                     pano_height=batch['pano'].shape[1],
                                                     pano_width=batch['pano'].shape[2])
                else:
                    pano1, mask1 = sample_from_patch(patch1,
                                                     theta=torch.zeros_like(angle_x),
                                                     phi=(batch['rotation_y1'] * 180 / torch.pi).to(angle_x.dtype),
                                                     pano_height=batch['pano'].shape[1],
                                                     pano_width=batch['pano'].shape[2])

                    delta_theta_gt = ((batch['rotation_x2'] - batch['rotation_x1']) * 180 / torch.pi).to(angle_x.dtype)
                    delta_theta_gt[torch.where(delta_theta_gt > 180)] -= 360.0
                    delta_theta_gt[torch.where(delta_theta_gt < -180)] += 360.0

                    pano2, mask2 = sample_from_patch(patch2,
                                                     theta=-delta_theta_gt,
                                                     phi=(batch['rotation_y2'] * 180 / torch.pi).to(angle_x.dtype),
                                                     pano_height=batch['pano'].shape[1],
                                                     pano_width=batch['pano'].shape[2])
            # pano is of shape B3HW, mask is of shape B1HW

            # shift the ground truth image laterally
            batch_size = batch['pano'].shape[0]
            y_grid, x_grid = torch.meshgrid(
                torch.linspace(-1, 1, batch['pano'].shape[1]),
                torch.linspace(-1, 1, batch['pano'].shape[2]))
            y_grid = y_grid[None, :].repeat([batch_size, 1, 1]).to(batch['pano'].device)
            x_grid = x_grid[None, :].repeat([batch_size, 1, 1]).to(batch['pano'].device)

            if self.use_gt_rots:
                x_grid = x_grid  # + (batch['rotation_x1'] / torch.pi).reshape([-1, 1, 1])
            else:
                x_grid = x_grid + (batch['rotation_x1'] / torch.pi).reshape([-1, 1, 1])
            x_grid[x_grid > 1] -= 2
            x_grid[x_grid < -1] += 2
            # Stack the x and y grids along the channel dimension
            x_grid = torch.tensor(x_grid, dtype=batch['pano'].dtype, device=batch['pano'].device)
            y_grid = torch.tensor(y_grid, dtype=batch['pano'].dtype, device=batch['pano'].device)
            grid = torch.stack([x_grid, y_grid], dim=3)  # BHW2

            batch['jpg'] = F.grid_sample(batch['pano'].permute([0, 3, 1, 2]), grid, align_corners=True).permute(
                [0, 2, 3, 1])

            '''# DEBUG
            import cv2
            cv2.imwrite('x_shifted.jpg', (batch['jpg'].cpu().numpy()[0]*255).astype(np.uint8))
            aligned = F.grid_sample(batch['pano'].permute([0, 3, 1, 2]), grid, align_corners=True).permute([0,2,3,1])
            cv2.imwrite('x_shifted_aligncorners.jpg', (aligned.cpu().numpy()[0]*255).astype(np.uint8))'''
            '''phis_pertubate = torch.linspace(0, 2*torch.pi, 50, 
                                        device=batch['jpg'].device, dtype=batch['jpg'].dtype)
            phis_pertubate = torch.sin(phis_pertubate) * 30 # the range of phi turbulance is 30
            phis_steps = torch.zeros_like(phis_pertubate)
            phis_steps[1:] = phis_pertubate[1:] - phis_pertubate[:-1]
            phis_steps = phis_steps[:,None].repeat([1, 4])
            delta_step = torch.ones_like(phis_steps[0])*360/50
            deformed_img = deform_a_little(img=batch['jpg'].permute([0,3,1,2]),
                                  delta_theta=delta_step,
                                  delta_phi=phis_steps[0],
                                  pano_height=batch['jpg'].shape[1],
                                  pano_width=batch['jpg'].shape[2]).permute([0,2,3,1])
            import cv2
            cv2.imwrite('d_img.png', cv2.cvtColor(deformed_img[0].cpu().numpy()*255, cv2.COLOR_RGB2BGR))
            cv2.imwrite('o_img.png', cv2.cvtColor(batch['jpg'][0].cpu().numpy()*255, cv2.COLOR_RGB2BGR))'''
            # normalize, as in controlNet dataset
            batch['jpg'] = batch['jpg'] * 2 - 1.0

            # merge these 2 panos and masks together
            batch['mask'] = torch.bitwise_or(mask1, mask2)

            batch['hint'] = torch.zeros_like(batch['jpg'])
            batch['hint'][batch['mask']] = pano1[batch['mask']]
            batch['hint'][(~mask1) * mask2] = pano2[(~mask1) * mask2]  # set pano1 as main, mask2 as help.
            batch['mask'] = batch['mask'][..., None]
            batch['mask'] = torch.where(batch['mask'] == 0, 1, 0)

        with torch.no_grad():
            '''
            TODO:
            Rota_Inpaint <- LatentInpaintDiffusion <- LatentDiffusion
            What did get input do in every inheritance?
            1. Latent Diffusion
            conditionining keys
            2. Latent Inpaint Diffusion
            concat_keys
            3. Rota_Inpaint

            '''
            assert (
                not self.cond_stage_trainable
            ), "trainable cond stages not yet supported for inpainting"
            # NOTE(wjh)
            z, all_conds, x, xrec, xc = super().get_input(
                batch,
                self.first_stage_key,
                return_first_stage_outputs=True,
            )

            assert exists(self.concat_keys)
            c_control = list()
            for ck in self.concat_keys:
                cc = (
                    rearrange(batch[ck], "b h w c -> b c h w")
                    .to(memory_format=torch.contiguous_format)
                    .float()
                )
                if bs is not None:
                    cc = cc[:bs]
                    cc = cc.to(self.device)
                c_control.append(cc)
            c_control = torch.cat(c_control, dim=1)

            all_conds['control_input'] = [c_control]

            if return_first_stage_outputs:
                return z, all_conds, x, xrec, xc

            return z, all_conds

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        '''
        NOTE (wjh)
        It seems like we do not need to modify this part.
        '''

        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond['c_crossattn'], 1)
        xc = torch.cat([x_noisy] + cond['c_concat'], dim=1)

        if 'control_input' not in cond:
            eps = diffusion_model(x=xc, timesteps=t, context=cond_txt, control=None,
                                  only_mid_control=self.only_mid_control)
        else:
            control = self.control_model(x=xc, hint=torch.cat(cond['control_input'], 1), timesteps=t, context=cond_txt)
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            eps = diffusion_model(x=xc, timesteps=t, context=cond_txt, control=control,
                                  only_mid_control=self.only_mid_control)

        return eps

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = PanoSampler(self)
        b, c, h, w = cond["control_input"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @staticmethod
    def rolling_padding(tensor_to_pad: torch.Tensor, half_pad_length: int):

        left_pad = tensor_to_pad[..., -half_pad_length:]
        right_pad = tensor_to_pad[..., :half_pad_length]

        return torch.cat([left_pad, tensor_to_pad, right_pad], dim=-1)

    @staticmethod
    def rolling_padding(tensor_to_pad:torch.Tensor, half_pad_length:int):

        left_pad = tensor_to_pad[...,-half_pad_length:]
        right_pad = tensor_to_pad[...,:half_pad_length]
        
        return torch.cat([left_pad, tensor_to_pad, right_pad], dim=-1)
    
    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)

        c_cat, c_control, c = c["c_concat"][0][:N], c["control_input"][0][:N], c["c_crossattn"][0][:N]
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z)
        
        # DEBUG: latent reconstruction
        '''import cv2
        test_img = cv2.cvtColor(cv2.imread('datasets/rotation_blip_dataset_train/raw_crops/undist/00565/panorama.jpg'),
                                cv2.COLOR_BGR2RGB)
        input_tensor = torch.from_numpy(test_img)/127.5 - 1.0 # -1 ~ 1
        input_tensor = input_tensor[None, :].permute([0, 3, 1, 2]).to(self.device)
        rolled_input_tensor = torch.roll(input_tensor,
                                         shifts=input_tensor.shape[-1]//2,
                                         dims=-1)

        # get encoded z
        encoder_posterior = self.encode_first_stage(input_tensor)
        z = self.get_first_stage_encoding(encoder_posterior).detach()
        rolled_encoder_posterior = self.encode_first_stage(rolled_input_tensor)
        rolled_z = self.get_first_stage_encoding(rolled_encoder_posterior).detach()

        # operate on latent space
        z_latent_roll = torch.roll(z, shifts=z.shape[-1]//2, dims=-1)

        recon = self.decode_first_stage(z)[0]
        rolled_recon = self.decode_first_stage(rolled_z)[0]
        recon_latent_roll = self.decode_first_stage(z_latent_roll)[0]

        # roll them to see the left and right
        recon = torch.roll(recon, shifts=recon.shape[-1]//2, dims=-1)
        rolled_recon = torch.roll(rolled_recon, shifts=rolled_recon.shape[-1]//2, dims=-1)

        # test if the trick works
        recon = (recon/2+0.5).permute([1,2,0]).cpu().numpy()
        rolled_recon = (rolled_recon/2+0.5).permute([1,2,0]).cpu().numpy()
        recon_latent_roll = (recon_latent_roll/2+0.5).permute([1,2,0]).cpu().numpy()
        # 
        recon_mixed = recon.copy()
        recon_mixed[:,recon.shape[1]//4:recon.shape[1]//4*3] = recon_latent_roll[:,recon.shape[1]//4:recon.shape[1]//4*3]
        #gt = (batch['jpg'][0]/2 + 0.5).cpu().numpy()
        cv2.imwrite('recon.png', cv2.cvtColor(recon*255, cv2.COLOR_RGB2BGR))
        cv2.imwrite('recon_latent_roll.png', cv2.cvtColor(recon_latent_roll*255, cv2.COLOR_RGB2BGR))
        cv2.imwrite('rolled_recon.png', cv2.cvtColor(rolled_recon*255, cv2.COLOR_RGB2BGR))
        cv2.imwrite('recon_latent_roll_mixed.png', cv2.cvtColor(recon_mixed*255, cv2.COLOR_RGB2BGR))'''

        log["control"] = c_control[:, -3:] * 2.0 - 1.0  # NOTE(wjh): only output rgb layers.
        _mask = c_control[:, :1]
        log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            samples, z_denoise_row = self.sample_log(
                cond={"c_concat": [c_cat], "control_input": [c_control], "c_crossattn": [c]},
                batch_size=N, ddim=use_ddim,
                ddim_steps=ddim_steps, eta=ddim_eta, mask=_mask)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            # NOTE(wjh):
            # used in condition guided sample:
            # model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)

            uc_cross = self.get_unconditional_conditioning(N)
            uc_control = c_control  # torch.zeros_like(c_cat)
            uc_full = {"control_input": [uc_control], "c_crossattn": [uc_cross], "c_concat": [c_cat]}
            # uc_full = {"c_crossattn": [uc_cross], "c_concat": [c_cat]}
            samples_cfg, _ = self.sample_log(
                # cond={"c_concat": [c_cat], "c_crossattn": [c]},
                cond={"c_concat": [c_cat], "c_crossattn": [c], "control_input": [c_control]},
                batch_size=N, ddim=use_ddim,
                ddim_steps=ddim_steps, eta=ddim_eta,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=uc_full,
                # mask=c_cat[:, :1],
                # x0=z[:N]
            )
            # trick, make the ends meet
            if self.padding_augment:
                pad_length = samples_cfg.shape[-1] // 8
                samples_cfg = self.rolling_padding(samples_cfg, pad_length)
            
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            if self.padding_augment:
                pad_length_hr = pad_length * 8
                x_samples_cfg = x_samples_cfg[..., pad_length_hr:-pad_length_hr]
            # x_samples_cfg = torch.clip(x_samples_cfg, min=-1, max=1)
            log[
                f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg  # x_samples_cfg *  _mask + log["control"] * (1 - _mask)
            # log["pred_x0"] = self.decode_first_stage(_['pred_x0'][0])

        return log

    def configure_optimizers(self):
        '''
        NOTE(wjh): different learning rates for rotation and control model
        '''
        lr_control = self.learning_rate
        lr_rotation = self.learning_rate_rotation

        control_params = list(self.control_model.parameters())
        rotation_params = list(self.rotation_model.parameters())
        if not self.sd_locked:
            control_params += list(self.model.diffusion_model.output_blocks.parameters())
            control_params += list(self.model.diffusion_model.out.parameters())
            # TODO(wjh)
            # How about if we want to finetune a bit on the decoder, we can add something here.
            # self.first_stage_model.decoder

        control_opt = torch.optim.AdamW(control_params, lr=lr_control)
        rotation_opt = torch.optim.AdamW(rotation_params, lr=lr_rotation)

        return [control_opt, rotation_opt]

    def training_step(self, batch, batch_idx, optimizer_idx):
        for k in self.ucg_training:
            p = self.ucg_training[k]["p"]
            val = self.ucg_training[k]["val"]
            if val is None:
                val = ""
            for i in range(len(batch[k])):
                if self.ucg_prng.choice(2, p=[1 - p, p]):
                    batch[k][i] = val

        loss, loss_dict = self.shared_step(batch)
        if optimizer_idx == 1 and self.rotation_supervise:
            # Add rotation loss
            loss += batch['loss_rot'].mean() * self.rotation_loss_lambda
            loss_dict.update({'train/loss_rot': batch['loss_rot'].mean() * self.rotation_loss_lambda})
        if optimizer_idx==1 and self.rotation_supervise:
            # Add rotation loss
            loss += batch['loss_rot'].mean() * self.rotation_loss_lambda
            loss_dict.update({'train/loss_rot': batch['loss_rot'].mean() * self.rotation_loss_lambda})

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            if optimizer_idx == 0:
                lr = self.optimizers()[0].param_groups[0]['lr']
                self.log('lr_control', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)
            elif optimizer_idx == 1:
                lr = self.optimizers()[1].param_groups[0]['lr']
                self.log('lr_rotate', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()
    
    def p_losses(self, x_start, cond, t, noise=None):
        # NOTE(wjh): loss terms
        # x_start, is x_0
        noise = default(noise, lambda: torch.randn_like(x_start))
        # x_t, sampled without parameters
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        # eps, i.e. x_{t-1} - x_{t}
        if self.roll_augment:
            roll_pixels_lr = int(torch.randint(0, cond['c_concat'][0].shape[-1], [1])[0])
            roll_pixels_hr = int(roll_pixels_lr * 8)
            shifted_img = torch.roll(x_noisy, roll_pixels_lr, dims=-1)
            cond['c_concat'][0] = torch.roll(cond['c_concat'][0], roll_pixels_lr, dims=-1)
            cond['control_input'][0] = torch.roll(cond['control_input'][0], roll_pixels_hr, dims=-1)
            if self.padding_augment:
                pad_length_half_lr = x_noisy.shape[-1] // 8
                pad_length_half_hr = pad_length_half_lr * 8
                shifted_img = self.rolling_padding(shifted_img, pad_length_half_lr)
                cond['c_concat'][0] = self.rolling_padding(cond['c_concat'][0], pad_length_half_lr)
                cond['control_input'][0] = self.rolling_padding(cond['control_input'][0], pad_length_half_hr)
                model_output = torch.roll(
                    self.apply_model(shifted_img, t, cond)[..., pad_length_half_lr:-pad_length_half_lr],
                    -roll_pixels_lr, dims=-1)
            else:
                model_output = torch.roll(self.apply_model(shifted_img, t, cond), -roll_pixels_lr, dims=-1)
        else:
            model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()

        # l2 loss between pred noise and gt noise
        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        # NOTE(wjh): not learnable log variance, fixed at 1 (i.e., log(var) = 0.0)
        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()
        if self.use_equivarient_loss:
            shifted_img = torch.roll(x_noisy, x_noisy.shape[-1] // 2, dims=-1)
            cond['c_concat'][0] = torch.roll(cond['c_concat'][0], cond['c_concat'][0].shape[-1] // 2, dims=-1)
            cond['control_input'][0] = torch.roll(cond['control_input'][0], cond['control_input'][0].shape[-1] // 2,
                                                  dims=-1)
            if self.padding_augment:
                pad_length_half_lr = x_noisy.shape[-1] // 8
                pad_length_half_hr = pad_length_half_lr * 8
                shifted_img = self.rolling_padding(shifted_img, pad_length_half_lr)
                cond['c_concat'][0] = self.rolling_padding(cond['c_concat'][0], pad_length_half_lr)
                cond['control_input'][0] = self.rolling_padding(cond['control_input'][0], pad_length_half_hr)
                rolled_model_outputs = torch.roll(
                    self.apply_model(shifted_img, t, cond)[..., pad_length_half_lr:-pad_length_half_lr],
                    -roll_pixels_lr, dims=-1)
            else:
                rolled_model_outputs = torch.roll(self.apply_model(shifted_img, t, cond), x_noisy.shape[-1] // 2,
                                                  dims=-1)

            loss_equi = self.get_loss(rolled_model_outputs, model_output, mean=False).mean([1, 2, 3])
            loss_dict.update({f'{prefix}/loss_equivarient': loss_equi.mean() * self.equi_loss_lambda})

            loss += self.equi_loss_lambda * loss_equi.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        # NOTE(wjh): self.original_elbo_weight=0, which means loss_vlb is useless.
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict

    @torch.no_grad()
    def save_latent_to_img(self, z, save_path='recon_pad.png', img_idx=0):
        # B images, only save the first

        recon_img = self.decode_first_stage(z)[img_idx].permute([1, 2, 0]) * 0.5 + 0.5
        recon_rolled = torch.roll(recon_img, recon_img.shape[1] // 2, dims=1)
        # cv2.imwrite(save_path, cv2.cvtColor(recon_rolled.cpu().numpy()*255, cv2.COLOR_RGB2BGR))

        return cv2.imwrite(save_path, cv2.cvtColor(recon_rolled.cpu().numpy() * 255, cv2.COLOR_RGB2BGR))

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        # Still Bug to fix
        if self.padding_augment:
            '''
            TODO:
            after this, the connecting problem is even worse
            '''
            pad_length = z.shape[-1] // 10
            return self.scale_factor * z[..., pad_length:-pad_length]

        return self.scale_factor * z

    @torch.no_grad()
    def encode_first_stage(self, x):
        if self.padding_augment:
            pad_length_half = x.shape[-1] // 8
            x = self.rolling_padding(x, pad_length_half)

        return self.first_stage_model.encode(x)

    def p_losses(self, x_start, cond, t, noise=None):
        # NOTE(wjh): loss terms
        # x_start, is x_0
        noise = default(noise, lambda: torch.randn_like(x_start))
        # x_t, sampled without parameters
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        # eps, i.e. x_{t-1} - x_{t}
        if self.roll_augment:
            roll_pixels_lr = int(torch.randint(0,cond['c_concat'][0].shape[-1], [1])[0])
            roll_pixels_hr = int(roll_pixels_lr*8)
            shifted_img = torch.roll(x_noisy, roll_pixels_lr, dims=-1)
            cond['c_concat'][0] = torch.roll(cond['c_concat'][0], roll_pixels_lr, dims=-1)
            cond['control_input'][0] = torch.roll(cond['control_input'][0], roll_pixels_hr, dims=-1)
            if self.padding_augment:
                pad_length_half_lr = x_noisy.shape[-1] // 8
                pad_length_half_hr = pad_length_half_lr * 8
                shifted_img = self.rolling_padding(shifted_img, pad_length_half_lr)
                cond['c_concat'][0] = self.rolling_padding(cond['c_concat'][0], pad_length_half_lr)
                cond['control_input'][0] = self.rolling_padding(cond['control_input'][0], pad_length_half_hr)
                model_output = torch.roll(self.apply_model(shifted_img, t, cond)[...,pad_length_half_lr:-pad_length_half_lr],-roll_pixels_lr, dims=-1)
            else:        
                model_output = torch.roll(self.apply_model(shifted_img, t, cond), -roll_pixels_lr, dims=-1)
        
        elif self.deform_augment:
            bs = x_noisy.shape[0]
            roll_theta = torch.randn([bs]).to(self.device)*180.0 # -180, 180
            roll_phi = torch.randn([bs]).to(self.device)*15.0+15.0 # -15, 15

            x_noisy = deform_a_little(img=x_noisy,
                                    delta_theta=roll_theta,
                                    delta_phi=roll_phi,
                                    pano_height=x_noisy.shape[2],
                                    pano_width=x_noisy.shape[3])
            cond['c_concat'][0] = deform_a_little(img=cond['c_concat'][0],
                                    delta_theta=roll_theta,
                                    delta_phi=roll_phi,
                                    pano_height=cond['c_concat'][0].shape[2],
                                    pano_width=cond['c_concat'][0].shape[3])
            cond['control_input'][0] = deform_a_little(img=cond['control_input'][0],
                                    delta_theta=roll_theta,
                                    delta_phi=roll_phi,
                                    pano_height=cond['control_input'][0].shape[2],
                                    pano_width=cond['control_input'][0].shape[3])
            model_output = self.apply_model(x_noisy, t, cond)
            model_output = deform_a_little(img=model_output,
                                    delta_theta=-roll_theta,
                                    delta_phi=-roll_phi,
                                    pano_height=model_output.shape[2],
                                    pano_width=model_output.shape[3])
            
        else:
            model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()

        # l2 loss between pred noise and gt noise
        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})
        

        # NOTE(wjh): not learnable log variance, fixed at 1 (i.e., log(var) = 0.0)
        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()
        if self.use_equivarient_loss:
            shifted_img = torch.roll(x_noisy, x_noisy.shape[-1]//2, dims=-1)
            cond['c_concat'][0] = torch.roll(cond['c_concat'][0], cond['c_concat'][0].shape[-1]//2, dims=-1)
            cond['control_input'][0] = torch.roll(cond['control_input'][0], cond['control_input'][0].shape[-1]//2, dims=-1)
            if self.padding_augment:
                pad_length_half_lr = x_noisy.shape[-1] // 8
                pad_length_half_hr = pad_length_half_lr * 8
                shifted_img = self.rolling_padding(shifted_img, pad_length_half_lr)
                cond['c_concat'][0] = self.rolling_padding(cond['c_concat'][0], pad_length_half_lr)
                cond['control_input'][0] = self.rolling_padding(cond['control_input'][0], pad_length_half_hr)
                rolled_model_outputs = torch.roll(self.apply_model(shifted_img, t, cond)[...,pad_length_half_lr:-pad_length_half_lr],-roll_pixels_lr, dims=-1)
            else:
                rolled_model_outputs = torch.roll(self.apply_model(shifted_img, t, cond), x_noisy.shape[-1]//2, dims=-1)
            
            loss_equi = self.get_loss(rolled_model_outputs, model_output, mean=False).mean([1, 2, 3])
            loss_dict.update({f'{prefix}/loss_equivarient': loss_equi.mean()*self.equi_loss_lambda})

            loss += self.equi_loss_lambda * loss_equi.mean()
        

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        # NOTE(wjh): self.original_elbo_weight=0, which means loss_vlb is useless.
        loss += (self.original_elbo_weight * loss_vlb) 
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict

    @torch.no_grad()
    def save_latent_to_img(self, z, save_path='recon_pad.png', img_idx=0):
        # B images, only save the first
        
        recon_img = self.decode_first_stage(z)[img_idx].permute([1,2,0])*0.5+0.5
        recon_rolled = torch.roll(recon_img, recon_img.shape[1]//2, dims=1)
        #cv2.imwrite(save_path, cv2.cvtColor(recon_rolled.cpu().numpy()*255, cv2.COLOR_RGB2BGR))

        return cv2.imwrite(save_path, cv2.cvtColor(recon_rolled.cpu().numpy()*255, cv2.COLOR_RGB2BGR))

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        # Still Bug to fix
        if self.padding_augment:
            '''
            TODO:
            after this, the connecting problem is even worse
            '''
            pad_length = z.shape[-1]//10
            return self.scale_factor * z[..., pad_length:-pad_length]
        
        return self.scale_factor * z
    
    @torch.no_grad()
    def encode_first_stage(self, x):
        if self.padding_augment:
            pad_length_half = x.shape[-1] // 8
            x = self.rolling_padding(x, pad_length_half)
        
        return self.first_stage_model.encode(x)
    
class No_Rota_Inpaint(LatentInpaintDiffusion):
    def __init__(
            self,
            control_stage_config,
            concat_keys=("mask", "hint"),
            masked_image_key="hint",
            control_key=None,
            only_mid_control=False,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.masked_image_key = masked_image_key
        assert self.masked_image_key in concat_keys
        self.concat_keys = concat_keys

        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13

    # @torch.no_grad()
    def get_input(
            self, batch, k, bs=None, return_first_stage_outputs=False, *args, **kwargs):

        if 'hint' not in batch:  # In test, we don't need to sample this everytime
            if self.down_scale == 2:
                patch1, patch2 = inverse_normalize(batch['img1']), inverse_normalize(batch['img2'])
            elif self.down_scale == 1:
                patch1, patch2 = batch['img1_original'].permute(0, 3, 1, 2), batch['img2_original'].permute(0, 3, 1, 2)
            else:
                raise ValueError

            
            if self.use_gt_rots:
                # fix gt
                pano1, mask1 = sample_from_patch(patch1,
                                                    theta=-(batch['rotation_x1'] * 180 / torch.pi).to(patch1.dtype),
                                                    phi=(batch['rotation_y1'] * 180 / torch.pi).to(patch1.dtype),
                                                    # angle_y1,
                                                    pano_height=batch['pano'].shape[1],
                                                    pano_width=batch['pano'].shape[2])

                delta_theta_gt = ((batch['rotation_x2'] - batch['rotation_x1']) * 180 / torch.pi).to(patch1.dtype)
                delta_theta_gt[torch.where(delta_theta_gt > 180)] -= 360.0
                delta_theta_gt[torch.where(delta_theta_gt < -180)] += 360.0

                pano2, mask2 = sample_from_patch(patch2,
                                                    theta=-(batch['rotation_x2'] * 180 / torch.pi).to(patch1.dtype),
                                                    phi=(batch['rotation_y2'] * 180 / torch.pi).to(patch1.dtype),
                                                    pano_height=batch['pano'].shape[1],
                                                    pano_width=batch['pano'].shape[2])
            else:
                pano1, mask1 = sample_from_patch(patch1,
                                                    theta=torch.zeros_like(batch['rotation_x1']),
                                                    phi=(batch['rotation_y1'] * 180 / torch.pi).to(patch1.dtype),
                                                    pano_height=batch['pano'].shape[1],
                                                    pano_width=batch['pano'].shape[2])

                delta_theta_gt = ((batch['rotation_x2'] - batch['rotation_x1']) * 180 / torch.pi).to(patch1.dtype)
                delta_theta_gt[torch.where(delta_theta_gt > 180)] -= 360.0
                delta_theta_gt[torch.where(delta_theta_gt < -180)] += 360.0

                pano2, mask2 = sample_from_patch(patch2,
                                                    theta=-delta_theta_gt,
                                                    phi=(batch['rotation_y2'] * 180 / torch.pi).to(patch1.dtype),
                                                    pano_height=batch['pano'].shape[1],
                                                    pano_width=batch['pano'].shape[2])
                
            # shift the ground truth image laterally
            batch_size = batch['pano'].shape[0]
            y_grid, x_grid = torch.meshgrid(
                torch.linspace(-1, 1, batch['pano'].shape[1]),
                torch.linspace(-1, 1, batch['pano'].shape[2]))
            y_grid = y_grid[None, :].repeat([batch_size, 1, 1]).to(batch['pano'].device)
            x_grid = x_grid[None, :].repeat([batch_size, 1, 1]).to(batch['pano'].device)

            if self.use_gt_rots:
                x_grid = x_grid  # + (batch['rotation_x1'] / torch.pi).reshape([-1, 1, 1])
            else:
                x_grid = x_grid + (batch['rotation_x1'] / torch.pi).reshape([-1, 1, 1])
            x_grid[x_grid > 1] -= 2
            x_grid[x_grid < -1] += 2
            # Stack the x and y grids along the channel dimension
            x_grid = torch.tensor(x_grid, dtype=batch['pano'].dtype, device=batch['pano'].device)
            y_grid = torch.tensor(y_grid, dtype=batch['pano'].dtype, device=batch['pano'].device)
            grid = torch.stack([x_grid, y_grid], dim=3)  # BHW2

            batch['jpg'] = F.grid_sample(batch['pano'].permute([0, 3, 1, 2]), grid, align_corners=True).permute(
                [0, 2, 3, 1])

            # normalize, as in controlNet dataset
            batch['jpg'] = batch['jpg'] * 2 - 1.0

            # merge these 2 panos and masks together
            batch['mask'] = torch.bitwise_or(mask1, mask2)

            batch['hint'] = torch.zeros_like(batch['jpg'])
            batch['hint'][batch['mask']] = pano1[batch['mask']]
            batch['hint'][(~mask1) * mask2] = pano2[(~mask1) * mask2]  # set pano1 as main, mask2 as help.
            batch['mask'] = batch['mask'][..., None]
            batch['mask'] = torch.where(batch['mask'] == 0, 1, 0)


        with torch.no_grad():
            '''
            TODO:
            Rota_Inpaint <- LatentInpaintDiffusion <- LatentDiffusion
            What did get input do in every inheritance?
            1. Latent Diffusion
            conditionining keys
            2. Latent Inpaint Diffusion
            concat_keys
            3. Rota_Inpaint
            '''
            assert (
                not self.cond_stage_trainable
            ), "trainable cond stages not yet supported for inpainting"
            # NOTE(wjh)
            z, all_conds, x, xrec, xc = super().get_input(
                batch,
                self.first_stage_key,
                return_first_stage_outputs=True,
            )

            assert exists(self.concat_keys)
            c_control = list()
            for ck in self.concat_keys:
                cc = (
                    rearrange(batch[ck], "b h w c -> b c h w")
                    .to(memory_format=torch.contiguous_format)
                    .float()
                )
                if bs is not None:
                    cc = cc[:bs]
                    cc = cc.to(self.device)
                c_control.append(cc)
            c_control = torch.cat(c_control, dim=1)

            all_conds['control_input'] = [c_control]

            if return_first_stage_outputs:
                return z, all_conds, x, xrec, xc

            return z, all_conds

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):

        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond['c_crossattn'], 1)
        xc = torch.cat([x_noisy] + cond['c_concat'], dim=1)

        if 'control_input' not in cond:
            eps = diffusion_model(x=xc, timesteps=t, context=cond_txt, control=None,
                                  only_mid_control=self.only_mid_control)
        else:
            control = self.control_model(x=xc, hint=torch.cat(cond['control_input'], 1), timesteps=t, context=cond_txt)
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            eps = diffusion_model(x=xc, timesteps=t, context=cond_txt, control=control,
                                  only_mid_control=self.only_mid_control)

        return eps

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = PanoSampler(self)
        b, c, h, w = cond["control_input"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @staticmethod
    def rolling_padding(tensor_to_pad: torch.Tensor, half_pad_length: int):

        left_pad = tensor_to_pad[..., -half_pad_length:]
        right_pad = tensor_to_pad[..., :half_pad_length]

        return torch.cat([left_pad, tensor_to_pad, right_pad], dim=-1)

    @staticmethod
    def rolling_padding(tensor_to_pad:torch.Tensor, half_pad_length:int):

        left_pad = tensor_to_pad[...,-half_pad_length:]
        right_pad = tensor_to_pad[...,:half_pad_length]
        
        return torch.cat([left_pad, tensor_to_pad, right_pad], dim=-1)
    
    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)

        c_cat, c_control, c = c["c_concat"][0][:N], c["control_input"][0][:N], c["c_crossattn"][0][:N]
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z)
        
        # DEBUG: latent reconstruction
        # import cv2
        # test_img = cv2.cvtColor(cv2.imread('datasets/rotation_blip_dataset_train/raw_crops/undist/00565/panorama.jpg'),
        #                         cv2.COLOR_BGR2RGB)
        # input_tensor = torch.from_numpy(test_img)/127.5 - 1.0 # -1 ~ 1
        # input_tensor = input_tensor[None, :].permute([0, 3, 1, 2]).to(self.device)
        # rolled_input_tensor = torch.roll(input_tensor,
        #                                  shifts=input_tensor.shape[-1]//2,
        #                                  dims=-1)

        # # get encoded z
        # encoder_posterior = self.encode_first_stage(input_tensor)
        # z = self.get_first_stage_encoding(encoder_posterior).detach()
        # rolled_encoder_posterior = self.encode_first_stage(rolled_input_tensor)
        # rolled_z = self.get_first_stage_encoding(rolled_encoder_posterior).detach()

        # # operate on latent space
        # z_latent_roll = torch.roll(z, shifts=z.shape[-1]//2, dims=-1)

        # recon = self.decode_first_stage(z)[0]
        # rolled_recon = self.decode_first_stage(rolled_z)[0]
        # recon_latent_roll = self.decode_first_stage(z_latent_roll)[0]

        # # roll them to see the left and right
        # recon = torch.roll(recon, shifts=recon.shape[-1]//2, dims=-1)
        # rolled_recon = torch.roll(rolled_recon, shifts=rolled_recon.shape[-1]//2, dims=-1)

        # # test if the trick works
        # recon = (recon/2+0.5).permute([1,2,0]).cpu().numpy()
        # rolled_recon = (rolled_recon/2+0.5).permute([1,2,0]).cpu().numpy()
        # recon_latent_roll = (recon_latent_roll/2+0.5).permute([1,2,0]).cpu().numpy()
        # # 
        # recon_mixed = recon.copy()
        # recon_mixed[:,recon.shape[1]//4:recon.shape[1]//4*3] = recon_latent_roll[:,recon.shape[1]//4:recon.shape[1]//4*3]
        # #gt = (batch['jpg'][0]/2 + 0.5).cpu().numpy()
        # cv2.imwrite('recon.png', cv2.cvtColor(recon*255, cv2.COLOR_RGB2BGR))
        # cv2.imwrite('recon_latent_roll.png', cv2.cvtColor(recon_latent_roll*255, cv2.COLOR_RGB2BGR))
        # cv2.imwrite('rolled_recon.png', cv2.cvtColor(rolled_recon*255, cv2.COLOR_RGB2BGR))
        # cv2.imwrite('recon_latent_roll_mixed.png', cv2.cvtColor(recon_mixed*255, cv2.COLOR_RGB2BGR))

        log["control"] = c_control[:, -3:] * 2.0 - 1.0  # NOTE(wjh): only output rgb layers.
        _mask = c_control[:, :1]
        log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            samples, z_denoise_row = self.sample_log(
                cond={"c_concat": [c_cat], "control_input": [c_control], "c_crossattn": [c]},
                batch_size=N, ddim=use_ddim,
                ddim_steps=ddim_steps, eta=ddim_eta, mask=_mask)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            # NOTE(wjh):
            # used in condition guided sample:
            # model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)

            uc_cross = self.get_unconditional_conditioning(N)
            uc_control = c_control  # torch.zeros_like(c_cat)
            uc_full = {"control_input": [uc_control], "c_crossattn": [uc_cross], "c_concat": [c_cat]}
            # uc_full = {"c_crossattn": [uc_cross], "c_concat": [c_cat]}
            samples_cfg, _ = self.sample_log(
                # cond={"c_concat": [c_cat], "c_crossattn": [c]},
                cond={"c_concat": [c_cat], "c_crossattn": [c], "control_input": [c_control]},
                batch_size=N, ddim=use_ddim,
                ddim_steps=ddim_steps, eta=ddim_eta,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=uc_full,
                # mask=c_cat[:, :1],
                # x0=z[:N]
            )
            # trick, make the ends meet
            if self.padding_augment:
                pad_length = samples_cfg.shape[-1] // 8
                samples_cfg = self.rolling_padding(samples_cfg, pad_length)
            
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            if self.padding_augment:
                pad_length_hr = pad_length * 8
                x_samples_cfg = x_samples_cfg[..., pad_length_hr:-pad_length_hr]
            # x_samples_cfg = torch.clip(x_samples_cfg, min=-1, max=1)
            log[
                f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg  # x_samples_cfg *  _mask + log["control"] * (1 - _mask)
            # log["pred_x0"] = self.decode_first_stage(_['pred_x0'][0])

        return log

    def configure_optimizers(self):
        lr_control = self.learning_rate

        control_params = list(self.control_model.parameters())
        if not self.sd_locked:
            control_params += list(self.model.diffusion_model.output_blocks.parameters())
            control_params += list(self.model.diffusion_model.out.parameters())

        control_opt = torch.optim.AdamW(control_params, lr=lr_control)

        return [control_opt]

    def training_step(self, batch, batch_idx):
        for k in self.ucg_training:
            p = self.ucg_training[k]["p"]
            val = self.ucg_training[k]["val"]
            if val is None:
                val = ""
            for i in range(len(batch[k])):
                if self.ucg_prng.choice(2, p=[1 - p, p]):
                    batch[k][i] = val

        loss, loss_dict = self.shared_step(batch)

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers()[0].param_groups[0]['lr']
            self.log('lr_control', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()
    
    def p_losses(self, x_start, cond, t, noise=None):
        # NOTE(wjh): loss terms
        # x_start, is x_0
        noise = default(noise, lambda: torch.randn_like(x_start))
        # x_t, sampled without parameters
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        # eps, i.e. x_{t-1} - x_{t}
        if self.roll_augment:
            roll_pixels_lr = int(torch.randint(0, cond['c_concat'][0].shape[-1], [1])[0])
            roll_pixels_hr = int(roll_pixels_lr * 8)
            shifted_img = torch.roll(x_noisy, roll_pixels_lr, dims=-1)
            cond['c_concat'][0] = torch.roll(cond['c_concat'][0], roll_pixels_lr, dims=-1)
            cond['control_input'][0] = torch.roll(cond['control_input'][0], roll_pixels_hr, dims=-1)
            if self.padding_augment:
                pad_length_half_lr = x_noisy.shape[-1] // 8
                pad_length_half_hr = pad_length_half_lr * 8
                shifted_img = self.rolling_padding(shifted_img, pad_length_half_lr)
                cond['c_concat'][0] = self.rolling_padding(cond['c_concat'][0], pad_length_half_lr)
                cond['control_input'][0] = self.rolling_padding(cond['control_input'][0], pad_length_half_hr)
                model_output = torch.roll(
                    self.apply_model(shifted_img, t, cond)[..., pad_length_half_lr:-pad_length_half_lr],
                    -roll_pixels_lr, dims=-1)
            else:
                model_output = torch.roll(self.apply_model(shifted_img, t, cond), -roll_pixels_lr, dims=-1)
        else:
            model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()

        # l2 loss between pred noise and gt noise
        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        # NOTE(wjh): not learnable log variance, fixed at 1 (i.e., log(var) = 0.0)
        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()
        if self.use_equivarient_loss:
            shifted_img = torch.roll(x_noisy, x_noisy.shape[-1] // 2, dims=-1)
            cond['c_concat'][0] = torch.roll(cond['c_concat'][0], cond['c_concat'][0].shape[-1] // 2, dims=-1)
            cond['control_input'][0] = torch.roll(cond['control_input'][0], cond['control_input'][0].shape[-1] // 2,
                                                  dims=-1)
            if self.padding_augment:
                pad_length_half_lr = x_noisy.shape[-1] // 8
                pad_length_half_hr = pad_length_half_lr * 8
                shifted_img = self.rolling_padding(shifted_img, pad_length_half_lr)
                cond['c_concat'][0] = self.rolling_padding(cond['c_concat'][0], pad_length_half_lr)
                cond['control_input'][0] = self.rolling_padding(cond['control_input'][0], pad_length_half_hr)
                rolled_model_outputs = torch.roll(
                    self.apply_model(shifted_img, t, cond)[..., pad_length_half_lr:-pad_length_half_lr],
                    -roll_pixels_lr, dims=-1)
            else:
                rolled_model_outputs = torch.roll(self.apply_model(shifted_img, t, cond), x_noisy.shape[-1] // 2,
                                                  dims=-1)

            loss_equi = self.get_loss(rolled_model_outputs, model_output, mean=False).mean([1, 2, 3])
            loss_dict.update({f'{prefix}/loss_equivarient': loss_equi.mean() * self.equi_loss_lambda})

            loss += self.equi_loss_lambda * loss_equi.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        # NOTE(wjh): self.original_elbo_weight=0, which means loss_vlb is useless.
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict

    @torch.no_grad()
    def save_latent_to_img(self, z, save_path='recon_pad.png', img_idx=0):
        # B images, only save the first

        recon_img = self.decode_first_stage(z)[img_idx].permute([1, 2, 0]) * 0.5 + 0.5
        recon_rolled = torch.roll(recon_img, recon_img.shape[1] // 2, dims=1)
        # cv2.imwrite(save_path, cv2.cvtColor(recon_rolled.cpu().numpy()*255, cv2.COLOR_RGB2BGR))

        return cv2.imwrite(save_path, cv2.cvtColor(recon_rolled.cpu().numpy() * 255, cv2.COLOR_RGB2BGR))

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        # Still Bug to fix
        if self.padding_augment:
            pad_length = z.shape[-1] // 10
            return self.scale_factor * z[..., pad_length:-pad_length]

        return self.scale_factor * z

    @torch.no_grad()
    def encode_first_stage(self, x):
        if self.padding_augment:
            pad_length_half = x.shape[-1] // 8
            x = self.rolling_padding(x, pad_length_half)

        return self.first_stage_model.encode(x)

    def p_losses(self, x_start, cond, t, noise=None):
        # NOTE(wjh): loss terms
        # x_start, is x_0
        noise = default(noise, lambda: torch.randn_like(x_start))
        # x_t, sampled without parameters
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        # eps, i.e. x_{t-1} - x_{t}
        if self.roll_augment:
            roll_pixels_lr = int(torch.randint(0,cond['c_concat'][0].shape[-1], [1])[0])
            roll_pixels_hr = int(roll_pixels_lr*8)
            shifted_img = torch.roll(x_noisy, roll_pixels_lr, dims=-1)
            cond['c_concat'][0] = torch.roll(cond['c_concat'][0], roll_pixels_lr, dims=-1)
            cond['control_input'][0] = torch.roll(cond['control_input'][0], roll_pixels_hr, dims=-1)
            if self.padding_augment:
                pad_length_half_lr = x_noisy.shape[-1] // 8
                pad_length_half_hr = pad_length_half_lr * 8
                shifted_img = self.rolling_padding(shifted_img, pad_length_half_lr)
                cond['c_concat'][0] = self.rolling_padding(cond['c_concat'][0], pad_length_half_lr)
                cond['control_input'][0] = self.rolling_padding(cond['control_input'][0], pad_length_half_hr)
                model_output = torch.roll(self.apply_model(shifted_img, t, cond)[...,pad_length_half_lr:-pad_length_half_lr],-roll_pixels_lr, dims=-1)
            else:        
                model_output = torch.roll(self.apply_model(shifted_img, t, cond), -roll_pixels_lr, dims=-1)
        
        elif self.deform_augment:
            bs = x_noisy.shape[0]
            roll_theta = torch.randn([bs]).to(self.device)*180.0 # -180, 180
            roll_phi = torch.randn([bs]).to(self.device)*15.0+15.0 # -15, 15

            x_noisy = deform_a_little(img=x_noisy,
                                    delta_theta=roll_theta,
                                    delta_phi=roll_phi,
                                    pano_height=x_noisy.shape[2],
                                    pano_width=x_noisy.shape[3])
            cond['c_concat'][0] = deform_a_little(img=cond['c_concat'][0],
                                    delta_theta=roll_theta,
                                    delta_phi=roll_phi,
                                    pano_height=cond['c_concat'][0].shape[2],
                                    pano_width=cond['c_concat'][0].shape[3])
            cond['control_input'][0] = deform_a_little(img=cond['control_input'][0],
                                    delta_theta=roll_theta,
                                    delta_phi=roll_phi,
                                    pano_height=cond['control_input'][0].shape[2],
                                    pano_width=cond['control_input'][0].shape[3])
            model_output = self.apply_model(x_noisy, t, cond)
            model_output = deform_a_little(img=model_output,
                                    delta_theta=-roll_theta,
                                    delta_phi=-roll_phi,
                                    pano_height=model_output.shape[2],
                                    pano_width=model_output.shape[3])
            
        else:
            model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()

        # l2 loss between pred noise and gt noise
        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})
        

        # NOTE(wjh): not learnable log variance, fixed at 1 (i.e., log(var) = 0.0)
        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()
        if self.use_equivarient_loss:
            shifted_img = torch.roll(x_noisy, x_noisy.shape[-1]//2, dims=-1)
            cond['c_concat'][0] = torch.roll(cond['c_concat'][0], cond['c_concat'][0].shape[-1]//2, dims=-1)
            cond['control_input'][0] = torch.roll(cond['control_input'][0], cond['control_input'][0].shape[-1]//2, dims=-1)
            if self.padding_augment:
                pad_length_half_lr = x_noisy.shape[-1] // 8
                pad_length_half_hr = pad_length_half_lr * 8
                shifted_img = self.rolling_padding(shifted_img, pad_length_half_lr)
                cond['c_concat'][0] = self.rolling_padding(cond['c_concat'][0], pad_length_half_lr)
                cond['control_input'][0] = self.rolling_padding(cond['control_input'][0], pad_length_half_hr)
                rolled_model_outputs = torch.roll(self.apply_model(shifted_img, t, cond)[...,pad_length_half_lr:-pad_length_half_lr],-roll_pixels_lr, dims=-1)
            else:
                rolled_model_outputs = torch.roll(self.apply_model(shifted_img, t, cond), x_noisy.shape[-1]//2, dims=-1)
            
            loss_equi = self.get_loss(rolled_model_outputs, model_output, mean=False).mean([1, 2, 3])
            loss_dict.update({f'{prefix}/loss_equivarient': loss_equi.mean()*self.equi_loss_lambda})

            loss += self.equi_loss_lambda * loss_equi.mean()
        

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        # NOTE(wjh): self.original_elbo_weight=0, which means loss_vlb is useless.
        loss += (self.original_elbo_weight * loss_vlb) 
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict

    @torch.no_grad()
    def save_latent_to_img(self, z, save_path='recon_pad.png', img_idx=0):
        # B images, only save the first
        
        recon_img = self.decode_first_stage(z)[img_idx].permute([1,2,0])*0.5+0.5
        recon_rolled = torch.roll(recon_img, recon_img.shape[1]//2, dims=1)
        #cv2.imwrite(save_path, cv2.cvtColor(recon_rolled.cpu().numpy()*255, cv2.COLOR_RGB2BGR))

        return cv2.imwrite(save_path, cv2.cvtColor(recon_rolled.cpu().numpy()*255, cv2.COLOR_RGB2BGR))

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        # Still Bug to fix
        if self.padding_augment:
            pad_length = z.shape[-1]//10
            return self.scale_factor * z[..., pad_length:-pad_length]
        
        return self.scale_factor * z
    
    @torch.no_grad()
    def encode_first_stage(self, x):
        if self.padding_augment:
            pad_length_half = x.shape[-1] // 8
            x = self.rolling_padding(x, pad_length_half)
        
        return self.first_stage_model.encode(x)
    