"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, extract_into_tensor
# from RotationModels.utils.pano_utils import deform_a_littlefrom RotationModels.utils.pano_utils import deform_a_little

class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps, verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta, verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                    1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               dynamic_threshold=None,
               ucg_schedule=None,
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                ctmp = conditioning[list(conditioning.keys())[0]]
                while isinstance(ctmp, list): ctmp = ctmp[0]
                cbs = ctmp.shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")

            elif isinstance(conditioning, list):
                for ctmp in conditioning:
                    if ctmp.shape[0] != batch_size:
                        print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")

            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    dynamic_threshold=dynamic_threshold,
                                                    ucg_schedule=ucg_schedule
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, dynamic_threshold=None,
                      ucg_schedule=None):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0, timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            if ucg_schedule is not None:
                assert len(ucg_schedule) == len(time_range)
                unconditional_guidance_scale = ucg_schedule[i]

            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,
                                      dynamic_threshold=dynamic_threshold)
            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      dynamic_threshold=None):
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            model_output = self.model.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            if isinstance(c, dict):
                assert isinstance(unconditional_conditioning, dict)
                c_in = dict()
                for k in c:
                    if isinstance(c[k], list):
                        c_in[k] = [torch.cat([
                            unconditional_conditioning[k][i],
                            c[k][i]]) for i in range(len(c[k]))]
                    else:
                        c_in[k] = torch.cat([
                            unconditional_conditioning[k],
                            c[k]])
            elif isinstance(c, list):
                c_in = list()
                assert isinstance(unconditional_conditioning, list)
                for i in range(len(c)):
                    c_in.append(torch.cat([unconditional_conditioning[i], c[i]]))
            else:
                c_in = torch.cat([unconditional_conditioning, c])
            model_uncond, model_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)

        if self.model.parameterization == "v":
            e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
        else:
            e_t = model_output

        if score_corrector is not None:
            assert self.model.parameterization == "eps", 'not implemented'
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)

        # current prediction for x_0
        if self.model.parameterization != "v":
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        else:
            pred_x0 = self.model.predict_start_from_z_and_v(x, t, model_output)

        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

        if dynamic_threshold is not None:
            raise NotImplementedError()

        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t ** 2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0

    @torch.no_grad()
    def encode(self, x0, c, t_enc, use_original_steps=False, return_intermediates=None,
               unconditional_guidance_scale=1.0, unconditional_conditioning=None, callback=None):
        num_reference_steps = self.ddpm_num_timesteps if use_original_steps else self.ddim_timesteps.shape[0]

        assert t_enc <= num_reference_steps
        num_steps = t_enc

        if use_original_steps:
            alphas_next = self.alphas_cumprod[:num_steps]
            alphas = self.alphas_cumprod_prev[:num_steps]
        else:
            alphas_next = self.ddim_alphas[:num_steps]
            alphas = torch.tensor(self.ddim_alphas_prev[:num_steps])

        x_next = x0
        intermediates = []
        inter_steps = []
        for i in tqdm(range(num_steps), desc='Encoding Image'):
            t = torch.full((x0.shape[0],), i, device=self.model.device, dtype=torch.long)
            if unconditional_guidance_scale == 1.:
                noise_pred = self.model.apply_model(x_next, t, c)
            else:
                assert unconditional_conditioning is not None
                e_t_uncond, noise_pred = torch.chunk(
                    self.model.apply_model(torch.cat((x_next, x_next)), torch.cat((t, t)),
                                           torch.cat((unconditional_conditioning, c))), 2)
                noise_pred = e_t_uncond + unconditional_guidance_scale * (noise_pred - e_t_uncond)

            xt_weighted = (alphas_next[i] / alphas[i]).sqrt() * x_next
            weighted_noise_pred = alphas_next[i].sqrt() * (
                    (1 / alphas_next[i] - 1).sqrt() - (1 / alphas[i] - 1).sqrt()) * noise_pred
            x_next = xt_weighted + weighted_noise_pred
            if return_intermediates and i % (
                    num_steps // return_intermediates) == 0 and i < num_steps - 1:
                intermediates.append(x_next)
                inter_steps.append(i)
            elif return_intermediates and i >= num_steps - 2:
                intermediates.append(x_next)
                inter_steps.append(i)
            if callback: callback(i)

        out = {'x_encoded': x_next, 'intermediate_steps': inter_steps}
        if return_intermediates:
            out.update({'intermediates': intermediates})
        return x_next, out

    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False, callback=None):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
            if callback: callback(i)
        return x_dec


class PanoSampler(DDIMSampler):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__(model, schedule, **kwargs)

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, dynamic_threshold=None,
                      ucg_schedule=None):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0, timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        ''' TODO (wjh)
        add rotation augmentation to all the tensors,
        imgs, conds(obtained by other things.)
        New schedule of sampling! Using random rotation augmentation to help sample.
        '''
        phis_pertubate = torch.linspace(0, 2 * torch.pi, total_steps,
                                        device=img.device, dtype=img.dtype)
        phis_pertubate = torch.sin(phis_pertubate) * 30  # the range of phi turbulance is 30
        phis_steps = torch.zeros_like(phis_pertubate)
        phis_steps[1:] = phis_pertubate[1:] - phis_pertubate[:-1]
        phis_steps = phis_steps[:, None].repeat([1, b])

        delta_step = torch.ones_like(phis_steps[0]) * 360 / total_steps

        # flip_times = 0
        num_rotate_steps = 32
        offset_pixels_lr = int(img.shape[-1] / num_rotate_steps)
        offset_pixels_hr = int(offset_pixels_lr * 8)
        acc_pixels_lr = 0
        acc_pixels_hr = 0
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            if ucg_schedule is not None:
                assert len(ucg_schedule) == len(time_range)
                unconditional_guidance_scale = ucg_schedule[i]

            # deform a little bit on this.

            if self.model.roll_augment:  # i < num_rotate_steps
                # flip_times += 1
                '''img = deform_a_little(img=img,
                                    delta_theta=torch.ones_like(delta_step)*180,
                                    delta_phi=phis_steps[25],
                                    pano_height=img.shape[2],
                                    pano_width=img.shape[3])
                cond['c_concat'][0] = deform_a_little(img=cond['c_concat'][0],
                                    delta_theta=torch.ones_like(delta_step)*180,
                                    delta_phi=phis_steps[25],
                                    pano_height=cond['c_concat'][0].shape[2],
                                    pano_width=cond['c_concat'][0].shape[3])
                cond['control_input'][0] = deform_a_little(img=cond['control_input'][0],
                                    delta_theta=torch.ones_like(delta_step)*180,
                                    delta_phi=phis_steps[25],
                                    pano_height=cond['control_input'][0].shape[2],
                                    pano_width=cond['control_input'][0].shape[3])'''
                # flip at first, and then in the middle
                # offset_pixels_lr *= num_rotate_steps // 2
                # offset_pixels_hr *= num_rotate_steps // 2

                if i < num_rotate_steps:
                    img = torch.roll(img, offset_pixels_lr, dims=-1)
                    cond['c_concat'][0] = torch.roll(cond['c_concat'][0], offset_pixels_lr, dims=-1)
                    cond['control_input'][0] = torch.roll(cond['control_input'][0], offset_pixels_hr, dims=-1)
                    if unconditional_conditioning is not None:
                        unconditional_conditioning['c_concat'][0] = torch.roll(
                            unconditional_conditioning['c_concat'][0], offset_pixels_lr, dims=-1)
                        unconditional_conditioning['control_input'][0] = torch.roll(
                            unconditional_conditioning['control_input'][0], offset_pixels_hr, dims=-1)

                    acc_pixels_hr += offset_pixels_hr
                    acc_pixels_lr += offset_pixels_lr
                '''else:
                    img = torch.roll(img, -acc_pixels_lr, dims=-1)
                    cond['c_concat'][0] = torch.roll(cond['c_concat'][0], -acc_pixels_lr, dims=-1)
                    cond['control_input'][0] = torch.roll(cond['control_input'][0], -acc_pixels_hr, dims=-1)'''
            else:
                '''img = deform_a_little(img=img,
                                    delta_theta=torch.zeros_like(delta_step),
                                    delta_phi=torch.zeros_like(phis_steps[i]),
                                    pano_height=img.shape[2],
                                    pano_width=img.shape[3])
                cond['c_concat'][0] = deform_a_little(img=cond['c_concat'][0],
                                    delta_theta=torch.zeros_like(delta_step),
                                    delta_phi=torch.zeros_like(phis_steps[i]),
                                    pano_height=cond['c_concat'][0].shape[2],
                                    pano_width=cond['c_concat'][0].shape[3])
                cond['control_input'][0] = deform_a_little(img=cond['control_input'][0],
                                    delta_theta=torch.zeros_like(delta_step),
                                    delta_phi=torch.zeros_like(phis_steps[i]),
                                    pano_height=cond['control_input'][0].shape[2],
                                    pano_width=cond['control_input'][0].shape[3])'''
                a = 1

            if self.model.padding_augment:
                # The padding happens in the loop!
                # TODO:
                # maybe the condition do not need to be padded everytime,
                # since they did not really change.
                pad_length_half_lr = img.shape[-1] // 4
                pad_length_half_hr = pad_length_half_lr * 8
                img = self.model.rolling_padding(img, pad_length_half_lr)
                cond['c_concat'][0] = self.model.rolling_padding(cond['c_concat'][0], pad_length_half_lr)
                cond['control_input'][0] = self.model.rolling_padding(cond['control_input'][0], pad_length_half_hr)
                if unconditional_conditioning is not None:
                    unconditional_conditioning['c_concat'][0] = self.model.rolling_padding(
                        unconditional_conditioning['c_concat'][0], pad_length_half_lr)
                    unconditional_conditioning['control_input'][0] = self.model.rolling_padding(
                        unconditional_conditioning['control_input'][0], pad_length_half_hr)

            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,
                                      dynamic_threshold=dynamic_threshold)
            img, pred_x0 = outs
            if self.model.padding_augment:
                img = img[..., pad_length_half_lr:-pad_length_half_lr]
                cond['c_concat'][0] = cond['c_concat'][0][..., pad_length_half_lr:-pad_length_half_lr]
                cond['control_input'][0] = cond['control_input'][0][..., pad_length_half_hr:-pad_length_half_hr]
                if unconditional_conditioning is not None:
                    unconditional_conditioning['c_concat'][0] = unconditional_conditioning['c_concat'][0][...,
                                                                pad_length_half_lr:-pad_length_half_lr]
                    unconditional_conditioning['control_input'][0] = unconditional_conditioning['control_input'][0][...,
                                                                     pad_length_half_hr:-pad_length_half_hr]

            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        # DEBUG (wjh)
        # img = torch.roll(img, -offset_pixels_lr, dims=-1)
        # cond['c_concat'][0] = torch.roll(cond['c_concat'][0], -offset_pixels_lr, dims=-1)
        # cond['control_input'][0] = torch.roll(cond['control_input'][0], -offset_pixels_hr, dims=-1)

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      dynamic_threshold=None):
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            model_output = self.model.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            if isinstance(c, dict):
                assert isinstance(unconditional_conditioning, dict)
                c_in = dict()
                for k in c:
                    if isinstance(c[k], list):
                        c_in[k] = [torch.cat([
                            unconditional_conditioning[k][i],
                            c[k][i]]) for i in range(len(c[k]))]
                    else:
                        c_in[k] = torch.cat([
                            unconditional_conditioning[k],
                            c[k]])
            elif isinstance(c, list):
                c_in = list()
                assert isinstance(unconditional_conditioning, list)
                for i in range(len(c)):
                    c_in.append(torch.cat([unconditional_conditioning[i], c[i]]))
            else:
                c_in = torch.cat([unconditional_conditioning, c])
            model_uncond, model_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)

        if self.model.parameterization == "v":
            e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
        else:
            e_t = model_output

        if score_corrector is not None:
            assert self.model.parameterization == "eps", 'not implemented'
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)

        # current prediction for x_0
        if self.model.parameterization != "v":
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        else:
            pred_x0 = self.model.predict_start_from_z_and_v(x, t, model_output)

        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

        if dynamic_threshold is not None:
            raise NotImplementedError()

        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t ** 2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0



class PanoSampler(DDIMSampler):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__(model, schedule, **kwargs)
    
    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, dynamic_threshold=None,
                      ucg_schedule=None):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]


        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        ''' TODO (wjh)
        add rotation augmentation to all the tensors,
        imgs, conds(obtained by other things.)
        New schedule of sampling! Using random rotation augmentation to help sample.
        '''
        phis_pertubate = torch.linspace(0, 2*torch.pi, total_steps, 
                                        device=img.device, dtype=img.dtype)
        phis_pertubate = torch.sin(phis_pertubate) * 30 # the range of phi turbulance is 30
        phis_steps = torch.zeros_like(phis_pertubate)
        phis_steps[1:] = phis_pertubate[1:] - phis_pertubate[:-1]
        phis_steps = phis_steps[:,None].repeat([1, b])
        
        delta_step = torch.ones_like(phis_steps[0])*360/total_steps
        
        #flip_times = 0
        num_rotate_steps = 32
        offset_pixels_lr = int(img.shape[-1]/num_rotate_steps)
        offset_pixels_hr = int(offset_pixels_lr*8)
        acc_pixels_lr = 0
        acc_pixels_hr = 0
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            if ucg_schedule is not None:
                assert len(ucg_schedule) == len(time_range)
                unconditional_guidance_scale = ucg_schedule[i]

            # deform a little bit on this.
            
            if self.model.roll_augment: # i < num_rotate_steps
                #flip_times += 1
                '''img = deform_a_little(img=img,
                                    delta_theta=torch.ones_like(delta_step)*180,
                                    delta_phi=phis_steps[25],
                                    pano_height=img.shape[2],
                                    pano_width=img.shape[3])
                cond['c_concat'][0] = deform_a_little(img=cond['c_concat'][0],
                                    delta_theta=torch.ones_like(delta_step)*180,
                                    delta_phi=phis_steps[25],
                                    pano_height=cond['c_concat'][0].shape[2],
                                    pano_width=cond['c_concat'][0].shape[3])
                cond['control_input'][0] = deform_a_little(img=cond['control_input'][0],
                                    delta_theta=torch.ones_like(delta_step)*180,
                                    delta_phi=phis_steps[25],
                                    pano_height=cond['control_input'][0].shape[2],
                                    pano_width=cond['control_input'][0].shape[3])'''
                # flip at first, and then in the middle
                #offset_pixels_lr *= num_rotate_steps // 2
                #offset_pixels_hr *= num_rotate_steps // 2

                if i < num_rotate_steps:
                    img = torch.roll(img, offset_pixels_lr, dims=-1)
                    cond['c_concat'][0] = torch.roll(cond['c_concat'][0], offset_pixels_lr, dims=-1)
                    cond['control_input'][0] = torch.roll(cond['control_input'][0], offset_pixels_hr, dims=-1)
                    if unconditional_conditioning is not None:
                        unconditional_conditioning['c_concat'][0] = torch.roll(unconditional_conditioning['c_concat'][0], offset_pixels_lr, dims=-1)
                        unconditional_conditioning['control_input'][0] = torch.roll(unconditional_conditioning['control_input'][0], offset_pixels_hr, dims=-1)
                    
                    acc_pixels_hr += offset_pixels_hr
                    acc_pixels_lr += offset_pixels_lr
                '''else:
                    img = torch.roll(img, -acc_pixels_lr, dims=-1)
                    cond['c_concat'][0] = torch.roll(cond['c_concat'][0], -acc_pixels_lr, dims=-1)
                    cond['control_input'][0] = torch.roll(cond['control_input'][0], -acc_pixels_hr, dims=-1)'''
            elif self.model.deform_augment:
                img = deform_a_little(img=img,
                                        delta_theta=delta_step,
                                        delta_phi=phis_steps[i],
                                        pano_height=img.shape[2],
                                        pano_width=img.shape[3])
                cond['c_concat'][0] = deform_a_little(img=cond['c_concat'][0],
                                        delta_theta=delta_step,
                                        delta_phi=phis_steps[i],
                                        pano_height=cond['c_concat'][0].shape[2],
                                        pano_width=cond['c_concat'][0].shape[3])
                cond['control_input'][0] = deform_a_little(img=cond['control_input'][0],
                                        delta_theta=delta_step,
                                        delta_phi=phis_steps[i],
                                        pano_height=cond['control_input'][0].shape[2],
                                        pano_width=cond['control_input'][0].shape[3])
                if unconditional_conditioning is not None:
                    unconditional_conditioning['c_concat'][0] = deform_a_little(img=unconditional_conditioning['c_concat'][0],
                                        delta_theta=delta_step,
                                        delta_phi=phis_steps[i],
                                        pano_height=unconditional_conditioning['c_concat'][0].shape[2],
                                        pano_width=unconditional_conditioning['c_concat'][0].shape[3])
                    unconditional_conditioning['control_input'][0] = deform_a_little(img=unconditional_conditioning['control_input'][0],
                                        delta_theta=delta_step,
                                        delta_phi=phis_steps[i],
                                        pano_height=unconditional_conditioning['control_input'][0].shape[2],
                                        pano_width=unconditional_conditioning['control_input'][0].shape[3])
                
            else:
                '''img = deform_a_little(img=img,
                                    delta_theta=torch.zeros_like(delta_step),
                                    delta_phi=torch.zeros_like(phis_steps[i]),
                                    pano_height=img.shape[2],
                                    pano_width=img.shape[3])
                cond['c_concat'][0] = deform_a_little(img=cond['c_concat'][0],
                                    delta_theta=torch.zeros_like(delta_step),
                                    delta_phi=torch.zeros_like(phis_steps[i]),
                                    pano_height=cond['c_concat'][0].shape[2],
                                    pano_width=cond['c_concat'][0].shape[3])
                cond['control_input'][0] = deform_a_little(img=cond['control_input'][0],
                                    delta_theta=torch.zeros_like(delta_step),
                                    delta_phi=torch.zeros_like(phis_steps[i]),
                                    pano_height=cond['control_input'][0].shape[2],
                                    pano_width=cond['control_input'][0].shape[3])'''
                a = 1

            if self.model.padding_augment:
                # The padding happens in the loop!
                # TODO:
                # maybe the condition do not need to be padded everytime,
                # since they did not really change.
                pad_length_half_lr = img.shape[-1]//4
                pad_length_half_hr = pad_length_half_lr * 8
                img = self.model.rolling_padding(img, pad_length_half_lr)
                cond['c_concat'][0] = self.model.rolling_padding(cond['c_concat'][0], pad_length_half_lr)
                cond['control_input'][0] = self.model.rolling_padding(cond['control_input'][0], pad_length_half_hr)
                if unconditional_conditioning is not None:
                    unconditional_conditioning['c_concat'][0] = self.model.rolling_padding(unconditional_conditioning['c_concat'][0], pad_length_half_lr)
                    unconditional_conditioning['control_input'][0] = self.model.rolling_padding(unconditional_conditioning['control_input'][0], pad_length_half_hr)
            
            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,
                                      dynamic_threshold=dynamic_threshold)
            img, pred_x0 = outs
            if self.model.padding_augment:
                img = img[..., pad_length_half_lr:-pad_length_half_lr]
                cond['c_concat'][0] = cond['c_concat'][0][..., pad_length_half_lr:-pad_length_half_lr]
                cond['control_input'][0] = cond['control_input'][0][..., pad_length_half_hr:-pad_length_half_hr]
                if unconditional_conditioning is not None:
                    unconditional_conditioning['c_concat'][0] = unconditional_conditioning['c_concat'][0][..., pad_length_half_lr:-pad_length_half_lr]
                    unconditional_conditioning['control_input'][0] = unconditional_conditioning['control_input'][0][..., pad_length_half_hr:-pad_length_half_hr]
            
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        # DEBUG (wjh)
        # img = torch.roll(img, -offset_pixels_lr, dims=-1)
        # cond['c_concat'][0] = torch.roll(cond['c_concat'][0], -offset_pixels_lr, dims=-1)
        # cond['control_input'][0] = torch.roll(cond['control_input'][0], -offset_pixels_hr, dims=-1)
        
        return img, intermediates
    
    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      dynamic_threshold=None):
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            model_output = self.model.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            if isinstance(c, dict):
                assert isinstance(unconditional_conditioning, dict)
                c_in = dict()
                for k in c:
                    if isinstance(c[k], list):
                        c_in[k] = [torch.cat([
                            unconditional_conditioning[k][i],
                            c[k][i]]) for i in range(len(c[k]))]
                    else:
                        c_in[k] = torch.cat([
                                unconditional_conditioning[k],
                                c[k]])
            elif isinstance(c, list):
                c_in = list()
                assert isinstance(unconditional_conditioning, list)
                for i in range(len(c)):
                    c_in.append(torch.cat([unconditional_conditioning[i], c[i]]))
            else:
                c_in = torch.cat([unconditional_conditioning, c])
            model_uncond, model_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)

        if self.model.parameterization == "v":
            e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
        else:
            e_t = model_output

        if score_corrector is not None:
            assert self.model.parameterization == "eps", 'not implemented'
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        if self.model.parameterization != "v":
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        else:
            pred_x0 = self.model.predict_start_from_z_and_v(x, t, model_output)

        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

        if dynamic_threshold is not None:
            raise NotImplementedError()

        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0
