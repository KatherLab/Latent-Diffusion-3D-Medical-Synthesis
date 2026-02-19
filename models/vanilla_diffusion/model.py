from guided_diffusion.unet import UNetModel
import torch
from guided_diffusion.gaussian_diffusion import get_named_beta_schedule, ModelMeanType, ModelVarType,LossType
from guided_diffusion.respace import SpacedDiffusion, space_timesteps
from guided_diffusion.resample import UniformSampler


class DiffUNet(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model = UNetModel(image_size=96, in_channels=4,
                  model_channels=96, out_channels=2, dims=3,
                  num_res_blocks=1, attention_resolutions=[32,16,8],
                  channel_mult=[1, 2, 2, 2])
        
        betas = get_named_beta_schedule("linear", 1000)

        self.diffusion = SpacedDiffusion(use_timesteps=space_timesteps(1000, [1000]),
                                            betas=betas,
                                            model_mean_type=ModelMeanType.START_X,
                                            model_var_type=ModelVarType.FIXED_LARGE,
                                            loss_type=LossType.MSE,
                                            )

        self.sample_diffusion = SpacedDiffusion(use_timesteps=space_timesteps(1000, [50]),
                                            betas=betas,
                                            model_mean_type=ModelMeanType.START_X,
                                            model_var_type=ModelVarType.FIXED_LARGE,
                                            loss_type=LossType.MSE,
                                            )
        self.sampler = UniformSampler(1000)


    def forward(self, x, image=None, pred_type="denoise", step=None):
        if pred_type == "q_sample":
            noise = torch.randn_like(x).to(x.device)
            t, weight = self.sampler.sample(x.shape[0], x.device)
            return self.diffusion.q_sample(x, t, noise=noise), t, noise

        elif pred_type == "denoise":
            return self.model(x, step, image=image)

        elif pred_type == "ddim_sample":
            b = image.shape[0]
            sample_out = self.sample_diffusion.ddim_sample_loop(self.model, (b, 2, 96, 96, 96), model_kwargs={"image": image})
            
            sample_out = sample_out["pred_xstart"]

            return sample_out

