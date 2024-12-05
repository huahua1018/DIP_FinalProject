
import torch
import matplotlib.pyplot as plt

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, steps, steps)
    alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    #---------------------------------------------------
    # return torch.clip(betas, 0, 0.999)
    return torch.clip(betas*0.4, 0, 0.4) # TODO: change to 0.8?
    #---------------------------------------------------

betas=cosine_beta_schedule(200, s = 0.008)

# 繪製趨勢圖
plt.figure(figsize=(8, 6))
plt.plot(betas.numpy(), label='Cosine Beta Schedule')
plt.title('Cosine Beta Schedule Trend')
plt.xlabel('Timesteps')
plt.ylabel('Beta Values')
plt.grid(True)
plt.legend()
plt.savefig('Cosine_Beta_Schedule_Trend.png')