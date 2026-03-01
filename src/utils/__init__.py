import torch

import mitsuba as mi

def save_latents_as_image(sd, latents, filename):
    image: torch.Tensor = sd.decode_latents(latents)
    #image = sd.pipe.image_processor.postprocess(image, output_type='pt', do_denormalize=[False])
    image = image.squeeze(0).permute(1, 2, 0)
    mi.util.write_bitmap(filename, image)


def get_index_for_timestep(timesteps, t):
    for i, timestep in enumerate(timesteps):
        if timestep <= t:
            return i
    return len(timesteps) - 1


def hdr_to_sdr(img, exposure=1.0):
    # 1. Apply exposure
    img = img * exposure
    
    # 2. Tone mapping (ACES)
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    img = (img*(a*img+b)) / (img*(c*img+d)+e)
    
    # 3. Clamp
    img = torch.clamp(img, 0, 1)
    
    # 4. Gamma correction
    img = torch.where(
        img <= 0.0031308,
        12.92 * img,
        1.055 * torch.pow(img, 1/2.4) - 0.055
    )
    
    return img