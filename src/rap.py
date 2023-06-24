# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
from PIL import Image
#import utils
import numpy as np
import torch
from torch_utils import persistence
from torch_utils import misc
from torch_utils.ops import upfirdn2d
from torch_utils.ops import grid_sample_gradfix
import matplotlib.pyplot as plt
#from autoaugment import RandAugment


#----------------------------------------------------------------------------
# Coefficients of various wavelet decomposition low-pass filters.

wavelets = {
    'haar': [0.7071067811865476, 0.7071067811865476],
    'db1':  [0.7071067811865476, 0.7071067811865476],
    'db2':  [-0.12940952255092145, 0.22414386804185735, 0.836516303737469, 0.48296291314469025],
    'db3':  [0.035226291882100656, -0.08544127388224149, -0.13501102001039084, 0.4598775021193313, 0.8068915093133388, 0.3326705529509569],
    'db4':  [-0.010597401784997278, 0.032883011666982945, 0.030841381835986965, -0.18703481171888114, -0.02798376941698385, 0.6308807679295904, 0.7148465705525415, 0.23037781330885523],
    'db5':  [0.003335725285001549, -0.012580751999015526, -0.006241490213011705, 0.07757149384006515, -0.03224486958502952, -0.24229488706619015, 0.13842814590110342, 0.7243085284385744, 0.6038292697974729, 0.160102397974125],
    'db6':  [-0.00107730108499558, 0.004777257511010651, 0.0005538422009938016, -0.031582039318031156, 0.02752286553001629, 0.09750160558707936, -0.12976686756709563, -0.22626469396516913, 0.3152503517092432, 0.7511339080215775, 0.4946238903983854, 0.11154074335008017],
    'db7':  [0.0003537138000010399, -0.0018016407039998328, 0.00042957797300470274, 0.012550998556013784, -0.01657454163101562, -0.03802993693503463, 0.0806126091510659, 0.07130921926705004, -0.22403618499416572, -0.14390600392910627, 0.4697822874053586, 0.7291320908465551, 0.39653931948230575, 0.07785205408506236],
    'db8':  [-0.00011747678400228192, 0.0006754494059985568, -0.0003917403729959771, -0.00487035299301066, 0.008746094047015655, 0.013981027917015516, -0.04408825393106472, -0.01736930100202211, 0.128747426620186, 0.00047248457399797254, -0.2840155429624281, -0.015829105256023893, 0.5853546836548691, 0.6756307362980128, 0.3128715909144659, 0.05441584224308161],
    'sym2': [-0.12940952255092145, 0.22414386804185735, 0.836516303737469, 0.48296291314469025],
    'sym3': [0.035226291882100656, -0.08544127388224149, -0.13501102001039084, 0.4598775021193313, 0.8068915093133388, 0.3326705529509569],
    'sym4': [-0.07576571478927333, -0.02963552764599851, 0.49761866763201545, 0.8037387518059161, 0.29785779560527736, -0.09921954357684722, -0.012603967262037833, 0.0322231006040427],
    'sym5': [0.027333068345077982, 0.029519490925774643, -0.039134249302383094, 0.1993975339773936, 0.7234076904024206, 0.6339789634582119, 0.01660210576452232, -0.17532808990845047, -0.021101834024758855, 0.019538882735286728],
    'sym6': [0.015404109327027373, 0.0034907120842174702, -0.11799011114819057, -0.048311742585633, 0.4910559419267466, 0.787641141030194, 0.3379294217276218, -0.07263752278646252, -0.021060292512300564, 0.04472490177066578, 0.0017677118642428036, -0.007800708325034148],
    'sym7': [0.002681814568257878, -0.0010473848886829163, -0.01263630340325193, 0.03051551316596357, 0.0678926935013727, -0.049552834937127255, 0.017441255086855827, 0.5361019170917628, 0.767764317003164, 0.2886296317515146, -0.14004724044296152, -0.10780823770381774, 0.004010244871533663, 0.010268176708511255],
    'sym8': [-0.0033824159510061256, -0.0005421323317911481, 0.03169508781149298, 0.007607487324917605, -0.1432942383508097, -0.061273359067658524, 0.4813596512583722, 0.7771857517005235, 0.3644418948353314, -0.05194583810770904, -0.027219029917056003, 0.049137179673607506, 0.003808752013890615, -0.01495225833704823, -0.0003029205147213668, 0.0018899503327594609],
}

#----------------------------------------------------------------------------
# Helpers for constructing transformation matrices.

def matrix(*rows, device=None):
    assert all(len(row) == len(rows[0]) for row in rows)
    elems = [x for row in rows for x in row]
    ref = [x for x in elems if isinstance(x, torch.Tensor)]
    if len(ref) == 0:
        return misc.constant(np.asarray(rows), device=device)
    assert device is None or device == ref[0].device
    elems = [x if isinstance(x, torch.Tensor) else misc.constant(x, shape=ref[0].shape, device=ref[0].device) for x in elems]
    return torch.stack(elems, dim=-1).reshape(ref[0].shape + (len(rows), -1))

def translate2d(tx, ty, **kwargs):
    return matrix(
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1],
        **kwargs)

def translate3d(tx, ty, tz, **kwargs):
    return matrix(
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1],
        **kwargs)

def scale2d(sx, sy, **kwargs):
    return matrix(
        [sx, 0,  0],
        [0,  sy, 0],
        [0,  0,  1],
        **kwargs)

def scale3d(sx, sy, sz, **kwargs):
    return matrix(
        [sx, 0,  0,  0],
        [0,  sy, 0,  0],
        [0,  0,  sz, 0],
        [0,  0,  0,  1],
        **kwargs)
    
def shear(sx,sy,**kwargs):
    return matrix(
        [1, -sx,  0],
        [-sy,  1, 0],
        [0,  0,  1],
        **kwargs)

def transpose(sx,sy,**kwargs):
    return matrix(
        [sx, sy,  0],
        [sy,  sx, 0],
        [0,  0,  1],
        **kwargs)
    
def secondary_flip(t,**kwargs):
    return matrix(
        [1-t, -t,  t],
        [-t,  1-t, t],
        [0,  0,  1],
        **kwargs)

def rotate2d(theta, **kwargs):
    return matrix(
        [torch.cos(theta), torch.sin(-theta), 0],
        [torch.sin(theta), torch.cos(theta),  0],
        [0,                0,                 1],
        **kwargs)

def rotate3d(v, theta, **kwargs):
    vx = v[..., 0]; vy = v[..., 1]; vz = v[..., 2]
    s = torch.sin(theta); c = torch.cos(theta); cc = 1 - c
    return matrix(
        [vx*vx*cc+c,    vx*vy*cc-vz*s, vx*vz*cc+vy*s, 0],
        [vy*vx*cc+vz*s, vy*vy*cc+c,    vy*vz*cc-vx*s, 0],
        [vz*vx*cc-vy*s, vz*vy*cc+vx*s, vz*vz*cc+c,    0],
        [0,             0,             0,             1],
        **kwargs)

def translate2d_inv(tx, ty, **kwargs):
    return translate2d(-tx, -ty, **kwargs)

def scale2d_inv(sx, sy, **kwargs):
    return scale2d(1 / sx, 1 / sy, **kwargs)

def rotate2d_inv(theta, **kwargs):
    return rotate2d(-theta, **kwargs)

#----------------------------------------------------------------------------
# Versatile image augmentation pipeline from the paper
# "Training Generative Adversarial Networks with Limited Data".
#
# All augmentations are disabled by default; individual augmentations can
# be enabled by setting their probability multipliers to 1.
@persistence.persistent_class
class AugmentPipe(torch.nn.Module):
    def __init__(self,
        xflip=0, rotate90=0, xint=0, xint_max=0.125,
        scale=0, rotate=0, aniso=0, xfrac=0, scale_std=0.2, rotate_max=1, aniso_std=0.2, xfrac_std=0.125,
        brightness=0, contrast=0, lumaflip=0, hue=0, saturation=0, brightness_std=0.2, contrast_std=0.5, hue_max=1, saturation_std=1,
        imgfilter=0, imgfilter_bands=[1,1,1,1], imgfilter_std=1,
        noise=0, cutout=0, noise_std=0.1, cutout_size=0.5,
    ):
        super().__init__()
        self.register_buffer('p', torch.ones([]))       # Overall multiplier for augmentation probability.

        # Pixel blitting.
        self.xflip            = float(xflip)            # Probability multiplier for x-flip.
        self.rotate90         = float(rotate90)         # Probability multiplier for 90 degree rotations.
        self.xint             = float(xint)             # Probability multiplier for integer translation.
        self.xint_max         = float(xint_max)         # Range of integer translation, relative to image dimensions.

        # General geometric transformations.
        self.scale            = float(scale)            # Probability multiplier for isotropic scaling.
        self.rotate           = float(rotate)           # Probability multiplier for arbitrary rotation.
        self.aniso            = float(aniso)            # Probability multiplier for anisotropic scaling.
        self.xfrac            = float(xfrac)            # Probability multiplier for fractional translation.
        self.scale_std        = float(scale_std)        # Log2 standard deviation of isotropic scaling.
        self.rotate_max       = float(rotate_max)       # Range of arbitrary rotation, 1 = full circle.
        self.aniso_std        = float(aniso_std)        # Log2 standard deviation of anisotropic scaling.
        self.xfrac_std        = float(xfrac_std)        # Standard deviation of frational translation, relative to image dimensions.

        # Color transformations.
        self.brightness       = float(brightness)       # Probability multiplier for brightness.
        self.contrast         = float(contrast)         # Probability multiplier for contrast.
        self.lumaflip         = float(lumaflip)         # Probability multiplier for luma flip.
        self.hue              = float(hue)              # Probability multiplier for hue rotation.
        self.saturation       = float(saturation)       # Probability multiplier for saturation.
        self.brightness_std   = float(brightness_std)   # Standard deviation of brightness.
        self.contrast_std     = float(contrast_std)     # Log2 standard deviation of contrast.
        self.hue_max          = float(hue_max)          # Range of hue rotation, 1 = full circle.
        self.saturation_std   = float(saturation_std)   # Log2 standard deviation of saturation.

        # Image-space filtering.
        self.imgfilter        = float(imgfilter)        # Probability multiplier for image-space filtering.
        self.imgfilter_bands  = list(imgfilter_bands)   # Probability multipliers for individual frequency bands.
        self.imgfilter_std    = float(imgfilter_std)    # Log2 standard deviation of image-space filter amplification.

        # Image-space corruptions.
        self.noise            = float(noise)            # Probability multiplier for additive RGB noise.
        self.cutout           = float(cutout)           # Probability multiplier for cutout.
        self.noise_std        = float(noise_std)        # Standard deviation of additive RGB noise.
        self.cutout_size      = float(cutout_size)      # Size of the cutout rectangle, relative to image dimensions.

        # Setup orthogonal lowpass filter for geometric augmentations.
        self.register_buffer('Hz_geom', upfirdn2d.setup_filter(wavelets['sym6'],device='cuda:0'))

    def forward(self, images, debug_percentile=None):
        original_shape = images.shape
        images = torch.reshape(images,(images.shape[0],*images.shape[2:]))
        images = torch.permute(images,[0,3,1,2])
        batch_size, num_channels, height, width = images.shape
        device = images.device
        if debug_percentile is not None:
            debug_percentile = torch.as_tensor(debug_percentile, dtype=torch.float32, device=device)

        # -------------------------------------
        # Select parameters for pixel blitting.
        # -------------------------------------

        # Initialize inverse homogeneous 2D transform: G_inv @ pixel_out ==> pixel_in
        I_3 = torch.eye(3, device=device)
        G_inv = I_3

        # Apply x-flip with probability (xflip * strength).
        if self.xflip > 0:
            i = torch.floor(torch.rand([batch_size], device=device) * 2)
            i = torch.where(torch.rand([batch_size], device=device) < self.xflip * self.p, i, torch.zeros_like(i))
            if debug_percentile is not None:
                i = torch.full_like(i, torch.floor(debug_percentile * 2))
            G_inv = G_inv @ scale2d_inv(1 - 2 * i, 1)

        # Apply 90 degree rotations with probability (rotate90 * strength).
        if self.rotate90 > 0:
            i = torch.floor(torch.rand([batch_size], device=device) * 4)
            i = torch.where(torch.rand([batch_size], device=device) < self.rotate90 * self.p, i, torch.zeros_like(i))
            if debug_percentile is not None:
                i = torch.full_like(i, torch.floor(debug_percentile * 4))
            G_inv = G_inv @ rotate2d_inv(-np.pi / 2 * i)

        # Apply integer translation with probability (xint * strength).
        if self.xint > 0:
            t = (torch.rand([batch_size, 2], device=device) * 2 - 1) * self.xint_max
            t = torch.where(torch.rand([batch_size, 1], device=device) < self.xint * self.p, t, torch.zeros_like(t))
            if debug_percentile is not None:
                t = torch.full_like(t, (debug_percentile * 2 - 1) * self.xint_max)
            G_inv = G_inv @ translate2d_inv(torch.round(t[:,0] * width), torch.round(t[:,1] * height))

        # --------------------------------------------------------
        # Select parameters for general geometric transformations.
        # --------------------------------------------------------

        # Apply isotropic scaling with probability (scale * strength).
        if self.scale > 0:
            s = torch.exp2(torch.randn([batch_size], device=device) * self.scale_std)
            s = torch.where(torch.rand([batch_size], device=device) < self.scale * self.p, s, torch.ones_like(s))
            if debug_percentile is not None:
                s = torch.full_like(s, torch.exp2(torch.erfinv(debug_percentile * 2 - 1) * self.scale_std))
            G_inv = G_inv @ scale2d_inv(s, s)

        # Apply pre-rotation with probability p_rot.
        p_rot = 1 - torch.sqrt((1 - self.rotate * self.p).clamp(0, 1)) # P(pre OR post) = p
        if self.rotate > 0:
            theta = (torch.rand([batch_size], device=device) * 2 - 1) * np.pi * self.rotate_max
            theta = torch.where(torch.rand([batch_size], device=device) < p_rot, theta, torch.zeros_like(theta))
            if debug_percentile is not None:
                theta = torch.full_like(theta, (debug_percentile * 2 - 1) * np.pi * self.rotate_max)
            G_inv = G_inv @ rotate2d_inv(-theta) # Before anisotropic scaling.

        # Apply anisotropic scaling with probability (aniso * strength).
        if self.aniso > 0:
            s = torch.exp2(torch.randn([batch_size], device=device) * self.aniso_std)
            s = torch.where(torch.rand([batch_size], device=device) < self.aniso * self.p, s, torch.ones_like(s))
            if debug_percentile is not None:
                s = torch.full_like(s, torch.exp2(torch.erfinv(debug_percentile * 2 - 1) * self.aniso_std))
            G_inv = G_inv @ scale2d_inv(s, 1 / s)

        # Apply post-rotation with probability p_rot.
        if self.rotate > 0:
            theta = (torch.rand([batch_size], device=device) * 2 - 1) * np.pi * self.rotate_max
            theta = torch.where(torch.rand([batch_size], device=device) < p_rot, theta, torch.zeros_like(theta))
            if debug_percentile is not None:
                theta = torch.zeros_like(theta)
            G_inv = G_inv @ rotate2d_inv(-theta) # After anisotropic scaling.

        # Apply fractional translation with probability (xfrac * strength).
        if self.xfrac > 0:
            t = torch.randn([batch_size, 2], device=device) * self.xfrac_std
            t = torch.where(torch.rand([batch_size, 1], device=device) < self.xfrac * self.p, t, torch.zeros_like(t))
            if debug_percentile is not None:
                t = torch.full_like(t, torch.erfinv(debug_percentile * 2 - 1) * self.xfrac_std)
            G_inv = G_inv @ translate2d_inv(t[:,0] * width, t[:,1] * height)

        # ----------------------------------
        # Execute geometric transformations.
        # ----------------------------------

        # Execute if the transform is not identity.
        if G_inv is not I_3:

            # Calculate padding.
            cx = (width - 1) / 2
            cy = (height - 1) / 2
            cp = matrix([-cx, -cy, 1], [cx, -cy, 1], [cx, cy, 1], [-cx, cy, 1], device=device) # [idx, xyz]
            cp = G_inv @ cp.t() # [batch, xyz, idx]
            Hz_pad = self.Hz_geom.shape[0] // 4
            margin = cp[:, :2, :].permute(1, 0, 2).flatten(1) # [xy, batch * idx]
            margin = torch.cat([-margin, margin]).max(dim=1).values # [x0, y0, x1, y1]
            margin = margin + misc.constant([Hz_pad * 2 - cx, Hz_pad * 2 - cy] * 2, device=device)
            margin = margin.max(misc.constant([0, 0] * 2, device=device))
            margin = margin.min(misc.constant([width-1, height-1] * 2, device=device))
            mx0, my0, mx1, my1 = margin.ceil().to(torch.int32)

            # Pad image and adjust origin.
            images = torch.nn.functional.pad(input=images, pad=[mx0,mx1,my0,my1], mode='reflect')
            G_inv = translate2d((mx0 - mx1) / 2, (my0 - my1) / 2) @ G_inv

            # Upsample.
            images = upfirdn2d.upsample2d(x=images, f=self.Hz_geom, up=2)
            G_inv = scale2d(2, 2, device=device) @ G_inv @ scale2d_inv(2, 2, device=device)
            G_inv = translate2d(-0.5, -0.5, device=device) @ G_inv @ translate2d_inv(-0.5, -0.5, device=device)

            # Execute transformation.
            shape = [batch_size, num_channels, (height + Hz_pad * 2) * 2, (width + Hz_pad * 2) * 2]
            G_inv = scale2d(2 / images.shape[3], 2 / images.shape[2], device=device) @ G_inv @ scale2d_inv(2 / shape[3], 2 / shape[2], device=device)
            grid = torch.nn.functional.affine_grid(theta=G_inv[:,:2,:], size=shape, align_corners=False)
            images = grid_sample_gradfix.grid_sample(images, grid)

            # Downsample and crop.
            images = upfirdn2d.downsample2d(x=images, f=self.Hz_geom, down=2, padding=-Hz_pad*2, flip_filter=True)

        # Apply cutout with probability (cutout * strength).
        if self.cutout > 0:
            size = torch.full([batch_size, 2, 1, 1, 1], self.cutout_size, device=device)
            size = torch.where(torch.rand([batch_size, 1, 1, 1, 1], device=device) < self.cutout * self.p, size, torch.zeros_like(size))
            center = torch.rand([batch_size, 2, 1, 1, 1], device=device)
            if debug_percentile is not None:
                size = torch.full_like(size, self.cutout_size)
                center = torch.full_like(center, debug_percentile)
            coord_x = torch.arange(width, device=device).reshape([1, 1, 1, -1])
            coord_y = torch.arange(height, device=device).reshape([1, 1, -1, 1])
            mask_x = (((coord_x + 0.5) / width - center[:, 0]).abs() >= size[:, 0] / 2)
            mask_y = (((coord_y + 0.5) / height - center[:, 1]).abs() >= size[:, 1] / 2)
            mask = torch.logical_or(mask_x, mask_y).to(torch.float32)
            images = images * mask

        images = images.permute((0,2,3,1))
        images = images.view(original_shape)
        return images

#----------------------------------------------------------------------------

def srotateinde_2d(image,angle):
    num_circles = image.shape[-1] // 2
    for circle_idx in range(num_circles):
        vals = np.empty(0,dtype=image.dtype)
        num_pixel_line = (num_circles - circle_idx) * 2 + 1
        num_pixel = (num_pixel_line-1) * 4
        num_move = round(angle / (360/num_pixel))
        if num_move < 1:
            continue
        
        x1 = image[circle_idx,circle_idx:circle_idx + num_pixel_line -1]
        x2 = image[circle_idx:circle_idx + num_pixel_line - 1 ,circle_idx + num_pixel_line-1]
        x3 = image[circle_idx+num_pixel_line-1,circle_idx + num_pixel_line-1:circle_idx:-1]
        x4 = image[circle_idx+num_pixel_line-1:circle_idx:-1,circle_idx]
        
        vals = np.concatenate((vals,x1,x2,x3,x4))
        vals = np.roll(vals,num_move)
        
        image[circle_idx,circle_idx:circle_idx + num_pixel_line -1] = vals[:num_pixel_line-1]
        image[circle_idx:circle_idx + num_pixel_line - 1 ,circle_idx + num_pixel_line-1] = vals[num_pixel_line-1:num_pixel_line*2-2]
        image[circle_idx+num_pixel_line-1,circle_idx + num_pixel_line-1:circle_idx:-1] = vals[num_pixel_line*2-2:num_pixel_line*3-3]
        image[circle_idx+num_pixel_line-1:circle_idx:-1,circle_idx] = vals[num_pixel_line*3-3:]
        



# def srotateinde(image,angle,inde=1):
#     angle = int(angle.cpu())
#     inde = int(inde.cpu())
#     image_cp = image.cpu().numpy()
#     num_bands = image.shape[0]
#     for band_idx in range(num_bands):
#         srotateinde_2d(image_cp[band_idx],angle)
#         angle += inde * 15
    
#     return torch.from_numpy(image_cp).to(image.device)


    
    
srotate_dict = {}
def srotate(image,angle,device,num_circles):
    global srotate_dict
    patch_size = image.shape[-1]
    angle = int(angle)
    if patch_size in srotate_dict:
        if angle in srotate_dict[patch_size]:
            image[:,srotate_dict[patch_size][angle][0],srotate_dict[patch_size][angle][1]] = image[:,srotate_dict[patch_size][angle][2],srotate_dict[patch_size][angle][3]]
            return
    else:
        srotate_dict[patch_size] = {}
    
    idx = torch.empty(0,device=device,dtype=torch.long)
    idy = torch.empty(0,device=device,dtype=torch.long)
    r_idx = torch.empty(0,device=device,dtype=torch.long)
    r_idy = torch.empty(0,device=device,dtype=torch.long)
    for circle_idx in range(num_circles):
        num_pixel_line = (num_circles - circle_idx) * 2 
        num_pixel = num_pixel_line * 4
        num_move = round(angle / (360/num_pixel))
        #new_num_move = round(angle*((num_circles*2+1)-1-2*circle_idx)/90)
        # if num_move  != new_num_move:
        #     logging.info('asdgsdgasd')
        if num_move < 1:
            continue
        
        ix = torch.cat((torch.ones(num_pixel_line,device=device,dtype=torch.long)*circle_idx,
                        torch.arange(circle_idx,circle_idx + num_pixel_line,device=device,dtype=torch.long),
                        torch.ones(num_pixel_line,device=device,dtype=torch.long)*(circle_idx + num_pixel_line),
                        torch.arange(circle_idx+num_pixel_line,circle_idx,-1,device=device,dtype=torch.long)))
        
        iy = torch.cat((torch.arange(circle_idx,circle_idx + num_pixel_line,device=device,dtype=torch.long),
                        torch.ones(num_pixel_line,device=device,dtype=torch.long)*(circle_idx + num_pixel_line),
                        torch.arange(circle_idx + num_pixel_line,circle_idx,-1,device=device,dtype=torch.long),
                        torch.ones(num_pixel_line,device=device,dtype=torch.long)*circle_idx))
        
        r_ix = torch.roll(ix,num_move)
        r_iy = torch.roll(iy,num_move)
        
        idx = torch.cat((idx,ix))
        idy = torch.cat((idy,iy))
        r_idx = torch.cat((r_idx,r_ix))
        r_idy = torch.cat((r_idy,r_iy))
        
    image[:,idx,idy] = image[:,r_idx,r_idy]
    srotate_dict[patch_size][angle] = [idx,idy,r_idx,r_idy]
    
    
def srotateinde(image,angle,device,num_circles,num_band,inde):
    angle = int(angle)
    inde = int(inde)
    idx = torch.empty(0,device=device,dtype=torch.long)
    idy = torch.empty(0,device=device,dtype=torch.long)
    r_idx = torch.empty(0,device=device,dtype=torch.long)
    r_idy = torch.empty(0,device=device,dtype=torch.long)
    for band in range(num_band):
        for circle_idx in range(num_circles):
            num_pixel_line = (num_circles - circle_idx) * 2 
            num_pixel = num_pixel_line * 4
            num_move = round(angle / (360/num_pixel))
            if num_move < 1:
                continue
            
            ix = torch.cat((torch.ones(num_pixel_line,device=device,dtype=torch.long)*circle_idx,
                            torch.arange(circle_idx,circle_idx + num_pixel_line,device=device,dtype=torch.long),
                            torch.ones(num_pixel_line,device=device,dtype=torch.long)*(circle_idx + num_pixel_line),
                            torch.arange(circle_idx+num_pixel_line,circle_idx,-1,device=device,dtype=torch.long)))
            
            iy = torch.cat((torch.arange(circle_idx,circle_idx + num_pixel_line,device=device,dtype=torch.long),
                            torch.ones(num_pixel_line,device=device,dtype=torch.long)*(circle_idx + num_pixel_line),
                            torch.arange(circle_idx + num_pixel_line,circle_idx,-1,device=device,dtype=torch.long),
                            torch.ones(num_pixel_line,device=device,dtype=torch.long)*circle_idx))
            
            r_ix = torch.roll(ix,num_move)
            r_iy = torch.roll(iy,num_move)
            
            idx = torch.cat((idx,ix))
            idy = torch.cat((idy,iy))
            r_idx = torch.cat((r_idx,r_ix))
            r_idy = torch.cat((r_idy,r_iy))
        
        image[band,idx,idy] = image[band,r_idx,r_idy]
        angle += inde * 15

class RandomAugmentPipe(torch.nn.Module):
    def __init__(self,num_channels,patch_size,device, augs:list,augargs:dict) -> None:
        super().__init__()
        self.num_channels, self.height, self.width = num_channels,patch_size,patch_size
        self.device = device
        self.augs = np.array(augs)
        self.aug_len = len(augs)
        self.xint_max = torch.as_tensor(augargs['xint_max'],device=device)
        self.scale_std = torch.as_tensor(augargs['scale_std'],device=device)
        self.rotate_max = torch.ones((1,),device=device)
        self.aniso_std = torch.as_tensor(augargs['aniso_std'],device=device)
        self.xfrac_std = torch.as_tensor(augargs['xfrac_std'],device=device)
        self.shear_max = torch.as_tensor(augargs['shear_max'],device=device)
        self.cutout_size = torch.as_tensor(augargs['cutout_size'],device=device)
        self.register_buffer('Hz_geom',upfirdn2d.setup_filter(wavelets['haar'],device=self.device))
        #self.Hz_gemo = None
        
    def forward(self,images,debug=False):
        if debug:
            augs = self.augs
        else:
            num_aug = torch.randint(0,self.aug_len+1,(1,))
            if num_aug.item() == 0:
                return images
            aug_idx = torch.randperm(self.aug_len)[:num_aug]
            augs = self.augs[aug_idx]
            if len(aug_idx) == 1:
                augs = [augs]
        
        original_shape = images.shape
        batch_size = images.shape[0]
        images = images.view(batch_size,*images.shape[2:])
        images = images.permute((0,3,1,2))
        I_3 = torch.eye(3, device=self.device)
        G_inv = I_3
        if 'xflip' in augs:
            i = torch.floor(torch.rand([batch_size], device=self.device) * 2)
            #i = torch.where(torch.rand([batch_size], device=self.device) < torch.rand(1,device=self.device) , i, torch.zeros_like(i))
            if debug:
                i = torch.ones([batch_size],device=self.device)
            G_inv = G_inv @ scale2d_inv(1 - 2 * i, 1)
        
        if 'yflip' in augs:
            i = torch.floor(torch.rand([batch_size], device=self.device) * 2)
            #i = torch.where(torch.rand([batch_size], device=self.device) < torch.rand(1,device=self.device), i, torch.zeros_like(i))
            if debug:
                i = torch.ones([batch_size],device=self.device)
            G_inv = G_inv @ scale2d_inv(1, 1 - 2 * i)
            
        if 'transpose' in augs:
            i = torch.floor(torch.rand([batch_size,2], device=self.device) * 2)
            if debug:
                i = torch.ones([batch_size,2],device=self.device)
            #i = torch.where(torch.rand([batch_size,1], device=self.device) < torch.rand(1,device=self.device), i, torch.zeros_like(i))
            i[:,1] = torch.where(i[:,0] == 0,1,0)
            G_inv = G_inv @ transpose(i[:,1],i[:,0])
        
        if 'sflip' in augs:
            t = torch.floor(torch.rand([batch_size], device=self.device)*2)
            if debug:
                t = torch.ones([batch_size], device=self.device)
            G_inv = G_inv @ secondary_flip(t)
        
        if 'rotate90' in augs:
            i = torch.floor(torch.rand([batch_size], device=self.device) * 4)
            #i = torch.where(torch.rand([batch_size], device=self.device) < torch.rand(1,device=self.device), i, torch.zeros_like(i))
            if debug:
                #i = torch.floor(torch.rand([batch_size], device=self.device) * 3) + 1
                i = torch.ones([batch_size], device=self.device) * 2
            G_inv = G_inv @ rotate2d_inv(-np.pi / 2 * i)    

        if 'xint' in augs:
            t = (torch.rand([batch_size, 2], device=self.device) * 2 - 1) * self.xint_max
            #t = torch.where(torch.rand([batch_size, 1], device=self.device) < torch.rand(1,device=self.device), t, torch.zeros_like(t))
            if debug:
                t = (2 * torch.floor(torch.rand([batch_size, 2], device=self.device) * 2) - 1) * self.xint_max
            G_inv = G_inv @ translate2d_inv(torch.round(t[:,0] * self.width), torch.round(t[:,1] * self.height))
        
        
        if 'scale' in augs:
            s = torch.exp2(torch.randn([batch_size], device=self.device) * self.scale_std)
            #s = torch.where(torch.rand([batch_size], device=self.device) < torch.rand(1,device=self.device), s, torch.ones_like(s))
            if debug:
                s = torch.exp2(torch.ones([batch_size], device=self.device) * self.scale_std)
            G_inv = G_inv @ scale2d_inv(s, s)
        
        p_rot = 1 - torch.sqrt((1 - torch.rand(1,device=self.device)).clamp(0, 1)) # P(pre OR post) = p
        if 'rotate' in augs:
            theta = (torch.rand([batch_size], device=self.device) * 2 - 1) * np.pi * self.rotate_max
            #theta = torch.where(torch.rand([batch_size], device=self.device) < p_rot, theta, torch.zeros_like(theta))
            if debug:
                theta = (torch.rand([batch_size], device=self.device) * 2 - 1) * np.pi * 0.9
            G_inv = G_inv @ rotate2d_inv(theta) # Before anisotropic scaling.
            
        if 'aniso' in augs:
            s = torch.exp2(torch.randn([batch_size], device=self.device) * self.aniso_std)
            #s = torch.where(torch.rand([batch_size], device=self.device) < torch.rand(1,device=self.device), s, torch.ones_like(s))
            if debug:
                s = torch.exp2(torch.ones([batch_size], device=self.device) * self.aniso_std)
            G_inv = G_inv @ scale2d_inv(s, 1 / s)

        # Apply post-rotation with probability p_rot.
        if 'rotate' in augs:
            theta = (torch.rand([batch_size], device=self.device) * 2 - 1) * np.pi * self.rotate_max
            #theta = torch.where(torch.rand([batch_size], device=self.device) < p_rot, theta, torch.zeros_like(theta))
            if debug:
                theta = (torch.rand([batch_size], device=self.device) * 2 - 1) * np.pi * 0.9
            G_inv = G_inv @ rotate2d_inv(theta) # After anisotropic scaling.

        if 'xfrac' in augs:
            t = torch.randn([batch_size, 2], device=self.device) * self.xfrac_std
            #t = torch.where(torch.rand([batch_size, 1], device=self.device) < torch.rand(1,device=self.device), t, torch.zeros_like(t))
            if debug:
                t = torch.ones([batch_size, 2], device=self.device) * self.xfrac_std
            G_inv = G_inv @ translate2d_inv(t[:,0] * self.width, t[:,1] * self.height)
            
        if 'shearx' in augs:
            t = (torch.rand(batch_size, device=self.device) * 2 - 1) * self.shear_max
            #t = torch.where(torch.rand(batch_size, device=self.device) < torch.rand(1,device=self.device), t, torch.zeros_like(t))
            if debug:
                t = (2 * torch.floor(torch.rand(batch_size, device=self.device) * 2) - 1) * self.shear_max
            G_inv = G_inv @ shear(t,0)
            
        if 'sheary' in augs:
            t = (torch.rand(batch_size, device=self.device) * 2 - 1) * self.shear_max
            #t = torch.where(torch.rand(batch_size, device=self.device) < torch.rand(1,device=self.device), t, torch.zeros_like(t))
            if debug:
                t = (2 * torch.floor(torch.rand(batch_size, device=self.device) * 2) - 1) * self.shear_max
            G_inv = G_inv @ shear(0,t)
        
        if G_inv is not I_3:
            # Calculate padding.
            cx = (self.width - 1) / 2
            cy = (self.height - 1) / 2
            cp = matrix([-cx, -cy, 1], [cx, -cy, 1], [cx, cy, 1], [-cx, cy, 1], device=self.device) # [idx, xyz]
            cp = G_inv @ cp.t() # [batch, xyz, idx]
            if self.Hz_geom is None:
                Hz_pad = 0
            else:
                Hz_pad = self.Hz_geom.shape[0] // 4
            margin = cp[:, :2, :].permute(1, 0, 2).flatten(1) # [xy, batch * idx]
            margin = torch.cat([-margin, margin]).max(dim=1).values # [x0, y0, x1, y1]
            margin = margin + misc.constant([Hz_pad * 2 - cx, Hz_pad * 2 - cy] * 2, device=self.device)
            margin = margin.max(misc.constant([0, 0] * 2, device=self.device))
            margin = margin.min(misc.constant([self.width-1, self.height-1] * 2, device=self.device))
            mx0, my0, mx1, my1 = margin.ceil().to(torch.int32)

            # Pad image and adjust origin.
            images = torch.nn.functional.pad(input=images, pad=[mx0,mx1,my0,my1], mode='reflect')
            G_inv = translate2d((mx0 - mx1) / 2, (my0 - my1) / 2) @ G_inv

            # Upsample.
            images = upfirdn2d.upsample2d(x=images, f=self.Hz_geom, up=2)
            G_inv = scale2d(2, 2, device=self.device) @ G_inv @ scale2d_inv(2, 2, device=self.device)
            G_inv = translate2d(-0.5, -0.5, device=self.device) @ G_inv @ translate2d_inv(-0.5, -0.5, device=self.device)

            # Execute transformation.
            shape = [batch_size, self.num_channels, (self.height + Hz_pad * 2) * 2, (self.width + Hz_pad * 2) * 2]
            G_inv = scale2d(2 / images.shape[3], 2 / images.shape[2], device=self.device) @ G_inv @ scale2d_inv(2 / shape[3], 2 / shape[2], device=self.device)
            grid = torch.nn.functional.affine_grid(theta=G_inv[:,:2,:], size=shape, align_corners=False)
            images = grid_sample_gradfix.grid_sample(images, grid)

            # Downsample and crop.
            images = upfirdn2d.downsample2d(x=images, f=self.Hz_geom, down=2, padding=-Hz_pad*2, flip_filter=True)
        
        if 'srotate' in augs:
            i = torch.floor(torch.rand(batch_size, device=self.device) * 24)
            if debug:
                i = torch.floor(torch.rand(batch_size, device=self.device) * 23) + 1
            num_circles = self.width // 2
            for idx in range(batch_size):
                if i[idx] == 0:
                    continue
                srotate(images[idx],i[idx]*15,self.device,num_circles)
        
        
        # if 'vscale' in augs:
        #     offset = 0.2
        #     t = torch.rand([batch_size],device=self.device) * (offset*2) + (1-offset)
        #     for i in range(batch_size):
        #         images[i] *= t[i]
        
        
        if 'cutout' in augs:
            size = torch.full([batch_size, 2, 1, 1, 1], self.cutout_size, device=self.device)
            #size = torch.where(torch.rand([batch_size, 1, 1, 1, 1], device=self.device) < torch.rand(1,device=self.device), size, torch.zeros_like(size))
            center = torch.rand([batch_size, 2, 1, 1, 1], device=self.device)
            if debug:
                center = torch.full_like(center, 0.5)
            coord_x = torch.arange(self.width, device=self.device).reshape([1, 1, 1, -1])
            coord_y = torch.arange(self.height, device=self.device).reshape([1, 1, -1, 1])
            mask_x = (((coord_x + 0.5) / self.width - center[:, 0]).abs() >= size[:, 0] / 2)
            mask_y = (((coord_y + 0.5) / self.height - center[:, 1]).abs() >= size[:, 1] / 2)
            mask = torch.logical_or(mask_x, mask_y).to(torch.float32)
            images = images * mask
        
        images = images.permute((0,2,3,1))
        images = images.view(original_shape)
        return images


class RandomAugmentPipeRandom(torch.nn.Module):
    def __init__(self,num_channels,patch_size,device, augs:list,augargs:dict) -> None:
        super().__init__()
        self.num_channels, self.height, self.width = num_channels,patch_size,patch_size
        self.device = device
        self.augs = np.array(augs)
        self.aug_len = len(augs)
        self.xint_max = torch.as_tensor(augargs['xint_max'],device=device)
        self.scale_std = torch.as_tensor(augargs['scale_std'],device=device)
        self.rotate_max = torch.ones((1,),device=device)
        self.aniso_std = torch.as_tensor(augargs['aniso_std'],device=device)
        self.xfrac_std = torch.as_tensor(augargs['xfrac_std'],device=device)
        self.shear_max = torch.as_tensor(augargs['shear_max'],device=device)
        self.cutout_size = torch.as_tensor(augargs['cutout_size'],device=device)
        self.register_buffer('Hz_geom',upfirdn2d.setup_filter(wavelets['sym6'],device=self.device))
        self.aug_funcs= [self.aug_cutout,self.aug_srotate,[self.aug_xflip,self.aug_yflip,self.aug_rotate90,self.aug_xint,self.aug_transpose,self.aug_scale,
                                                       self.aug_prerotate,self.aug_postrotate,self.aug_aniso,self.aug_xfrac,self.aug_shearx,self.aug_sheary]]
    
    def aug_cutout(self,images,batch_size):  
        size = torch.full([batch_size, 2, 1, 1, 1], self.cutout_size, device=self.device)
        size = torch.where(torch.rand([batch_size, 1, 1, 1, 1], device=self.device) < torch.rand(1,device=self.device), size, torch.zeros_like(size))
        center = torch.rand([batch_size, 2, 1, 1, 1], device=self.device)
        coord_x = torch.arange(self.width, device=self.device).reshape([1, 1, 1, -1])
        coord_y = torch.arange(self.height, device=self.device).reshape([1, 1, -1, 1])
        mask_x = (((coord_x + 0.5) / self.width - center[:, 0]).abs() >= size[:, 0] / 2)
        mask_y = (((coord_y + 0.5) / self.height - center[:, 1]).abs() >= size[:, 1] / 2)
        mask = torch.logical_or(mask_x, mask_y).to(torch.float32)
        images = images * mask
        return images
    
    def aug_srotate(self, images,batch_size):
        i = torch.floor(torch.rand(batch_size, device=self.device) * 24)
        i = torch.where(torch.rand(batch_size, device=self.device) < torch.rand(1,device=self.device), i, torch.zeros_like(i))
        num_circles = self.width // 2
        for idx in range(batch_size):
            if i[idx] == 0:
                continue
            srotate(images[idx],i[idx]*15,self.device,num_circles)
        return images
    
    def aug_xflip(self,batch_size,G_inv,p_rot):
        i = torch.floor(torch.rand([batch_size], device=self.device) * 2)
        i = torch.where(torch.rand([batch_size], device=self.device) < torch.rand(1,device=self.device) , i, torch.zeros_like(i))
        G_inv = G_inv @ scale2d_inv(1 - 2 * i, 1)
        return G_inv
    
    def aug_yflip(self,batch_size,G_inv,p_rot):
        i = torch.floor(torch.rand([batch_size], device=self.device) * 2)
        i = torch.where(torch.rand([batch_size], device=self.device) < torch.rand(1,device=self.device), i, torch.zeros_like(i))
        G_inv = G_inv @ scale2d_inv(1, 1 - 2 * i)
        return G_inv
    
    def aug_rotate90(self,batch_size,G_inv,p_rot):
        i = torch.floor(torch.rand([batch_size], device=self.device) * 4)
        i = torch.where(torch.rand([batch_size], device=self.device) < torch.rand(1,device=self.device), i, torch.zeros_like(i))
        G_inv = G_inv @ rotate2d_inv(-np.pi / 2 * i)    
        return G_inv
        
    def aug_xint(self,batch_size,G_inv,p_rot):
        t = (torch.rand([batch_size, 2], device=self.device) * 2 - 1) * self.xint_max
        t = torch.where(torch.rand([batch_size, 1], device=self.device) < torch.rand(1,device=self.device), t, torch.zeros_like(t))
        G_inv = G_inv @ translate2d_inv(torch.round(t[:,0] * self.width), torch.round(t[:,1] * self.height))
        return G_inv
        
    def aug_transpose(self,batch_size,G_inv,p_rot):
        i = torch.floor(torch.rand([batch_size,2], device=self.device) * 2)
        i[:,1] = torch.where(i[:,0] == 0,1,0)
        i = torch.where(torch.rand([batch_size,1], device=self.device) < torch.rand(1,device=self.device), i, torch.zeros_like(i))
        G_inv = G_inv @ transpose(i[:,1],i[:,0])
        return G_inv
    
    def aug_scale(self,batch_size,G_inv,p_rot):
        s = torch.exp2(torch.randn([batch_size], device=self.device) * self.scale_std)
        s = torch.where(torch.rand([batch_size], device=self.device) < torch.rand(1,device=self.device), s, torch.ones_like(s))
        G_inv = G_inv @ scale2d_inv(s, s)
        return G_inv
    
    def aug_prerotate(self,batch_size,G_inv,p_rot):
        theta = (torch.rand([batch_size], device=self.device) * 2 - 1) * np.pi * self.rotate_max
        theta = torch.where(torch.rand([batch_size], device=self.device) < p_rot, theta, torch.zeros_like(theta))
        G_inv = G_inv @ rotate2d_inv(-theta) # Before anisotropic scaling.
        return G_inv
    
    def aug_postrotate(self,batch_size,G_inv,p_rot):
        theta = (torch.rand([batch_size], device=self.device) * 2 - 1) * np.pi * self.rotate_max
        theta = torch.where(torch.rand([batch_size], device=self.device) < p_rot, theta, torch.zeros_like(theta))
        G_inv = G_inv @ rotate2d_inv(-theta) # Before anisotropic scaling.
        return G_inv
    
    def aug_aniso(self,batch_size,G_inv,p_rot):
        s = torch.exp2(torch.randn([batch_size], device=self.device) * self.aniso_std)
        s = torch.where(torch.rand([batch_size], device=self.device) < torch.rand(1,device=self.device), s, torch.ones_like(s))
        G_inv = G_inv @ scale2d_inv(s, 1 / s)
        return G_inv
        
    def aug_xfrac(self,batch_size,G_inv,p_rot): 
        t = torch.randn([batch_size, 2], device=self.device) * self.xfrac_std
        t = torch.where(torch.rand([batch_size, 1], device=self.device) < torch.rand(1,device=self.device), t, torch.zeros_like(t))
        G_inv = G_inv @ translate2d_inv(t[:,0] * self.width, t[:,1] * self.height)
        return G_inv
    
    def aug_shearx(self,batch_size,G_inv,p_rot):  
        t = (torch.rand(batch_size, device=self.device) * 2 - 1) * self.shear_max
        t = torch.where(torch.rand(batch_size, device=self.device) < torch.rand(1,device=self.device), t, torch.zeros_like(t))
        G_inv = G_inv @ shear(t,0)
        return G_inv 

    def aug_sheary(self,batch_size,G_inv,p_rot):     
        t = (torch.rand(batch_size, device=self.device) * 2 - 1) * self.shear_max
        t = torch.where(torch.rand(batch_size, device=self.device) < torch.rand(1,device=self.device), t, torch.zeros_like(t))
        G_inv = G_inv @ shear(0,t)
        return G_inv 

    def forward(self,images):
        num_aug = torch.randint(0,self.aug_len+1,(1,))
        if num_aug == 0:
            return images
        aug_idx = torch.randperm(self.aug_len)[:num_aug]
        augs = self.augs[aug_idx]
        
        original_shape = images.shape
        batch_size = images.shape[0]
        images = images.view(batch_size,*images.shape[2:])
        images = images.permute((0,3,1,2))
        
        randp = torch.randperm(3).tolist()
        for idx in randp:
            if str(type(self.aug_funcs[idx])) == "<class 'method'>":
                func = self.aug_funcs[idx]
                func_name = func.__name__
                if func_name.split('_')[1].replace('pre','').replace('post','') in augs:
                    images = func(images,batch_size)
            else:
                I_3 = torch.eye(3, device=self.device)
                G_inv = I_3
                randd = torch.randperm(len(self.aug_funcs[idx])).tolist()
                p_rot = 1 - torch.sqrt((1 - torch.rand(1,device=self.device)).clamp(0, 1)) # P(pre OR post) = p
                for idxd in randd:
                    func = self.aug_funcs[idx][idxd]
                    func_name = func.__name__
                    if func_name.split('_')[1].replace('pre','').replace('post','') in augs:
                        G_inv = func(batch_size,G_inv,p_rot)
                    
                if G_inv is not I_3:
                    # Calculate padding.
                    cx = (self.width - 1) / 2
                    cy = (self.height - 1) / 2
                    cp = matrix([-cx, -cy, 1], [cx, -cy, 1], [cx, cy, 1], [-cx, cy, 1], device=self.device) # [idx, xyz]
                    cp = G_inv @ cp.t() # [batch, xyz, idx]
                    if self.Hz_geom is None:
                        Hz_pad = 0
                    else:
                        Hz_pad = self.Hz_geom.shape[0] // 4
                    margin = cp[:, :2, :].permute(1, 0, 2).flatten(1) # [xy, batch * idx]
                    margin = torch.cat([-margin, margin]).max(dim=1).values # [x0, y0, x1, y1]
                    margin = margin + misc.constant([Hz_pad * 2 - cx, Hz_pad * 2 - cy] * 2, device=self.device)
                    margin = margin.max(misc.constant([0, 0] * 2, device=self.device))
                    margin = margin.min(misc.constant([self.width-1, self.height-1] * 2, device=self.device))
                    mx0, my0, mx1, my1 = margin.ceil().to(torch.int32)

                    # Pad image and adjust origin.
                    images = torch.nn.functional.pad(input=images, pad=[mx0,mx1,my0,my1], mode='reflect')
                    G_inv = translate2d((mx0 - mx1) / 2, (my0 - my1) / 2) @ G_inv
                    #G = G @ translate2d_inv
                    # Upsample.
                    
                    images = upfirdn2d.upsample2d(x=images, f=self.Hz_geom, up=2)
                    G_inv = scale2d(2, 2, device=self.device) @ G_inv @ scale2d_inv(2, 2, device=self.device)
                    G_inv = translate2d(-0.5, -0.5, device=self.device) @ G_inv @ translate2d_inv(-0.5, -0.5, device=self.device)

                    # # Execute transformation.
                    shape = [batch_size, self.num_channels, (self.height + Hz_pad * 2) *2, (self.width + Hz_pad * 2) *2]
                    G_inv = scale2d(2 / images.shape[3], 2 / images.shape[2], device=self.device) @ G_inv @ scale2d_inv(2 / shape[3], 2 / shape[2], device=self.device)
                    grid = torch.nn.functional.affine_grid(theta=G_inv[:,:2,:], size=shape, align_corners=False)
                    images = grid_sample_gradfix.grid_sample(images, grid)
                    #images = torch.nn.functional.grid_sample(input=images, grid=grid, mode='bilinear', padding_mode='zeros', align_corners=False)

                    # Downsample and crop.
                    images = upfirdn2d.downsample2d(x=images, f=self.Hz_geom, down=2, padding=-Hz_pad*2, flip_filter=True)
                
        images = images.permute((0,2,3,1))
        images = images.view(original_shape)
        return images



class RandomAugmentPipeReversed(torch.nn.Module):
    def __init__(self,num_channels,patch_size,device, augs:list,augargs:dict) -> None:
        super().__init__()
        self.num_channels, self.height, self.width = num_channels,patch_size,patch_size
        self.device = device
        self.augs = np.array(augs)
        self.aug_len = len(augs)
        self.xint_max = torch.as_tensor(augargs['xint_max'],device=device)
        self.scale_std = torch.as_tensor(augargs['scale_std'],device=device)
        self.rotate_max = torch.ones((1,),device=device)
        self.aniso_std = torch.as_tensor(augargs['aniso_std'],device=device)
        self.xfrac_std = torch.as_tensor(augargs['xfrac_std'],device=device)
        self.shear_max = torch.as_tensor(augargs['shear_max'],device=device)
        self.cutout_size = torch.as_tensor(augargs['cutout_size'],device=device)
        self.register_buffer('Hz_geom',upfirdn2d.setup_filter(wavelets['sym6'],device=self.device))
        
    def forward(self,images):
        num_aug = torch.randint(0,self.aug_len+1,(1,))
        if num_aug == 0:
            return images
        aug_idx = torch.randperm(self.aug_len)[:num_aug]
        augs = self.augs[aug_idx]
        
        original_shape = images.shape
        batch_size = images.shape[0]
        images = images.view(batch_size,*images.shape[2:])
        images = images.permute((0,3,1,2))
        if 'srotate' in augs:
            i = torch.floor(torch.rand(batch_size, device=self.device) * 24)
            i = torch.where(torch.rand(batch_size, device=self.device) < torch.rand(1,device=self.device), i, torch.zeros_like(i))
            num_circles = self.width // 2
            for idx in range(batch_size):
                if i[idx] == 0:
                    continue
                srotate(images[idx],i[idx]*15,self.device,num_circles)

        if 'cutout' in augs:
            size = torch.full([batch_size, 2, 1, 1, 1], self.cutout_size, device=self.device)
            size = torch.where(torch.rand([batch_size, 1, 1, 1, 1], device=self.device) < torch.rand(1,device=self.device), size, torch.zeros_like(size))
            center = torch.rand([batch_size, 2, 1, 1, 1], device=self.device)
            coord_x = torch.arange(self.width, device=self.device).reshape([1, 1, 1, -1])
            coord_y = torch.arange(self.height, device=self.device).reshape([1, 1, -1, 1])
            mask_x = (((coord_x + 0.5) / self.width - center[:, 0]).abs() >= size[:, 0] / 2)
            mask_y = (((coord_y + 0.5) / self.height - center[:, 1]).abs() >= size[:, 1] / 2)
            mask = torch.logical_or(mask_x, mask_y).to(torch.float32)
            images = images * mask
        

        I_3 = torch.eye(3, device=self.device)
        G_inv = I_3
        if 'sheary' in augs:
            t = (torch.rand(batch_size, device=self.device) * 2 - 1) * self.shear_max
            t = torch.where(torch.rand(batch_size, device=self.device) < torch.rand(1,device=self.device), t, torch.zeros_like(t))
            G_inv = G_inv @ shear(0,t)

        if 'shearx' in augs:
            t = (torch.rand(batch_size, device=self.device) * 2 - 1) * self.shear_max
            t = torch.where(torch.rand(batch_size, device=self.device) < torch.rand(1,device=self.device), t, torch.zeros_like(t))
            G_inv = G_inv @ shear(t,0)

        if 'xfrac' in augs:
            t = torch.randn([batch_size, 2], device=self.device) * self.xfrac_std
            t = torch.where(torch.rand([batch_size, 1], device=self.device) < torch.rand(1,device=self.device), t, torch.zeros_like(t))
            G_inv = G_inv @ translate2d_inv(t[:,0] * self.width, t[:,1] * self.height)

        p_rot = 1 - torch.sqrt((1 - torch.rand(1,device=self.device)).clamp(0, 1)) # P(pre OR post) = p
        if 'rotate' in augs:
            theta = (torch.rand([batch_size], device=self.device) * 2 - 1) * np.pi * self.rotate_max
            theta = torch.where(torch.rand([batch_size], device=self.device) < p_rot, theta, torch.zeros_like(theta))
            G_inv = G_inv @ rotate2d_inv(-theta) # Before anisotropic scaling.
            
        if 'aniso' in augs:
            s = torch.exp2(torch.randn([batch_size], device=self.device) * self.aniso_std)
            s = torch.where(torch.rand([batch_size], device=self.device) < torch.rand(1,device=self.device), s, torch.ones_like(s))
            G_inv = G_inv @ scale2d_inv(s, 1 / s)
            
        # Apply post-rotation with probability p_rot.
        if 'rotate' in augs:
            theta = (torch.rand([batch_size], device=self.device) * 2 - 1) * np.pi * self.rotate_max
            theta = torch.where(torch.rand([batch_size], device=self.device) < p_rot, theta, torch.zeros_like(theta))
            G_inv = G_inv @ rotate2d_inv(-theta) # After anisotropic scaling.

        if 'scale' in augs:
            s = torch.exp2(torch.randn([batch_size], device=self.device) * self.scale_std)
            s = torch.where(torch.rand([batch_size], device=self.device) < torch.rand(1,device=self.device), s, torch.ones_like(s))
            G_inv = G_inv @ scale2d_inv(s, s)
            
        if 'transpose' in augs:
            i = torch.floor(torch.rand([batch_size,2], device=self.device) * 2)
            i[:,1] = torch.where(i[:,0] == 0,1,0)
            i = torch.where(torch.rand([batch_size,1], device=self.device) < torch.rand(1,device=self.device), i, torch.zeros_like(i))
            G_inv = G_inv @ transpose(i[:,1],i[:,0])
            
        if 'xint' in augs:
            t = (torch.rand([batch_size, 2], device=self.device) * 2 - 1) * self.xint_max
            t = torch.where(torch.rand([batch_size, 1], device=self.device) < torch.rand(1,device=self.device), t, torch.zeros_like(t))
            G_inv = G_inv @ translate2d_inv(torch.round(t[:,0] * self.width), torch.round(t[:,1] * self.height))
 
        if 'rotate90' in augs:
            i = torch.floor(torch.rand([batch_size], device=self.device) * 4)
            i = torch.where(torch.rand([batch_size], device=self.device) < torch.rand(1,device=self.device), i, torch.zeros_like(i))
            G_inv = G_inv @ rotate2d_inv(-np.pi / 2 * i)  
 
        if 'yflip' in augs:
            i = torch.floor(torch.rand([batch_size], device=self.device) * 2)
            i = torch.where(torch.rand([batch_size], device=self.device) < torch.rand(1,device=self.device), i, torch.zeros_like(i))
            G_inv = G_inv @ scale2d_inv(1, 1 - 2 * i)
 
        if 'xflip' in augs:
            i = torch.floor(torch.rand([batch_size], device=self.device) * 2)
            i = torch.where(torch.rand([batch_size], device=self.device) < torch.rand(1,device=self.device) , i, torch.zeros_like(i))
            G_inv = G_inv @ scale2d_inv(1 - 2 * i, 1)

        if G_inv is not I_3:
            # Calculate padding.
            cx = (self.width - 1) / 2
            cy = (self.height - 1) / 2
            cp = matrix([-cx, -cy, 1], [cx, -cy, 1], [cx, cy, 1], [-cx, cy, 1], device=self.device) # [idx, xyz]
            cp = G_inv @ cp.t() # [batch, xyz, idx]
            if self.Hz_geom is None:
                Hz_pad = 0
            else:
                Hz_pad = self.Hz_geom.shape[0] // 4
            margin = cp[:, :2, :].permute(1, 0, 2).flatten(1) # [xy, batch * idx]
            margin = torch.cat([-margin, margin]).max(dim=1).values # [x0, y0, x1, y1]
            margin = margin + misc.constant([Hz_pad * 2 - cx, Hz_pad * 2 - cy] * 2, device=self.device)
            margin = margin.max(misc.constant([0, 0] * 2, device=self.device))
            margin = margin.min(misc.constant([self.width-1, self.height-1] * 2, device=self.device))
            mx0, my0, mx1, my1 = margin.ceil().to(torch.int32)

            # Pad image and adjust origin.
            images = torch.nn.functional.pad(input=images, pad=[mx0,mx1,my0,my1], mode='reflect')
            G_inv = translate2d((mx0 - mx1) / 2, (my0 - my1) / 2) @ G_inv
            #G = G @ translate2d_inv
            # Upsample.
            
            images = upfirdn2d.upsample2d(x=images, f=self.Hz_geom, up=2)
            G_inv = scale2d(2, 2, device=self.device) @ G_inv @ scale2d_inv(2, 2, device=self.device)
            G_inv = translate2d(-0.5, -0.5, device=self.device) @ G_inv @ translate2d_inv(-0.5, -0.5, device=self.device)

            # # Execute transformation.
            shape = [batch_size, self.num_channels, (self.height + Hz_pad * 2) *2, (self.width + Hz_pad * 2) *2]
            G_inv = scale2d(2 / images.shape[3], 2 / images.shape[2], device=self.device) @ G_inv @ scale2d_inv(2 / shape[3], 2 / shape[2], device=self.device)
            grid = torch.nn.functional.affine_grid(theta=G_inv[:,:2,:], size=shape, align_corners=False)
            images = grid_sample_gradfix.grid_sample(images, grid)
            #images = torch.nn.functional.grid_sample(input=images, grid=grid, mode='bilinear', padding_mode='zeros', align_corners=False)

            # Downsample and crop.
            images = upfirdn2d.downsample2d(x=images, f=self.Hz_geom, down=2, padding=-Hz_pad*2, flip_filter=True)
            

        images = images.permute((0,2,3,1))
        images = images.view(original_shape)
        return images




class RandomAugmentPipePaper(torch.nn.Module):
    def __init__(self,num_channels,patch_size,device, augs:list,augargs:dict) -> None:
        super().__init__()
        self.num_channels, self.height, self.width = num_channels,patch_size,patch_size
        self.device = device
        self.augs = np.array(augs)
        self.aug_len = len(augs)
        self.xint_max = torch.as_tensor(augargs['xint_max'],device=device)
        self.scale_std = torch.as_tensor(augargs['scale_std'],device=device)
        self.rotate_max = torch.ones((1,),device=device)
        self.aniso_std = torch.as_tensor(augargs['aniso_std'],device=device)
        self.xfrac_std = torch.as_tensor(augargs['xfrac_std'],device=device)
        self.shear_max = torch.as_tensor(augargs['shear_max'],device=device)
        self.cutout_size = torch.as_tensor(augargs['cutout_size'],device=device)
        self.register_buffer('Hz_geom',upfirdn2d.setup_filter(wavelets['sym6'],device=self.device))
        
    def forward(self,images):
        num_aug = torch.randint(0,self.aug_len+1,(1,))
        if num_aug == 0:
            return images
        aug_idx = torch.randperm(self.aug_len)[:num_aug]
        augs = self.augs[aug_idx]
        
        original_shape = images.shape
        batch_size = images.shape[0]
        images = images.view(batch_size,*images.shape[2:])
        images = images.permute((0,3,1,2))
        I_3 = torch.eye(3, device=self.device)
        G_inv = I_3
        if 'xflip' in augs:
            i = torch.floor(torch.rand([batch_size], device=self.device) * 2)
            i = torch.where(torch.rand([batch_size], device=self.device) < torch.rand(1,device=self.device) , i, torch.zeros_like(i))
            G_inv = G_inv @ scale2d(1 - 2 * i, 1)
        
        if 'yflip' in augs:
            i = torch.floor(torch.rand([batch_size], device=self.device) * 2)
            i = torch.where(torch.rand([batch_size], device=self.device) < torch.rand(1,device=self.device), i, torch.zeros_like(i))
            G_inv = G_inv @ scale2d(1, 1 - 2 * i)
        
        if 'rotate90' in augs:
            i = torch.floor(torch.rand([batch_size], device=self.device) * 4)
            i = torch.where(torch.rand([batch_size], device=self.device) < torch.rand(1,device=self.device), i, torch.zeros_like(i))
            G_inv = G_inv @ rotate2d(-np.pi / 2 * i)
        
        if 'xint' in augs:
            t = (torch.rand([batch_size, 2], device=self.device) * 2 - 1) * self.xint_max
            t = torch.where(torch.rand([batch_size, 1], device=self.device) < torch.rand(1,device=self.device), t, torch.zeros_like(t))
            G_inv = G_inv @ translate2d(t[:,0], t[:,1])
        
        if 'transpose' in augs:
            i = torch.floor(torch.rand([batch_size,2], device=self.device) * 2)
            i[:,1] = torch.where(i[:,0] == 0,1,0)
            i = torch.where(torch.rand([batch_size,1], device=self.device) < torch.rand(1,device=self.device), i, torch.zeros_like(i))
            G_inv = G_inv @ transpose(i[:,1],i[:,0])
            
        if 'scale' in augs:
            s = torch.exp2(torch.randn([batch_size], device=self.device) * self.scale_std)
            s = torch.where(torch.rand([batch_size], device=self.device) < torch.rand(1,device=self.device), s, torch.ones_like(s))
            G_inv = G_inv @ scale2d(s, s)
        
        p_rot = 1 - torch.sqrt((1 - torch.rand(1,device=self.device)).clamp(0, 1)) # P(pre OR post) = p
        if 'rotate' in augs:
            theta = (torch.rand([batch_size], device=self.device) * 2 - 1) * np.pi * self.rotate_max
            theta = torch.where(torch.rand([batch_size], device=self.device) < p_rot, theta, torch.zeros_like(theta))
            G_inv = G_inv @ rotate2d(-theta) # Before anisotropic scaling.
            
        if 'aniso' in augs:
            s = torch.exp2(torch.randn([batch_size], device=self.device) * self.aniso_std)
            s = torch.where(torch.rand([batch_size], device=self.device) < torch.rand(1,device=self.device), s, torch.ones_like(s))
            G_inv = G_inv @ scale2d(s, 1 / s)
            
        # Apply post-rotation with probability p_rot.
        if 'rotate' in augs:
            theta = (torch.rand([batch_size], device=self.device) * 2 - 1) * np.pi * self.rotate_max
            theta = torch.where(torch.rand([batch_size], device=self.device) < p_rot, theta, torch.zeros_like(theta))
            G_inv = G_inv @ rotate2d(-theta) # After anisotropic scaling.

        if 'xfrac' in augs:
            t = torch.randn([batch_size, 2], device=self.device) * self.xfrac_std
            t = torch.where(torch.rand([batch_size, 1], device=self.device) < torch.rand(1,device=self.device), t, torch.zeros_like(t))
            G_inv = G_inv @ translate2d(t[:,0] , t[:,1] )
        
        if 'shearx' in augs:
            t = (torch.rand(batch_size, device=self.device) * 2 - 1) * self.shear_max
            t = torch.where(torch.rand(batch_size, device=self.device) < torch.rand(1,device=self.device), t, torch.zeros_like(t))
            G_inv = G_inv @ shear(t,0)
            
        if 'sheary' in augs:
            t = (torch.rand(batch_size, device=self.device) * 2 - 1) * self.shear_max
            t = torch.where(torch.rand(batch_size, device=self.device) < torch.rand(1,device=self.device), t, torch.zeros_like(t))
            G_inv = G_inv @ shear(0,t)
        
        if G_inv is not I_3:
            # Calculate padding.
            cx = (self.width - 1) / 2
            cy = (self.height - 1) / 2
            cp = matrix([-cx, -cy, 1], [cx, -cy, 1], [cx, cy, 1], [-cx, cy, 1], device=self.device) # [idx, xyz]
            cp = G_inv @ cp.t() # [batch, xyz, idx]
            if self.Hz_geom is None:
                Hz_pad = 0
            else:
                Hz_pad = self.Hz_geom.shape[0] // 4
            margin = cp[:, :2, :].permute(1, 0, 2).flatten(1) # [xy, batch * idx]
            margin = torch.cat([-margin, margin]).max(dim=1).values # [x0, y0, x1, y1]
            margin = margin + misc.constant([Hz_pad * 2 - cx, Hz_pad * 2 - cy] * 2, device=self.device)
            margin = margin.max(misc.constant([0, 0] * 2, device=self.device))
            margin = margin.min(misc.constant([self.width-1, self.height-1] * 2, device=self.device))
            mx0, my0, mx1, my1 = margin.ceil().to(torch.int32)

            # Pad image and adjust origin.
            images = torch.nn.functional.pad(input=images, pad=[mx0,mx1,my0,my1], mode='reflect')
            G_inv = translate2d((mx0 - mx1) / 2, (my0 - my1) / 2) @ G_inv
            #G = G @ translate2d_inv
            # Upsample.
            
            images = upfirdn2d.upsample2d(x=images, f=self.Hz_geom, up=2)
            G_inv = scale2d(2, 2, device=self.device) @ G_inv @ scale2d_inv(2, 2, device=self.device)
            G_inv = translate2d(-0.5, -0.5, device=self.device) @ G_inv @ translate2d_inv(-0.5, -0.5, device=self.device)

            # # Execute transformation.
            shape = [batch_size, self.num_channels, (self.height + Hz_pad * 2) *2, (self.width + Hz_pad * 2) *2]
            G_inv = scale2d(2 / images.shape[3], 2 / images.shape[2], device=self.device) @ G_inv @ scale2d_inv(2 / shape[3], 2 / shape[2], device=self.device)
            grid = torch.nn.functional.affine_grid(theta=G_inv[:,:2,:], size=shape, align_corners=False)
            images = grid_sample_gradfix.grid_sample(images, grid)
            #images = torch.nn.functional.grid_sample(input=images, grid=grid, mode='bilinear', padding_mode='zeros', align_corners=False)

            # Downsample and crop.
            images = upfirdn2d.downsample2d(x=images, f=self.Hz_geom, down=2, padding=-Hz_pad*2, flip_filter=True)
            
        if 'cutout' in augs:
            size = torch.full([batch_size, 2, 1, 1, 1], self.cutout_size, device=self.device)
            size = torch.where(torch.rand([batch_size, 1, 1, 1, 1], device=self.device) < torch.rand(1,device=self.device), size, torch.zeros_like(size))
            center = torch.rand([batch_size, 2, 1, 1, 1], device=self.device)
            coord_x = torch.arange(self.width, device=self.device).reshape([1, 1, 1, -1])
            coord_y = torch.arange(self.height, device=self.device).reshape([1, 1, -1, 1])
            mask_x = (((coord_x + 0.5) / self.width - center[:, 0]).abs() >= size[:, 0] / 2)
            mask_y = (((coord_y + 0.5) / self.height - center[:, 1]).abs() >= size[:, 1] / 2)
            mask = torch.logical_or(mask_x, mask_y).to(torch.float32)
            images = images * mask
        
        if 'srotate' in augs:
            i = torch.floor(torch.rand(batch_size, device=self.device) * 24)
            i = torch.where(torch.rand(batch_size, device=self.device) < torch.rand(1,device=self.device), i, torch.zeros_like(i))
            num_circles = self.width // 2
            for idx in range(batch_size):
                if i[idx] == 0:
                    continue
                srotate(images[idx],i[idx]*15,self.device,num_circles)
        
        images = images.permute((0,2,3,1))
        images = images.view(original_shape)
        return images


@persistence.persistent_class
class RandomAugmentPipeADA(torch.nn.Module):
    def __init__(self,
        xflip=0, rotate90=0, xint=0, xint_max=0.125,
        scale=0, rotate=0, aniso=0, xfrac=0, scale_std=0.2, rotate_max=1, aniso_std=0.2, xfrac_std=0.125,
        brightness=0, contrast=0, lumaflip=0, hue=0, saturation=0, brightness_std=0.2, contrast_std=0.5, hue_max=1, saturation_std=1,
        imgfilter=0, imgfilter_bands=[1,1,1,1], imgfilter_std=1,
        noise=0, cutout=0, noise_std=0.1, cutout_size=0.5,
    ):
        super().__init__()
        self.register_buffer('p', torch.ones([]))       # Overall multiplier for augmentation probability.

        # Pixel blitting.
        self.xflip            = float(xflip)            # Probability multiplier for x-flip.
        self.rotate90         = float(rotate90)         # Probability multiplier for 90 degree rotations.
        self.xint             = float(xint)             # Probability multiplier for integer translation.
        self.xint_max         = float(xint_max)         # Range of integer translation, relative to image dimensions.

        # General geometric transformations.
        self.scale            = float(scale)            # Probability multiplier for isotropic scaling.
        self.rotate           = float(rotate)           # Probability multiplier for arbitrary rotation.
        self.aniso            = float(aniso)            # Probability multiplier for anisotropic scaling.
        self.xfrac            = float(xfrac)            # Probability multiplier for fractional translation.
        self.scale_std        = float(scale_std)        # Log2 standard deviation of isotropic scaling.
        self.rotate_max       = float(rotate_max)       # Range of arbitrary rotation, 1 = full circle.
        self.aniso_std        = float(aniso_std)        # Log2 standard deviation of anisotropic scaling.
        self.xfrac_std        = float(xfrac_std)        # Standard deviation of frational translation, relative to image dimensions.

        # Color transformations.
        self.brightness       = float(brightness)       # Probability multiplier for brightness.
        self.contrast         = float(contrast)         # Probability multiplier for contrast.
        self.lumaflip         = float(lumaflip)         # Probability multiplier for luma flip.
        self.hue              = float(hue)              # Probability multiplier for hue rotation.
        self.saturation       = float(saturation)       # Probability multiplier for saturation.
        self.brightness_std   = float(brightness_std)   # Standard deviation of brightness.
        self.contrast_std     = float(contrast_std)     # Log2 standard deviation of contrast.
        self.hue_max          = float(hue_max)          # Range of hue rotation, 1 = full circle.
        self.saturation_std   = float(saturation_std)   # Log2 standard deviation of saturation.

        # Image-space filtering.
        self.imgfilter        = float(imgfilter)        # Probability multiplier for image-space filtering.
        self.imgfilter_bands  = list(imgfilter_bands)   # Probability multipliers for individual frequency bands.
        self.imgfilter_std    = float(imgfilter_std)    # Log2 standard deviation of image-space filter amplification.

        # Image-space corruptions.
        self.noise            = float(noise)            # Probability multiplier for additive RGB noise.
        self.cutout           = float(cutout)           # Probability multiplier for cutout.
        self.noise_std        = float(noise_std)        # Standard deviation of additive RGB noise.
        self.cutout_size      = float(cutout_size)      # Size of the cutout rectangle, relative to image dimensions.

        # Setup orthogonal lowpass filter for geometric augmentations.
        self.register_buffer('Hz_geom', upfirdn2d.setup_filter(wavelets['sym6']))
        
        # Construct filter bank for image-space filtering.
        # Hz_lo = np.asarray(wavelets['sym2'])            # H(z)
        # Hz_hi = Hz_lo * ((-1) ** np.arange(Hz_lo.size)) # H(-z)
        # Hz_lo2 = np.convolve(Hz_lo, Hz_lo[::-1]) / 2    # H(z) * H(z^-1) / 2
        # Hz_hi2 = np.convolve(Hz_hi, Hz_hi[::-1]) / 2    # H(-z) * H(-z^-1) / 2
        # Hz_fbank = np.eye(4, 1)                         # Bandpass(H(z), b_i)
        # for i in range(1, Hz_fbank.shape[0]):
        #     Hz_fbank = np.dstack([Hz_fbank, np.zeros_like(Hz_fbank)]).reshape(Hz_fbank.shape[0], -1)[:, :-1]
        #     Hz_fbank = scipy.signal.convolve(Hz_fbank, [Hz_lo2])
        #     Hz_fbank[i, (Hz_fbank.shape[1] - Hz_hi2.size) // 2 : (Hz_fbank.shape[1] + Hz_hi2.size) // 2] += Hz_hi2
        # self.register_buffer('Hz_fbank', torch.as_tensor(Hz_fbank, dtype=torch.float32))

    def forward(self, images):
        original_shape = images.shape
        images = images.view(images.shape[0],*images.shape[2:])
        images = images.permute((0,3,1,2))
        assert isinstance(images, torch.Tensor) and images.ndim == 4
        batch_size, num_channels, height, width = images.shape
        device = images.device
        self.p = torch.randn((1,),device=device)
        # -------------------------------------
        # Select parameters for pixel blitting.
        # -------------------------------------

        # Initialize inverse homogeneous 2D transform: G_inv @ pixel_out ==> pixel_in
        I_3 = torch.eye(3, device=device)
        G_inv = I_3

        # Apply x-flip with probability (xflip * strength).
        if self.xflip > 0:
            i = torch.floor(torch.rand([batch_size], device=device) * 2)
            i = torch.where(torch.rand([batch_size], device=device) < self.xflip * self.p, i, torch.zeros_like(i))
            G_inv = G_inv @ scale2d_inv(1 - 2 * i, 1)

        # Apply 90 degree rotations with probability (rotate90 * strength).
        if self.rotate90 > 0:
            i = torch.floor(torch.rand([batch_size], device=device) * 4)
            i = torch.where(torch.rand([batch_size], device=device) < self.rotate90 * self.p, i, torch.zeros_like(i))
            G_inv = G_inv @ rotate2d_inv(-np.pi / 2 * i)

        # Apply integer translation with probability (xint * strength).
        if self.xint > 0:
            t = (torch.rand([batch_size, 2], device=device) * 2 - 1) * self.xint_max
            t = torch.where(torch.rand([batch_size, 1], device=device) < self.xint * self.p, t, torch.zeros_like(t))
            G_inv = G_inv @ translate2d_inv(torch.round(t[:,0] * width), torch.round(t[:,1] * height))

        # --------------------------------------------------------
        # Select parameters for general geometric transformations.
        # --------------------------------------------------------

        # Apply isotropic scaling with probability (scale * strength).
        if self.scale > 0:
            s = torch.exp2(torch.randn([batch_size], device=device) * self.scale_std)
            s = torch.where(torch.rand([batch_size], device=device) < self.scale * self.p, s, torch.ones_like(s))
            G_inv = G_inv @ scale2d_inv(s, s)

        # Apply pre-rotation with probability p_rot.
        p_rot = 1 - torch.sqrt((1 - self.rotate * self.p).clamp(0, 1)) # P(pre OR post) = p
        if self.rotate > 0:
            theta = (torch.rand([batch_size], device=device) * 2 - 1) * np.pi * self.rotate_max
            theta = torch.where(torch.rand([batch_size], device=device) < p_rot, theta, torch.zeros_like(theta))
            G_inv = G_inv @ rotate2d_inv(-theta) # Before anisotropic scaling.

        # Apply anisotropic scaling with probability (aniso * strength).
        if self.aniso > 0:
            s = torch.exp2(torch.randn([batch_size], device=device) * self.aniso_std)
            s = torch.where(torch.rand([batch_size], device=device) < self.aniso * self.p, s, torch.ones_like(s))
            G_inv = G_inv @ scale2d_inv(s, 1 / s)

        # Apply post-rotation with probability p_rot.
        if self.rotate > 0:
            theta = (torch.rand([batch_size], device=device) * 2 - 1) * np.pi * self.rotate_max
            theta = torch.where(torch.rand([batch_size], device=device) < p_rot, theta, torch.zeros_like(theta))
            G_inv = G_inv @ rotate2d_inv(-theta) # After anisotropic scaling.

        # Apply fractional translation with probability (xfrac * strength).
        if self.xfrac > 0:
            t = torch.randn([batch_size, 2], device=device) * self.xfrac_std
            t = torch.where(torch.rand([batch_size, 1], device=device) < self.xfrac * self.p, t, torch.zeros_like(t))
            G_inv = G_inv @ translate2d_inv(t[:,0] * width, t[:,1] * height)

        # ----------------------------------
        # Execute geometric transformations.
        # ----------------------------------

        # Execute if the transform is not identity.
        if G_inv is not I_3:

            # Calculate padding.
            cx = (width - 1) / 2
            cy = (height - 1) / 2
            cp = matrix([-cx, -cy, 1], [cx, -cy, 1], [cx, cy, 1], [-cx, cy, 1], device=device) # [idx, xyz]
            cp = G_inv @ cp.t() # [batch, xyz, idx]
            Hz_pad = self.Hz_geom.shape[0] // 4
            margin = cp[:, :2, :].permute(1, 0, 2).flatten(1) # [xy, batch * idx]
            margin = torch.cat([-margin, margin]).max(dim=1).values # [x0, y0, x1, y1]
            margin = margin + misc.constant([Hz_pad * 2 - cx, Hz_pad * 2 - cy] * 2, device=device)
            margin = margin.max(misc.constant([0, 0] * 2, device=device))
            margin = margin.min(misc.constant([width-1, height-1] * 2, device=device))
            mx0, my0, mx1, my1 = margin.ceil().to(torch.int32)

            # Pad image and adjust origin.
            images = torch.nn.functional.pad(input=images, pad=[mx0,mx1,my0,my1], mode='reflect')
            G_inv = translate2d((mx0 - mx1) / 2, (my0 - my1) / 2) @ G_inv

            # Upsample.
            images = upfirdn2d.upsample2d(x=images, f=self.Hz_geom, up=2)
            G_inv = scale2d(2, 2, device=device) @ G_inv @ scale2d_inv(2, 2, device=device)
            G_inv = translate2d(-0.5, -0.5, device=device) @ G_inv @ translate2d_inv(-0.5, -0.5, device=device)

            # Execute transformation.
            shape = [batch_size, num_channels, (height + Hz_pad * 2) * 2, (width + Hz_pad * 2) * 2]
            G_inv = scale2d(2 / images.shape[3], 2 / images.shape[2], device=device) @ G_inv @ scale2d_inv(2 / shape[3], 2 / shape[2], device=device)
            grid = torch.nn.functional.affine_grid(theta=G_inv[:,:2,:], size=shape, align_corners=False)
            images = grid_sample_gradfix.grid_sample(images, grid)

            # Downsample and crop.
            images = upfirdn2d.downsample2d(x=images, f=self.Hz_geom, down=2, padding=-Hz_pad*2, flip_filter=True)

        
                # --------------------------------------------
        # Select parameters for color transformations.
        # --------------------------------------------

        # Initialize homogeneous 3D transformation matrix: C @ color_in ==> color_out
        # I_4 = torch.eye(4, device=device)
        # C = I_4

        # # Apply brightness with probability (brightness * strength).
        # if self.brightness > 0:
        #     b = torch.randn([batch_size], device=device) * self.brightness_std
        #     b = torch.where(torch.rand([batch_size], device=device) < self.brightness * self.p, b, torch.zeros_like(b))
        #     C = translate3d(b, b, b) @ C

        # # Apply contrast with probability (contrast * strength).
        # if self.contrast > 0:
        #     c = torch.exp2(torch.randn([batch_size], device=device) * self.contrast_std)
        #     c = torch.where(torch.rand([batch_size], device=device) < self.contrast * self.p, c, torch.ones_like(c))
        #     C = scale3d(c, c, c) @ C

        # # Apply luma flip with probability (lumaflip * strength).
        # v = misc.constant(np.asarray([1, 1, 1, 0]) / np.sqrt(3), device=device) # Luma axis.
        # if self.lumaflip > 0:
        #     i = torch.floor(torch.rand([batch_size, 1, 1], device=device) * 2)
        #     i = torch.where(torch.rand([batch_size, 1, 1], device=device) < self.lumaflip * self.p, i, torch.zeros_like(i))
        #     C = (I_4 - 2 * v.ger(v) * i) @ C # Householder reflection.

        # # Apply hue rotation with probability (hue * strength).
        # if self.hue > 0 and num_channels > 1:
        #     theta = (torch.rand([batch_size], device=device) * 2 - 1) * np.pi * self.hue_max
        #     theta = torch.where(torch.rand([batch_size], device=device) < self.hue * self.p, theta, torch.zeros_like(theta))
        #     C = rotate3d(v, theta) @ C # Rotate around v.

        # # Apply saturation with probability (saturation * strength).
        # if self.saturation > 0 and num_channels > 1:
        #     s = torch.exp2(torch.randn([batch_size, 1, 1], device=device) * self.saturation_std)
        #     s = torch.where(torch.rand([batch_size, 1, 1], device=device) < self.saturation * self.p, s, torch.ones_like(s))
        #     C = (v.ger(v) + (I_4 - v.ger(v)) * s) @ C

        # # ------------------------------
        # # Execute color transformations.
        # # ------------------------------

        # # Execute if the transform is not identity.
        # if C is not I_4:
        #     images = images.reshape([batch_size, num_channels, height * width])
        #     if num_channels == 3:
        #         images = C[:, :3, :3] @ images + C[:, :3, 3:]
        #     elif num_channels == 1:
        #         C = C[:, :3, :].mean(dim=1, keepdims=True)
        #         images = images * C[:, :, :3].sum(dim=2, keepdims=True) + C[:, :, 3:]
        #     else:
        #         raise ValueError('Image must be RGB (3 channels) or L (1 channel)')
        #     images = images.reshape([batch_size, num_channels, height, width])

        # ----------------------
        # Image-space filtering.
        # ----------------------

        # if self.imgfilter > 0:
        #     num_bands = self.Hz_fbank.shape[0]
        #     assert len(self.imgfilter_bands) == num_bands
        #     expected_power = misc.constant(np.array([10, 1, 1, 1]) / 13, device=device) # Expected power spectrum (1/f).

        #     # Apply amplification for each band with probability (imgfilter * strength * band_strength).
        #     g = torch.ones([batch_size, num_bands], device=device) # Global gain vector (identity).
        #     for i, band_strength in enumerate(self.imgfilter_bands):
        #         t_i = torch.exp2(torch.randn([batch_size], device=device) * self.imgfilter_std)
        #         t_i = torch.where(torch.rand([batch_size], device=device) < self.imgfilter * self.p * band_strength, t_i, torch.ones_like(t_i))
        #         t = torch.ones([batch_size, num_bands], device=device)                  # Temporary gain vector.
        #         t[:, i] = t_i                                                           # Replace i'th element.
        #         t = t / (expected_power * t.square()).sum(dim=-1, keepdims=True).sqrt() # Normalize power.
        #         g = g * t                                                               # Accumulate into global gain.

        #     # Construct combined amplification filter.
        #     Hz_prime = g @ self.Hz_fbank                                    # [batch, tap]
        #     Hz_prime = Hz_prime.unsqueeze(1).repeat([1, num_channels, 1])   # [batch, channels, tap]
        #     Hz_prime = Hz_prime.reshape([batch_size * num_channels, 1, -1]) # [batch * channels, 1, tap]

        #     # Apply filter.
        #     p = self.Hz_fbank.shape[1] // 2
        #     images = images.reshape([1, batch_size * num_channels, height, width])
        #     images = torch.nn.functional.pad(input=images, pad=[p,p,p,p], mode='reflect')
        #     images = conv2d_gradfix.conv2d(input=images, weight=Hz_prime.unsqueeze(2), groups=batch_size*num_channels)
        #     images = conv2d_gradfix.conv2d(input=images, weight=Hz_prime.unsqueeze(3), groups=batch_size*num_channels)
        #     images = images.reshape([batch_size, num_channels, height, width])
        
        # ------------------------
        # Image-space corruptions.
        # ------------------------
        # Apply additive RGB noise with probability (noise * strength).
        if self.noise > 0:
            sigma = torch.randn([batch_size, 1, 1, 1], device=device).abs() * self.noise_std
            sigma = torch.where(torch.rand([batch_size, 1, 1, 1], device=device) < self.noise * self.p, sigma, torch.zeros_like(sigma))
            images = images + torch.randn([batch_size, num_channels, height, width], device=device) * sigma
            
        # Apply cutout with probability (cutout * strength).
        if self.cutout > 0:
            size = torch.full([batch_size, 2, 1, 1, 1], self.cutout_size, device=device)
            size = torch.where(torch.rand([batch_size, 1, 1, 1, 1], device=device) < self.cutout * self.p, size, torch.zeros_like(size))
            center = torch.rand([batch_size, 2, 1, 1, 1], device=device)
            coord_x = torch.arange(width, device=device).reshape([1, 1, 1, -1])
            coord_y = torch.arange(height, device=device).reshape([1, 1, -1, 1])
            mask_x = (((coord_x + 0.5) / width - center[:, 0]).abs() >= size[:, 0] / 2)
            mask_y = (((coord_y + 0.5) / height - center[:, 1]).abs() >= size[:, 1] / 2)
            mask = torch.logical_or(mask_x, mask_y).to(torch.float32)
            images = images * mask
        
        images = images.permute((0,2,3,1))
        images = images.view(original_shape)
        return images


if __name__ == "__main__":
    torch.set_printoptions(linewidth=1000,precision=4 )
    device = torch.device('cpu')
    # plt.ion()
    
    # image = torch.arange(50*27*27,device=device).reshape((50,27,27))
    
    # srotate2(image,torch.as_tensor(45,device=device))
    # srotate3(image,torch.as_tensor(45,device=device))
    
    # st = time.time()
    # for _ in range(2000):
    #     srotate2(image,torch.as_tensor(45,device=device))
    # print(time.time()-st)


    # images = torch.zeros((2,50,9,9),device=device)
    # images[:,:,4,0:5] = 1
    # images[:,:,0,:] = 1
    
    # plt.subplot(1,2,1)
    # plt.imshow(images[0].permute(1,2,0).cpu().numpy()[:,:,:3])
    # plt.show(block= False)
    
    # I_3 = torch.eye(3, device=device)
    # G_inv = I_3 @ scale2d_inv(1, torch.ones(2,device=device)*-1)
    # grid = torch.nn.functional.affine_grid(theta=G_inv[:,:2,:], size=images.shape, align_corners=False)
    # images = grid_sample_gradfix.grid_sample(images, grid)

    # plt.subplot(1,2,2)
    # plt.imshow(images[0].permute(1,2,0).cpu().numpy()[:,:,:3])
    # plt.show(block= False)
    
    prob = 1
    pipe = AugmentPipe(**{'rotate90':0,'xflip':0,'xint':0,'scale':0,'rotate':0,'aniso':0,'xfrac':0,'cutout':0,'yflip':0,'transpose':0,'shearx':0,'sheary':0,'noise':1,'noise_std':0.01,'srotate':0,'srotateinde':0})
    pipe.p = torch.ones(1).to(device)
    img = Image.open(r'.\dataset\test.jpg')
    img = img.resize((49,49))
    img= np.array(img,dtype=np.float32)
    #img = utils.normalization_v2(img)
    origin = torch.from_numpy(img).to(device=device)
    origin = origin.permute(2,0,1)
    origin = origin.reshape((1,*origin.shape))
    
    for i in range(5):
        origin = torch.cat((origin,origin))
    num_origin = origin.shape[0]
    img = np.transpose(origin[0].permute(0,1,2).cpu().numpy(),(1,2,0))
    
    plt.subplot(5,10,1)
    plt.imshow(img.astype(np.int32))
    plt.show(block= False)
    
    aug = pipe(origin)
    for i in range(num_origin):
        plt.subplot(5,10,i+2)
        img = aug[i].cpu().numpy()
        img = np.transpose(img,(1,2,0))
        img = img.astype(np.int32)
        plt.imshow(img)
        plt.show(block= False)
        
    pass