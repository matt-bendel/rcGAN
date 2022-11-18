from .mri_gen_wrapper import MRIGANWrapper
from .inpaint_gen_wrapper import InpaintGANWrapper

def get_gen_wrapper(is_mri, cfg, G):
    if is_mri:
        g_wrapper = MRIGANWrapper(G, cfg)
    else:
        g_wrapper = InpaintGANWrapper(G, cfg)

    return g_wrapper