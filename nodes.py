import torch

class Krea2Attention:
    @classmethod
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {}),
                "strength": ("FLOAT", {"default": 1, "min": -1e9,"max": 1e9, "step": 1 / 20, "round": 1e-6}),
            }
        }
    FUNCTION = "patch_model"
    RETURN_TYPES = ('MODEL',)
    CATEGORY = "model/patch/krea2"
    def patch_model(self, model, strength):
        k = 'diffusion_model.txtfusion.projector.weight'
        m = model.clone()
        w = m.model_state_dict()[k]
        m.add_patches({k: (torch.zeros_like(w),)}, 0.0, strength)
        return (m,)

NODE_CLASS_MAPPINGS = {
    "Krea 2 attention" : Krea2Attention,
    }