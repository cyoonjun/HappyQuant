import torch
import logging

from .utils import load_and_process_hessian


logger = logging.getLogger(__name__)


def get_lr_initializer(init_method: str):
    init_method = init_method.lower()
    if init_method == 'srr':     return srr_init
    else:
        raise ValueError(f"Unknown init method: {init_method}")


def srr_init(weight: torch.Tensor, scale: torch.Tensor, rank: int, layer_qera_config: dict, return_ada: bool = False):

    device, dtype = weight.device, weight.dtype
    W = weight.to(device=device, dtype=dtype)
    scale = scale.to(device=device, dtype=dtype)

    if scale.ndim == 1:
        scaled_W = scale.view(-1, 1) * W.T
        inv_scale_vec = 1.0 / scale

        def scale_U(U):
            return U * inv_scale_vec.view(-1, 1)

    elif scale.ndim == 2:
        scaled_W = scale @ W.T
        inv_scale = torch.linalg.inv(scale)

        def scale_U(U):
            return inv_scale @ U
    else:
        raise ValueError("Scale must be either 1D or 2D")

    U, S_vals, V_T = torch.linalg.svd(scaled_W, full_matrices=False)

    U_scaled = scale_U(U)                            
    col_norms = torch.linalg.norm(U_scaled, dim=0)   
    scores = (S_vals * col_norms).tolist()           

    sorted_indices = sorted(
        range(len(scores)),
        key=lambda k: scores[k],
        reverse=True
    )

    frac = torch.sum(S_vals[:rank] ** 2) / torch.sum(S_vals ** 2)  
    ada_rank = int(rank * frac)  
    ada_rank = min(rank, ada_rank)
    U_top = U[:, :ada_rank]  

    Ph_sorted = U_top @ U_top.T  

    if isinstance(layer_qera_config, dict):
        collector = layer_qera_config.get("freeze_alpha_collector", None)
        layer_name = layer_qera_config.get("layer_name", None)
        if collector is not None and layer_name is not None:
            freeze_r = int(ada_rank)  
            collector[layer_name] = {
                "freeze_r": freeze_r,
                "alpha_vec": None,  
                "frac": float(frac.detach().cpu().item()),
                "s_ref": None,  
            }
            
    if scale.ndim == 1:
        Ph_scaled = (
            Ph_sorted
            * inv_scale_vec.view(-1, 1)
            * scale.view(1, -1)
        )
    else:
        Ph_scaled = inv_scale @ (Ph_sorted @ scale)

    L = Ph_scaled 
    R = W.T       

    if return_ada:
        return L, R, ada_rank
    else:
        return L, R

def _compute_scale_inv_dot_U(scale: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    # scale^-1 @ U
    if scale.ndim == 1:
        scale = torch.where(
            scale <= 0, torch.ones_like(scale) * torch.finfo(scale.dtype).eps, scale
        )
        return torch.linalg.solve(torch.diag(scale), U)
    elif scale.ndim == 2:
        try:
            return torch.linalg.solve(scale, U)
        except RuntimeError as e:
            logger.warning(f"Matrix inversion failed: {e} Adding turbulence to scale")
            U_scale, S_scale, V_T_scale = torch.linalg.svd(scale)
            S_scale = torch.where(
                S_scale <= 0,
                torch.ones_like(S_scale) * torch.finfo(S_scale.dtype).eps,
                S_scale,
            )
            scale = U_scale @ torch.diag(S_scale) @ V_T_scale
            return torch.linalg.solve(scale, U)
    else:
        raise ValueError("Scale must be either a vector (diagonal) or a matrix")
