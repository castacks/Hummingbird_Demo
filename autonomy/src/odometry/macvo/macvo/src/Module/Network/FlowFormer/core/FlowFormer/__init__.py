def build_flowformer(cfg, device):
    name = cfg.transformer 
    if name == 'latentcostformer':
        from .LatentCostFormer.transformer import FlowFormer
        return FlowFormer(cfg[name], device=device)
    else:
        raise ValueError(f"FlowFormer = {name} is not a valid architecture!")

