def sparsity_regularization(model):
    reg = 0.0
    for module in model.modules():
        if hasattr(module, "m1"):
            reg = reg + module.m1.abs().sum()
        if hasattr(module, "m2"):
            reg = reg + module.m2.abs().sum()
        if hasattr(module, "m3"):
            reg = reg + module.m3.abs().sum()
        if hasattr(module, "z1"):
            reg = reg + module.z1.abs().sum()
        if hasattr(module, "z2"):
            reg = reg + module.z2.abs().sum()
        if hasattr(module, "z3"):
            reg = reg + module.z3.abs().sum()
    return reg