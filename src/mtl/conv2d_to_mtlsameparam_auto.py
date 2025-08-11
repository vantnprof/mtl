import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Tuple
from src.mtl.mtl_same_num_param import MultilinearTransformationLayerSameNumberParam
from src.mtl.utils import find_config_for_matrices, collect_layer_outputs, gather_in_out_channels
from typing import Optional


class MTLSameParamWrapper(nn.Module):
    def __init__(
        self,
        desire_param_num: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int,int],
        padding: Tuple[int,int],
        stride: Tuple[int,int] = (1,1),
        bias: bool=False,
        mtl_config: Optional[dict] = None
    ):
        super().__init__()
        self.mtl_config = mtl_config
        # 1) Core MTL layer
        self.mtl = MultilinearTransformationLayerSameNumberParam(
            desire_param_num,
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=bias,
            config_dim=mtl_config,
        )

        # 2) Mimic Conv2d stride if >1
        self.downsample = (
            nn.AvgPool2d(kernel_size=stride, stride=stride)
            if (stride[0] > 1 or stride[1] > 1)
            else None
        )
        # 3) Parse desired shape from config (if provided)
        if mtl_config and "desired_shape" in mtl_config:
            C_des, H_des, W_des = mtl_config["desired_shape"]
            self.desired_channels = C_des
            self.spatial_target = (H_des, W_des)
        else:
            self.desired_channels = None
            self.spatial_target = None

    def forward(self, x):
        # A) factorized MTL step
        out = self.mtl(x)

        # Channel adjust: non-parametric slice/pad to desired_channels if needed
        if self.desired_channels is not None and out.shape[1] != self.desired_channels:
            C_des = self.desired_channels
            C_out = out.shape[1]
            if C_out > C_des:
                out = out[:, :C_des, :, :]
            else:
                pad = torch.zeros((out.shape[0], C_des - C_out, out.shape[2], out.shape[3]),
                                  device=out.device, dtype=out.dtype)
                out = torch.cat([out, pad], dim=1)

        # E) force spatial to H_des×W_des
        if self.spatial_target is not None and out.shape[-2:] != self.spatial_target:
            # print("Surprise!")
            out = F.interpolate(out, size=self.spatial_target, mode="nearest")

        return out


def cnn2mtlsameparams(model_name: str, num_classes: int, 
                      input_size: Tuple[int, int, int]=(3, 32, 32), 
                      device="cuda"
    ):
    """
    Load a CNN (e.g., ResNet18) and replace all nn.Conv2d modules with MTL wrappers.
    """
    name = model_name.lower()
    if name == 'resnet18':
        model = models.resnet18(weights=None, num_classes=num_classes).to(device)
    else:
        raise NotImplementedError(f"Model {model_name} not supported")
    
    shapes_dict = collect_layer_outputs(model, input_size, device)
    parent = ""
    cnn_config_channels = {}
    gather_in_out_channels(module=model, parent=parent, cnn_config_channels=cnn_config_channels)
    input_shape = input_size
    for name in shapes_dict.keys():
        cnn_config_channels[name]["input_shape"] = input_shape
        cnn_config_channels[name]["output_shape"] = shapes_dict[name]
        input_shape = cnn_config_channels[name]["output_shape"]
    mtl_configs = {}
    no_replaced = set()
    for name, conv2d_config in cnn_config_channels.items():
        if not conv2d_config['is_conv2d']:
            continue
        D1 = conv2d_config['input_shape'][1]
        D2 = conv2d_config['input_shape'][2]
        D3 = conv2d_config['input_shape'][0]
        config = find_config_for_matrices(
            target_params=conv2d_config['total_params'],
            D1=D1, 
            D2=D2, 
            D3=D3,
            lowest_d1=(D1+2*conv2d_config['padding'][0]-conv2d_config['kernel_size'][0])//conv2d_config['stride'][0]+1
        )
        if config is None:
            print("No replacement for {}".format(name))
            no_replaced.add(name) 
            continue
        d1, d2, d3 = config
        # out_ch = conv2d_config['out_channels']
        # k_h, k_w = conv2d_config['kernel_size']
        # p_h, p_w = conv2d_config['padding']
        # s_h, s_w = conv2d_config['stride']
        # H_des = (D1 + 2*p_h - k_h) // s_h + 1
        # W_des = (D2 + 2*p_w - k_w) // s_w + 1
        mtl_configs[name] = {
            'D1': D1,
            'D2': D2,
            'D3': D3,
            'd1': d1,
            'd2': d2,
            'd3': d3,
            'desired_shape': conv2d_config['output_shape']
        }

    def replace_conv(module: nn.Module, parent: str):
        for attr_name, child in module.named_children():
            name = parent + "." + attr_name if parent else attr_name
            if isinstance(child, nn.Conv2d) and "downsample" not in name and name not in no_replaced:
                # replace Conv2d with MTL wrapper
                setattr(module, attr_name, MTLSameParamWrapper(
                    desire_param_num=child.kernel_size[0]*child.kernel_size[1]*child.in_channels*child.out_channels,
                    in_channels=child.in_channels,
                    out_channels=child.out_channels,
                    kernel_size=child.kernel_size,
                    padding=child.padding,
                    stride=child.stride,
                    bias=(child.bias is not None),
                    mtl_config=mtl_configs[name],
                ))
                print(" ****** Replaced {} by MTL Same Parameters".format(name))
            else:
                # leave downsample convs (and others) intact or recurse
                if not isinstance(child, nn.Conv2d):
                    next_parent = name
                    replace_conv(child, next_parent)
        return module
    
    parent = ""
    mtl_model = replace_conv(model, parent)  
      
    return mtl_model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = (3, 32, 32)
    # Quick sanity check: convert ResNet18 and run a dummy forward
    cnn_model = models.resnet18(weights=None, num_classes=10).to(device)

    # cnn_model = nn.Sequential(
    #     cnn_model.conv1, 
    #     cnn_model.bn1, 
    #     cnn_model.relu, cnn_model.maxpool, 
    #     cnn_model.layer1, 
    #     cnn_model.layer2, 
    #     cnn_model.layer3,
    #     cnn_model.layer4,
    # ).to(device)

    mtl_model = cnn2mtlsameparams('resnet18', input_size=input_size, num_classes=10, device=device)

    # mtl_model = nn.Sequential(
    #     mtl_model.conv1,
    #     mtl_model.bn1,
    #     mtl_model.relu,
    #     mtl_model.maxpool,
    #     mtl_model.layer1,
    #     mtl_model.layer2,  
    #     mtl_model.layer3,
    #     mtl_model.layer4,
    # ).to(device)


    mtl_model.eval()
    cnn_model.eval()

    x = torch.randn(1, *input_size).to(device)
    print("Input shape: ", x.shape)
    with torch.no_grad():
        cnn_out = cnn_model(x)
        mtl_out = mtl_model(x)

    print(f"CNN's output shape: {cnn_out.shape}")
    print(f"MTL's output shape: {mtl_out.shape}")

    # Trainable parameters comparison
    cnn_params = sum(p.numel() for p in cnn_model.parameters() if p.requires_grad)
    mtl_params = sum(p.numel() for p in mtl_model.parameters() if p.requires_grad)
    print("CNN's trainable params: ", cnn_params)
    print("MTL's trainable params: ", mtl_params)