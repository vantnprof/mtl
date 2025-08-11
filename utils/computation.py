import torch
import torch.nn as nn
from typing import Union, List, Tuple, Dict
import torch
from src.mtl.mtl import MTL
# Global variable to store the total multiplications
total_multiplications = 0
import time


# def count_multiplications_hook(module: nn.Module, input: tuple, output: Union[torch.Tensor, tuple]) -> None:
#     """
#     A forward hook that counts the number of multiplications in a layer.
#     This function is registered to run after a forward pass on a module.
#     """
#     global total_multiplications
    
#     # Get the input tensor
#     input_tensor = input[0]
    
#     # --- Linear Layers ---
#     if isinstance(module, nn.Linear):
#         # For a linear layer, multiplications = input_features * output_features
#         # input_tensor.shape is (batch_size, ..., in_features)
#         # module.out_features is the number of output features
#         in_features = module.in_features
#         out_features = module.out_features
#         # We consider the operations for a single item in the batch
#         multiplications = in_features * out_features
#         total_multiplications += multiplications
#         # print(f"Layer: {module.__class__.__name__}, Mults: {multiplications}")

#     # --- Convolutional Layers (2D) ---
#     elif isinstance(module, nn.Conv2d):
#         # For a conv layer, mults = Cin * Cout * K^2 * Hout * Wout
#         # input_tensor.shape is (batch_size, in_channels, in_height, in_width)
#         # output.shape is (batch_size, out_channels, out_height, out_width)
#         in_channels = module.in_channels
#         out_channels = module.out_channels
#         kernel_h, kernel_w = module.kernel_size
#         out_h, out_w = output.shape[2:] # from the output tensor shape
        
#         # Multiplications per output pixel = in_channels * kernel_h * kernel_w
#         # Total multiplications = (mults per output pixel) * (number of output pixels) * out_channels
#         multiplications = in_channels * out_channels * kernel_h * kernel_w * out_h * out_w
#         total_multiplications += multiplications
#         # print(f"Layer: {module.__class__.__name__}, Mults: {multiplications}")
def count_multiplications_hook(module: nn.Module, input: tuple, output: Union[torch.Tensor, tuple]):
    """
    A forward hook that counts multiplications for standard and custom layers.
    - Handles nn.Linear and nn.Conv2d directly.
    - Adds a custom calculation for the einsum operations in the MTL layer.
    - Automatically handles composite layers like LowRankConv2d and MLConv
      by summing the multiplications of their constituent standard layers.
    """
    global total_multiplications

    # --- Standard Layers ---
    if isinstance(module, nn.Linear):
        total_multiplications += module.in_features * module.out_features

    elif isinstance(module, nn.Conv2d):
        _, C_out, H_out, W_out = output.shape
        mults = module.in_channels * C_out * module.kernel_size[0] * module.kernel_size[1] * H_out * W_out
        # For depthwise convolution, out_channels is not C_out but module.in_channels
        if module.groups == module.in_channels and module.in_channels > 1:
             mults /= module.in_channels # Correct for depthwise by removing the C_out factor
        total_multiplications += mults

    # --- Custom Multilinear Layer (MTL) ---
    elif isinstance(module, MTL):
        # This calculation is based on the three einsum operations in the MTL.forward method.
        # We use the shapes of the actual input and output tensors.
        _, C_in, H_in, W_in = input[0].shape
        _, C_out, H_out, W_out = output.shape

        # 1. Mode-3 product: einsum('bchw,oc->bohw', X, U3)
        mults1 = C_out * H_in * W_in * C_in
        
        # 2. Mode-1 product: einsum('bchw,oh->bcow', out, U1)
        mults2 = C_out * W_in * H_out * H_in
        
        # 3. Mode-2 product: einsum('bchw,ow->bcho', out, U2)
        mults3 = C_out * H_out * W_out * W_in
        
        total_multiplications += (mults1 + mults2 + mults3)

        print(module, total_multiplications)


def calculate_model_multiplications(model: nn.Module, input_shape: Union[List[int], Tuple[int, ...]]) -> int:
    """
    Calculates the total number of multiplications for a model's forward pass.

    Args:
        model (nn.Module): The PyTorch model.
        input_shape (Union[List[int], Tuple[int,...]]): The shape of the input tensor 
                                                        (including batch size, e.g., [1, 3, 224, 224]).

    Returns:
        int: The total number of theoretical multiplications.
    """
    global total_multiplications
    total_multiplications = 0  # Reset the global counter

    # Register the forward hook for all modules in the model
    hooks = []
    for module in model.modules():
        # We only care about layers with learnable parameters that perform multiplications
        if isinstance(module, (nn.Linear, nn.Conv2d)) or isinstance(module, MTL):
            hook = module.register_forward_hook(count_multiplications_hook)
            hooks.append(hook)
        

    # Create a dummy input tensor with the specified shape
    dummy_input = torch.randn(input_shape)

    # Perform a forward pass to trigger the hooks
    model.eval() # Set model to evaluation mode
    with torch.no_grad():
        model(dummy_input)

    # Remove the hooks to clean up
    for hook in hooks:
        hook.remove()
        
    return total_multiplications


def measure_cpu_inference_time(
    model: nn.Module, 
    input_shape: Tuple[int, ...], 
    num_runs: int = 100, 
    warmup_runs: int = 20
) -> Dict[str, float]:
    """
    Measures the average forward pass time of a PyTorch model on the CPU.

    Args:
        model (nn.Module): The PyTorch model to evaluate.
        input_shape (Tuple[int, ...]): The shape of the input tensor 
                                       (e.g., (1, 3, 224, 224)).
        num_runs (int): The number of timed forward passes to average.
        warmup_runs (int): The number of untimed forward passes to run for warm-up.

    Returns:
        Dict[str, float]: A dictionary containing the mean and standard deviation
                          of the inference time in milliseconds.
    """
    # 1. Prepare model and input
    device = torch.device("cpu")
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    # Create a dummy input tensor
    dummy_input = torch.randn(input_shape, device=device)

    timings = []

    with torch.no_grad():
        # 2. Warm-up phase
        print(f"Running {warmup_runs} warm-up iterations...")
        for _ in range(warmup_runs):
            _ = model(dummy_input)

        # 3. Measurement phase
        print(f"Running {num_runs} timed iterations...")
        for _ in range(num_runs):
            # time.perf_counter() is suitable for measuring short durations
            start_time = time.perf_counter()
            _ = model(dummy_input)
            end_time = time.perf_counter()
            
            # Append time in milliseconds
            timings.append((end_time - start_time) * 1000)

    # 4. Calculate statistics
    timings_tensor = torch.tensor(timings)
    mean_ms = timings_tensor.mean().item()
    std_ms = timings_tensor.std().item()

    return {
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "num_runs": num_runs
    }

