# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Ultralytics modules.

Example:
    Visualize a module with Netron.
    ```python
    from ultralytics.nn.modules import *
    import torch
    import os

    x = torch.ones(1, 128, 40, 40)
    m = Conv(128, 128)
    f = f'{m._get_name()}.onnx'
    torch.onnx.export(m, x, f)
    os.system(f'onnxsim {f} {f} && open {f}')
    ```
"""
from .UnblurNet import UnblurNet
from .UnblurNet1 import UnblurNet1
from .UnblurNet2 import UnblurNet2
from .UnblurNet3 import UnblurNet3
from .UnblurNet4 import UnblurNet4
from .UnblurNet5 import UnblurNet5
from .UnblurNet6 import UnblurNet6
from .unfog import *
from .gsconv import GSConv

from .block import (
    C1,
    C2,
    C3,
    C3TR,
    DFL,
    SPP,
    SPPF,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    ImagePoolingAttn,
    C3Ghost,
    C3x,
    GhostBottleneck,
    HGBlock,
    HGStem,
    Proto,
    RepC3,
    ResNetLayer,
    ContrastiveHead,
    BNContrastiveHead,
    RepNCSPELAN4,
    ADown,
    SPPELAN,
    CBFuse,
    CBLinear,
    Silence,
    PSA,
    C2fCIB,
    SCDown,
    RepVGGDW,
    BiLevelRoutingAttention,
)
from .conv import (
    CBAM,
    ChannelAttention,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostConv,
    LightConv,
    RepConv,
    SpatialAttention,

)
from .head import OBB, Classify, Detect, Pose, RTDETRDecoder, Segment, WorldDetect, v10Detect
from .transformer import (
    AIFI,
    MLP,
    DeformableTransformerDecoder,
    DeformableTransformerDecoderLayer,
    LayerNorm2d,
    MLPBlock,
    MSDeformAttn,
    TransformerBlock,
    TransformerEncoderLayer,
    TransformerLayer,
)

__all__ = (
    "Conv",
    "Conv2",
    "LightConv",
    "RepConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Concat",
    "TransformerLayer",
    "TransformerBlock",
    "MLPBlock",
    "LayerNorm2d",
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C2fAttn",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "Detect",
    "Segment",
    "Pose",
    "Classify",
    "TransformerEncoderLayer",
    "RepC3",
    "RTDETRDecoder",
    "AIFI",
    "DeformableTransformerDecoder",
    "DeformableTransformerDecoderLayer",
    "MSDeformAttn",
    "MLP",
    "ResNetLayer",
    "OBB",
    "WorldDetect",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "RepNCSPELAN4",
    "ADown",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "Silence",
    "PSA",
    "C2fCIB",
    "SCDown",
    "RepVGGDW",
    "v10Detect",
    "BiLevelRoutingAttention",
    "UnblurNet",
    "UnblurNet1",
    "UnblurNet2",
    "UnblurNet3",
    "UnblurNet4",
    "UnblurNet5",
    "UnblurNet6",
    "unfog_net",
    "GSConv"
)
