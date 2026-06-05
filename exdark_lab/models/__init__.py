# Models Package
# Conditional imports to avoid torch dependency in demo mode

try:
    from .retinex_module import MultiScaleRetinex, RetinexDecomposition
    from .physics_loss import PhysicsLoss, illumination_smooth_loss
    from .yolov10_retinex import YOLOv10nWithRetinex
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    MultiScaleRetinex = None
    RetinexDecomposition = None
    PhysicsLoss = None
    YOLOv10nWithRetinex = None
