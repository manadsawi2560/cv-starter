__version__ = "0.1.0"

# Explicit re-export เพื่อให้ Ruff รับรู้ว่าเราจงใจ export ออกนอกแพ็กเกจ
from .utils import (
    normalize_image as normalize_image,
    simple_cnn_model as simple_cnn_model,
)

__all__ = ["normalize_image", "simple_cnn_model", "__version__"]
