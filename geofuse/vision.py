import importlib
import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from .logger import get_logger

_log = get_logger("GVI")


# ---------------------------------------------------------
# DEVICE SELECTION UTILITY
# ---------------------------------------------------------
def get_best_device(preferred_device=None):
    """
    Select best available device prioritizing: CUDA > MPS > CPU

    Args:
        preferred_device (str): Optional override ('cuda', 'mps', 'cpu')

    Returns:
        torch.device: Best available device
    """
    if preferred_device:
        if preferred_device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif preferred_device == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        elif preferred_device == "cpu":
            return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


# ---------------------------------------------------------
# PATH CONFIGURATION
# ---------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
dl_core_path = os.path.join(current_dir, "dl_core")

if dl_core_path not in sys.path:
    sys.path.append(dl_core_path)


class DeepLabSegmenter:
    def __init__(self, model_name=None, num_classes=19, ckpt_path=None, device="cuda"):
        """
        Args:
            model_name (str): Optional. If None, it is auto-detected from weights.
            ckpt_path (str): Path to the .pth file.
            device (str): 'cuda', 'mps', or 'cpu'. Auto-selects best available if not valid.
        """
        self.device = get_best_device(device)
        _log("INFO", f"Using device: {self.device}")

        # 1. Resolve Model Path
        if ckpt_path is None:
            ckpt_path = os.path.join(current_dir, "model", "best_model.pth")

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"[ERROR] Model not found at: {ckpt_path}")

        _log("INFO", f"Loading checkpoint from: {ckpt_path}")

        # 2. Load Checkpoint (CPU first to inspect structure)
        # weights_only=False allows loading legacy NumPy data in the checkpoint
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Handle DataParallel prefix (remove "module.")
        state_dict = (
            checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
        )
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        # 3. AUTO-DETECT BACKBONE
        if model_name is None:
            model_name = self._detect_backbone(state_dict)
            _log("INFO", f"Auto-detected architecture: {model_name}")

        # 4. Initialize Network Architecture
        try:
            network_module = importlib.import_module("network")
            # Dynamically instantiate the class (e.g., deeplabv3plus_resnet101)
            self.model = network_module.modeling.__dict__[model_name](
                num_classes=num_classes, output_stride=16
            )
        except KeyError:
            _log("ERROR", f"Model '{model_name}' is not defined in network.modeling.")
            print(
                "Available models usually include: deeplabv3plus_resnet101, deeplabv3plus_mobilenet"
            )
            raise
        except Exception as e:
            print(f"[FATAL] Could not load network module from: {dl_core_path}")
            raise ImportError(f"DeepLab Import Failed: {e}")

        # 5. Load Weights
        try:
            # strict=False allows ignoring minor mismatches (like aux classifiers)
            self.model.load_state_dict(state_dict, strict=False)
        except RuntimeError as e:
            print(
                "[FATAL] Weight Mismatch. You might need to specify num_classes manually."
            )
            raise e

        self.model.to(self.device)
        self.model.eval()

        # TODO: MULTI_GPU_INFERENCE - Add DataParallel wrapper for multi-GPU batch inference
        # if torch.cuda.device_count() > 1:
        #     self.model = torch.nn.DataParallel(self.model)

        # 6. Setup Transforms & Colors
        self.transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.decode_fn = self._get_cityscapes_decode_fn()

    def _detect_backbone(self, state_dict):
        """
        Analyzes weight shapes to distinguish ResNet101 vs MobileNet vs Xception.
        """
        # We look at the first convolution of the ASPP module (classifier)
        # Key: classifier.aspp.convs.0.0.weight
        # Shape: [256, INPUT_CHANNELS, 1, 1]

        aspp_key = "classifier.aspp.convs.0.0.weight"

        if aspp_key in state_dict:
            input_channels = state_dict[aspp_key].shape[1]

            if input_channels == 2048:
                # 2048 channels = ResNet50/101 or Xception
                # To distinguish, check for ResNet specific keys
                if any("layer4" in k for k in state_dict.keys()):
                    return "deeplabv3plus_resnet101"
                else:
                    return "deeplabv3plus_xception"

            elif input_channels == 320:
                # 320 channels = MobileNetV2
                return "deeplabv3plus_mobilenet"

            elif input_channels == 512:
                return "deeplabv3plus_resnet18"  # Rare but possible

        # Fallback default if detection fails
        _log("WARN", "Could not auto-detect backbone. Defaulting to ResNet101.")
        return "deeplabv3plus_resnet101"

    def _get_cityscapes_decode_fn(self):
        valid_classes = np.arange(19)
        voc_cmap = np.zeros((256, 3), dtype=np.uint8)
        voc_cmap[8] = [107, 142, 35]  # Vegetation
        voc_cmap[9] = [152, 251, 152]  # Terrain

        def decode(mask):
            return voc_cmap[mask]

        return decode

    def predict(self, image_input):
        if isinstance(image_input, str):
            img = Image.open(image_input).convert("RGB")
        else:
            img = image_input.convert("RGB")

        img_t = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(img_t)
            pred_mask = output.max(1)[1].cpu().numpy()[0]

        return pred_mask

    def calculate_gvi_from_mask(self, mask_array):
        total_pixels = mask_array.size
        veg_pixels = np.sum(mask_array == 8)
        terrain_pixels = np.sum(mask_array == 9)

        return {
            "GVI_Vegetation": veg_pixels / total_pixels,
            "GVI_Terrain": terrain_pixels / total_pixels,
            "GVI_Total": (veg_pixels + terrain_pixels) / total_pixels,
        }
