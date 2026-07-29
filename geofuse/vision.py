import importlib
import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from .logger import get_logger

_log = get_logger("GVI")


# ────────────────────────────────────────────────────────────────────
# DEVICE SELECTION UTILITY
# ────────────────────────────────────────────────────────────────────
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


# ────────────────────────────────────────────────────────────────────
# PATH CONFIGURATION
# ────────────────────────────────────────────────────────────────────
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

        # cudnn.benchmark stays OFF on purpose.
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = False

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
            # strict=False tolerates benign extras (e.g. aux classifiers), but on
            # its own it will also happily leave the network at its *random*
            # initialisation if the checkpoint keys don't match the architecture
            # — which silently produces plausible-looking garbage masks. So
            # inspect what actually loaded and fail loudly on a real mismatch.
            incompatible = self.model.load_state_dict(state_dict, strict=False)
        except RuntimeError as e:
            print(
                "[FATAL] Weight Mismatch. You might need to specify num_classes manually."
            )
            raise e

        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)
        n_expected = len(self.model.state_dict())
        if missing:
            _log(
                "WARN",
                f"{len(missing)}/{n_expected} weights were NOT in the checkpoint "
                f"and keep their random init (e.g. {missing[:3]})",
            )
        if unexpected:
            _log(
                "WARN",
                f"{len(unexpected)} checkpoint tensors were ignored "
                f"(e.g. {unexpected[:3]})",
            )
        # More than a token handful missing means this is the wrong checkpoint
        # for this architecture; running on would emit meaningless segmentations.
        if len(missing) > 0.05 * max(n_expected, 1):
            raise RuntimeError(
                f"Checkpoint does not match '{model_name}': {len(missing)} of "
                f"{n_expected} weights missing. Refusing to run with a "
                f"partially-random model."
            )

        self.model.to(self.device)
        self.model.eval()

        # TODO: MULTI_GPU_INFERENCE - Add DataParallel wrapper for multi-GPU batch inference
        # if torch.cuda.device_count() > 1:
        #     self.model = torch.nn.DataParallel(self.model)

        # 6. Setup Transforms
        # No Resize here: callers normalise the panorama to one fixed size
        # first (``_SEGMENT_WIDTH`` in gvi.py).
        self.transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

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

    #: Cityscapes class indices this engine measures. Kept as named constants
    #: so the metric definition lives in one place instead of bare literals.
    VEGETATION_CLASS = 8
    TERRAIN_CLASS = 9

    def preprocess_to_tensor(self, image_input):
        """CPU-only path: returns a pinned (1, 3, H, W) tensor ready for H2D.

        Workers call this *outside* the GPU lock so multiple images can be
        preprocessed in parallel while another worker holds the GPU. The
        ``pin_memory()`` call makes the eventual ``to(device, non_blocking=True)``
        transfer go through pinned-DMA, overlapping with the previous forward.
        """
        if isinstance(image_input, str):
            img = Image.open(image_input).convert("RGB")
        else:
            img = image_input.convert("RGB")
        tensor = self.transform(img).unsqueeze(0)
        if self.device.type == "cuda":
            try:
                tensor = tensor.pin_memory()
            except RuntimeError:
                # Pinning can fail under heavy memory pressure — degrade
                # gracefully to a regular host tensor.
                pass
        return tensor

    def predict_batch(self, tensor):
        """GPU-only path: one forward for the whole batch.

        Accepts ``(3, H, W)``, ``(1, 3, H, W)`` or ``(B, 3, H, W)`` and always
        returns ``(B, H, W)`` integer class masks — one per input image, none
        discarded.
        """
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        tensor = tensor.to(self.device, non_blocking=True)
        with torch.no_grad():
            output = self.model(tensor)
            return output.max(1)[1].cpu().numpy()

    def predict_from_tensor(self, tensor):
        """Single-image path: preprocessed tensor -> one ``(H, W)`` mask.

        Rejects multi-image tensors rather than silently returning only the
        first mask (the previous ``[...][0]`` behaviour), which would have
        quietly dropped every other image in a batch.
        """
        masks = self.predict_batch(tensor)
        if masks.shape[0] != 1:
            raise ValueError(
                f"predict_from_tensor() takes a single image but got a batch "
                f"of {masks.shape[0]}; use predict_batch() for batched inference."
            )
        return masks[0]

    def predict(self, image_input):
        """Backward-compatible single-call path: CPU preprocess + GPU forward."""
        tensor = self.preprocess_to_tensor(image_input)
        return self.predict_from_tensor(tensor)

    def calculate_gvi_from_mask(self, mask_array):
        """Class fractions for ONE segmentation mask.

        Every metric is counted from the same ``(H, W)`` mask — i.e. the same
        single forward pass over the same single image — so vegetation and
        terrain are always measured at identical resolution and framing.
        ``GVI_Vegetation`` and ``GVI_Terrain`` are disjoint classes;
        ``GVI_Total`` is simply their sum.
        """
        total_pixels = mask_array.size
        veg_pixels = np.sum(mask_array == self.VEGETATION_CLASS)
        terrain_pixels = np.sum(mask_array == self.TERRAIN_CLASS)

        return {
            "GVI_Vegetation": veg_pixels / total_pixels,
            "GVI_Terrain": terrain_pixels / total_pixels,
            "GVI_Total": (veg_pixels + terrain_pixels) / total_pixels,
        }
