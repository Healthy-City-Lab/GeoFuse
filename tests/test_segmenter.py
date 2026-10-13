"""The segmenter's forward pass (``vision.DeepLabSegmenter``).

Class fractions counted on the device equal those counted from the mask; the
precision setting is validated and only CUDA runs fp16; on CUDA a CUDA-graph
replay gives the masks the eager forward gives (fp32 and fp16 alike, also with
the real model), shapes past the graph cap and failed captures run eagerly.
"""

import os
import sys
import threading
import unittest
from unittest import mock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import geopandas  # noqa: F401  (pyproj's DLLs load before torchvision's)
import numpy as np
import torch

import geofuse  # noqa: F401  (import order: torch before GeoPandas)
from geofuse import vision
from geofuse.vision import DeepLabSegmenter

CUDA = torch.cuda.is_available()
MODEL = os.path.join(ROOT, "geofuse", "model", "best_model.pth")


def _segmenter(device="cpu", precision="fp32", cuda_graphs=True):
    """A segmenter around a small random network (19 classes)."""
    seg = DeepLabSegmenter.__new__(DeepLabSegmenter)
    seg.device = torch.device(device)
    seg.precision = precision
    seg.cuda_graphs = cuda_graphs
    seg._graphs = {}
    seg._forward_lock = threading.Lock()
    torch.manual_seed(0)
    seg.model = (
        torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 19, 3, padding=1),
        )
        .to(seg.device)
        .eval()
    )
    return seg


def _batch(n=3, h=32, w=64, seed=1):
    return torch.from_numpy(
        np.random.default_rng(seed).normal(size=(n, 3, h, w)).astype(np.float32)
    )


class CountingTests(unittest.TestCase):
    def test_device_counts_equal_mask_counts(self):
        seg = _segmenter()
        batch = _batch()
        fractions, masks = seg.segment(batch, masks=True)
        reference = seg.predict_batch(batch)
        self.assertEqual(masks.dtype, np.uint8)
        np.testing.assert_array_equal(masks, reference)
        self.assertEqual(fractions, [seg.calculate_gvi_from_mask(m) for m in reference])
        self.assertIsNone(seg.segment(batch)[1])

    def test_a_single_image_is_a_batch_of_one(self):
        seg = _segmenter()
        image = _batch(n=1)[0]
        fractions, masks = seg.segment(image, masks=True)
        self.assertEqual((len(fractions), masks.shape), (1, (1, 32, 64)))


class PrecisionTests(unittest.TestCase):
    def test_precision_is_validated(self):
        seg = _segmenter()
        with self.assertRaises(ValueError):
            seg.precision = "fp64"
        self.assertEqual(vision.DEFAULT_PRECISION, "fp16")
        self.assertEqual(vision.PRECISIONS, ("fp16", "fp32"))

    def test_only_cuda_runs_fp16(self):
        seg = _segmenter(precision="fp16")
        self.assertEqual(seg.effective_precision, "fp32")
        seg.segment(_batch())
        self.assertFalse(seg.graphs_in_use)


@unittest.skipUnless(CUDA, "needs a CUDA device")
class CudaGraphTests(unittest.TestCase):
    def check_graph_equals_eager(self, seg, eager, batch):
        graphed = seg.segment(batch, masks=True)
        self.assertTrue(seg.graphs_in_use)
        expected = eager.segment(batch, masks=True)
        np.testing.assert_array_equal(graphed[1], expected[1])
        self.assertEqual(graphed[0], expected[0])

    def test_graph_replay_equals_eager(self):
        for precision in ("fp32", "fp16"):
            seg = _segmenter("cuda", precision)
            eager = _segmenter("cuda", precision, cuda_graphs=False)
            self.assertEqual(seg.effective_precision, precision)
            for seed in (1, 2):
                self.check_graph_equals_eager(seg, eager, _batch(n=1, seed=seed))

    def test_shapes_past_the_cap_run_eagerly(self):
        seg = _segmenter("cuda")
        eager = _segmenter("cuda", cuda_graphs=False)
        for n in range(1, vision._MAX_GRAPHS + 3):
            batch = _batch(n=n)
            np.testing.assert_array_equal(
                seg.segment(batch, masks=True)[1], eager.segment(batch, masks=True)[1]
            )
        self.assertEqual(len(seg._graphs), vision._MAX_GRAPHS)

    def test_a_failed_capture_runs_eagerly(self):
        seg = _segmenter("cuda")
        eager = _segmenter("cuda", cuda_graphs=False)
        with mock.patch.object(torch.cuda, "CUDAGraph", side_effect=RuntimeError):
            fractions, masks = seg.segment(_batch(), masks=True)
        self.assertFalse(seg.graphs_in_use)
        np.testing.assert_array_equal(masks, eager.segment(_batch(), masks=True)[1])

    @unittest.skipUnless(os.path.exists(MODEL), "model weights not installed")
    def test_real_model_graph_is_bit_identical_in_fp32(self):
        seg = DeepLabSegmenter(ckpt_path=MODEL, precision="fp32")
        rng = np.random.default_rng(3)
        image = torch.from_numpy(rng.normal(size=(1, 3, 512, 1024)).astype(np.float32))
        graphed = seg.segment(image, masks=True)[1]
        self.assertTrue(seg.graphs_in_use)
        seg.cuda_graphs = False
        np.testing.assert_array_equal(graphed, seg.segment(image, masks=True)[1])


if __name__ == "__main__":
    unittest.main()
