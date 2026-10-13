"""A small numeric stand-in for the Earth Engine calls ``NDVIEngine`` makes.

Images are functions of planar coordinates ``(X, Y)`` returning arrays, with
NaN for masked pixels; collections are lists of images with a date and
properties. The band algebra the NDVI preps use (``select``, arithmetic,
``bitwiseAnd`` / ``Or`` / ``And`` / ``eq`` / ``neq`` / ``lt`` / ``gte``,
``updateMask``, ``unmask``), the collection operations (``filterDate``,
``filter``, ``map``, ``merge``, ``size``) and the temporal reducers are
evaluated with numpy, so a test can run the real engine and check every output
pixel.

``FakeEarthEngine(aoi, collections, images)`` patches ``geofuse.ndvi`` while
active: ``collections`` maps a collection id to its scenes, ``images`` maps an
``ee.Image`` asset id (the water layer) to a static image. ``exports`` records
every export's band names and dtype; ``queried`` every collection id opened.
"""

from __future__ import annotations

import math
from unittest import mock

import numpy as np
import rasterio
import shapely
from pyproj import CRS, Transformer
from rasterio.transform import from_origin
from shapely.ops import transform as shp_transform

from geofuse import ndvi

GRID = CRS.from_epsg(32611)
_TO_GRID = Transformer.from_crs("EPSG:4326", GRID, always_xy=True).transform


def _const(v):
    return lambda X, Y: np.full(np.shape(X), float(v))


class Image:
    """One or more named bands, each a function of ``(X, Y)``."""

    def __init__(self, bands, props=None, dtype="float64"):
        self.bands = {k: (v if callable(v) else _const(v)) for k, v in bands.items()}
        self.props = dict(props or {})
        self.dtype = dtype

    # ── evaluation ──────────────────────────────────────────────
    def evaluate(self, X, Y):
        return np.stack(
            [np.asarray(f(X, Y), dtype=np.float64) for f in self.bands.values()]
        )

    def _one(self):
        if len(self.bands) != 1:
            raise ValueError(f"single-band op on bands {list(self.bands)}")
        name, fn = next(iter(self.bands.items()))
        return name, fn

    def _unary(self, op):
        name, fn = self._one()
        return Image({name: lambda X, Y: op(fn(X, Y))}, self.props, self.dtype)

    def _binary(self, other, op):
        name, fn = self._one()
        if isinstance(other, Image):
            _, g = other._one()
        else:
            g = _const(other)
        return Image({name: lambda X, Y: op(fn(X, Y), g(X, Y))}, self.props, self.dtype)

    # ── band selection and naming ───────────────────────────────
    def select(self, *names):
        return Image({n: self.bands[n] for n in names}, self.props, self.dtype)

    def rename(self, *names):
        bands = dict(zip(names, self.bands.values(), strict=False))
        return Image(bands, self.props, self.dtype)

    # ── arithmetic ──────────────────────────────────────────────
    def multiply(self, k):
        return self._binary(k, np.multiply)

    def add(self, k):
        return self._binary(k, np.add)

    def subtract(self, k):
        return self._binary(k, np.subtract)

    def divide(self, k):
        return self._binary(k, np.divide)

    # ── bit and logic operations (NaN stays NaN) ────────────────
    @staticmethod
    def _logic(f):
        def op(a, b):
            out = f(np.nan_to_num(a), np.nan_to_num(b)).astype(np.float64)
            return np.where(np.isnan(a) | np.isnan(b), np.nan, out)

        return op

    def bitwiseAnd(self, k):
        return self._binary(
            k, self._logic(lambda a, b: a.astype(np.int64) & b.astype(np.int64))
        )

    def Or(self, other):
        return self._binary(other, self._logic(lambda a, b: (a != 0) | (b != 0)))

    def And(self, other):
        return self._binary(other, self._logic(lambda a, b: (a != 0) & (b != 0)))

    def eq(self, k):
        return self._binary(k, self._logic(lambda a, b: a == b))

    def neq(self, k):
        return self._binary(k, self._logic(lambda a, b: a != b))

    def lt(self, k):
        return self._binary(k, self._logic(lambda a, b: a < b))

    def gte(self, k):
        return self._binary(k, self._logic(lambda a, b: a >= b))

    # ── masks ───────────────────────────────────────────────────
    def updateMask(self, mask):
        _, m = mask._one()

        def apply(f):
            return lambda X, Y: np.where(np.nan_to_num(m(X, Y)) != 0, f(X, Y), np.nan)

        return Image(
            {k: apply(f) for k, f in self.bands.items()}, self.props, self.dtype
        )

    def unmask(self, value=0):
        return self._unary(lambda a: np.where(np.isnan(a), value, a))

    def clip(self, _region):
        return self

    def toFloat(self):
        return Image(self.bands, self.props, "float32")

    def copyProperties(self, img, _names=None):
        return Image(self.bands, {**img.props, **self.props}, self.dtype)

    def propertyNames(self):
        return list(self.props)

    @staticmethod
    def cat(images):
        bands = {}
        for im in images:
            bands.update(im.bands)
        return Image(bands, {}, images[0].dtype)


class Number:
    def __init__(self, v):
        self.v = v

    def getInfo(self):
        return self.v


class Collection:
    def __init__(self, images):
        self.images = list(images)

    def filterBounds(self, _aoi):
        return self

    def filterDate(self, start, end):
        return Collection(
            im for im in self.images if str(start) <= im.props["date"] < str(end)
        )

    def filter(self, flt):
        op, prop, value = flt
        assert op == "lt"
        return Collection(im for im in self.images if im.props[prop] < value)

    def map(self, fn):
        return Collection(fn(im) for im in self.images)

    def merge(self, other):
        return Collection(self.images + other.images)

    def size(self):
        return Number(len(self.images))

    def select(self, name):
        return Collection(im.select(name) for im in self.images)

    def _reduce(self, reducer):
        images = list(self.images)

        def fn(X, Y):
            if not images:
                return np.full(np.shape(X), np.nan)
            stack = np.ma.masked_invalid(
                np.stack([im._one()[1](X, Y) for im in images])
            )
            return np.ma.filled(reducer(stack, axis=0).astype(np.float64), np.nan)

        return Image({"NDVI": fn})

    def median(self):
        return self._reduce(np.ma.median)

    def mean(self):
        return self._reduce(np.ma.mean)

    def max(self):
        return self._reduce(np.ma.max)


class FakeEarthEngine:
    """Patches ``geofuse.ndvi``'s Earth Engine seams while active."""

    def __init__(self, aoi, collections, images=None):
        self.aoi = aoi
        self.collections = collections
        self.images = images or {}
        self.exports: list[tuple[tuple[str, ...], str]] = []
        self.queried: list[str] = []

    def _image_collection(self, cid):
        self.queried.append(cid)
        return Collection(self.collections.get(cid, []))

    def export(self, image, filename, *, crs, crs_transform, region, timeout=300):
        self.exports.append((tuple(image.bands), image.dtype))
        region_grid = shp_transform(_TO_GRID, region)
        minx, miny, maxx, maxy = region_grid.bounds
        res = float(crs_transform[0])
        x0, x1 = math.floor(minx / res) * res, math.ceil(maxx / res) * res
        y0, y1 = math.floor(miny / res) * res, math.ceil(maxy / res) * res
        w, h = int(round((x1 - x0) / res)), int(round((y1 - y0) / res))
        X, Y = np.meshgrid(
            x0 + (np.arange(w) + 0.5) * res, y1 - (np.arange(h) + 0.5) * res
        )
        data = image.evaluate(X, Y).astype(image.dtype)
        if image.dtype == "float32":
            data[:, ~shapely.contains_xy(region_grid, X, Y)] = -np.inf
        profile = {
            "driver": "GTiff",
            "height": h,
            "width": w,
            "count": data.shape[0],
            "dtype": image.dtype,
            "crs": GRID,
            "transform": from_origin(x0, y1, res, res),
        }
        with rasterio.open(filename, "w", **profile) as dst:
            dst.write(data)

    def __enter__(self):
        fake = mock.MagicMock()
        fake.ImageCollection.side_effect = self._image_collection
        fake.Filter.lt.side_effect = lambda prop, v: ("lt", prop, v)
        fake.Image.side_effect = lambda asset: self.images[asset]
        fake.Image.cat.side_effect = Image.cat
        fake.FeatureCollection.return_value.geometry.return_value = (
            self.aoi.geometry.iloc[0]
        )
        self._patches = [
            mock.patch.object(ndvi, "ee", fake),
            mock.patch.object(ndvi, "shapely_to_ee_geometry", side_effect=lambda g: g),
            mock.patch.object(ndvi, "_export_ee_image_to_tif", side_effect=self.export),
        ]
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in reversed(self._patches):
            p.stop()
