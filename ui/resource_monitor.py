"""Floating bottom-right resource monitor (CPU / GPU / RAM / Network).

A daemon thread polls system metrics into bounded deques; the Streamlit
fragment only copies a snapshot under a short-held lock and emits a single
``st.markdown`` block of inline SVG + HTML. A tiny ``components.html`` JS
shim toggles a ``body.gf-rm-open`` class on click so expand/collapse is
purely client-side — no Streamlit rerun on toggle.

GPU sampling tries ``pynvml`` first and falls back to ``nvidia-smi`` via
subprocess. If neither is available, the GPU/VRAM rows are hidden.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from collections import deque

import streamlit as st
import streamlit.components.v1 as components
from chrome import load_asset

_SAMPLE_INTERVAL_S = 1.0
_HISTORY_LEN = int(60 / _SAMPLE_INTERVAL_S)  # 60 s window

_CHART_W = 200
_CHART_H = 44


class _ResourceSampler:
    """Background daemon polling CPU/RAM/GPU/Network into ring buffers."""

    def __init__(self) -> None:
        import psutil

        self._psutil = psutil
        self._lock = threading.Lock()
        self.cpu: deque[float] = deque(maxlen=_HISTORY_LEN)
        self.ram: deque[float] = deque(maxlen=_HISTORY_LEN)
        self.gpu: deque[float] = deque(maxlen=_HISTORY_LEN)
        self.gpu_mem: deque[float] = deque(maxlen=_HISTORY_LEN)
        self.net_rx: deque[float] = deque(maxlen=_HISTORY_LEN)  # Mbps in
        self.net_tx: deque[float] = deque(maxlen=_HISTORY_LEN)  # Mbps out

        self._init_gpu()

        psutil.cpu_percent(interval=None)
        self._last_net = psutil.net_io_counters()
        self._last_net_t = time.monotonic()

        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="ResourceSampler"
        )
        self._thread.start()

    def _init_gpu(self) -> None:
        self._pynvml = None
        self._gpu_handle = None
        self._nvidia_smi: str | None = None
        self.gpu_name = ""

        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(self._gpu_handle)
            self.gpu_name = name.decode() if isinstance(name, bytes) else name
            self._pynvml = pynvml
            return
        except Exception:
            self._pynvml = None

        smi = shutil.which("nvidia-smi")
        if smi:
            try:
                out = subprocess.run(
                    [smi, "--query-gpu=name", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    timeout=2.0,
                )
                if out.returncode == 0 and out.stdout.strip():
                    self.gpu_name = out.stdout.strip().splitlines()[0].strip()
                    self._nvidia_smi = smi
            except Exception:
                pass

    @property
    def has_gpu(self) -> bool:
        return self._pynvml is not None or self._nvidia_smi is not None

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._sample()
            except Exception:
                pass
            self._stop.wait(_SAMPLE_INTERVAL_S)

    def _sample(self) -> None:
        ps = self._psutil
        cpu_v = float(ps.cpu_percent(interval=None))
        ram_v = float(ps.virtual_memory().percent)

        net = ps.net_io_counters()
        now = time.monotonic()
        dt = max(now - self._last_net_t, 1e-6)
        rx = (net.bytes_recv - self._last_net.bytes_recv) * 8.0 / 1e6 / dt
        tx = (net.bytes_sent - self._last_net.bytes_sent) * 8.0 / 1e6 / dt
        self._last_net = net
        self._last_net_t = now

        gpu_v, gpu_mem_v = self._sample_gpu()

        with self._lock:
            self.cpu.append(cpu_v)
            self.ram.append(ram_v)
            self.gpu.append(gpu_v)
            self.gpu_mem.append(gpu_mem_v)
            self.net_rx.append(max(rx, 0.0))
            self.net_tx.append(max(tx, 0.0))

    def _sample_gpu(self) -> tuple[float, float]:
        if self._pynvml is not None:
            try:
                util = self._pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                mem = self._pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                return float(util.gpu), float(mem.used) / float(mem.total) * 100.0
            except Exception:
                return 0.0, 0.0

        if self._nvidia_smi is not None:
            try:
                out = subprocess.run(
                    [
                        self._nvidia_smi,
                        "--query-gpu=utilization.gpu,memory.used,memory.total",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=1.5,
                )
                if out.returncode == 0:
                    line = out.stdout.strip().splitlines()[0]
                    util_s, used_s, total_s = (p.strip() for p in line.split(","))
                    used = float(used_s)
                    total = float(total_s)
                    return (
                        float(util_s),
                        (used / total * 100.0) if total > 0 else 0.0,
                    )
            except Exception:
                return 0.0, 0.0

        return 0.0, 0.0

    def snapshot(self) -> dict[str, list[float]]:
        with self._lock:
            return {
                "cpu": list(self.cpu),
                "ram": list(self.ram),
                "gpu": list(self.gpu),
                "gpu_mem": list(self.gpu_mem),
                "net_rx": list(self.net_rx),
                "net_tx": list(self.net_tx),
            }


@st.cache_resource
def _get_sampler() -> _ResourceSampler | None:
    """Process-level singleton; ``None`` if psutil isn't importable."""
    try:
        return _ResourceSampler()
    except Exception:
        return None


# -- Chart helpers -----------------------------------------------------------


def _series_paths(values: list[float], vmax: float) -> tuple[str, str]:
    """Return ``(polyline_points, filled_area_path)`` for an SVG series."""
    if not values:
        return "", ""
    n = len(values)
    vmax_safe = vmax if vmax > 0 else 1.0
    if n == 1:
        y = _CHART_H - min(max(values[0], 0.0), vmax_safe) / vmax_safe * _CHART_H
        pts = f"0,{y:.1f} {_CHART_W},{y:.1f}"
        area = (
            f"M 0,{_CHART_H} L 0,{y:.1f} L {_CHART_W},{y:.1f} L {_CHART_W},{_CHART_H} Z"
        )
        return pts, area
    step = _CHART_W / (n - 1)
    xs = [i * step for i in range(n)]
    ys = [_CHART_H - min(max(v, 0.0), vmax_safe) / vmax_safe * _CHART_H for v in values]
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys, strict=True))
    inner = " L ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys, strict=True))
    area = f"M {xs[0]:.1f},{_CHART_H} L {inner} L {xs[-1]:.1f},{_CHART_H} Z"
    return pts, area


def _chart_block(
    label: str,
    values: list[float],
    current: float,
    unit: str,
    color: str,
    grad_id: str,
    vmax: float = 100.0,
) -> str:
    pts, area = _series_paths(values, vmax)
    return f"""
<div class="gf-rm-block">
  <div class="gf-rm-row">
    <span class="gf-rm-label">{label}</span>
    <span class="gf-rm-value" style="color:{color};">{current:.0f}{unit}</span>
  </div>
  <svg class="gf-rm-svg" width="100%" height="{_CHART_H}"
       viewBox="0 0 {_CHART_W} {_CHART_H}" preserveAspectRatio="none">
    <defs>
      <linearGradient id="{grad_id}" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="{color}" stop-opacity="0.55"/>
        <stop offset="100%" stop-color="{color}" stop-opacity="0.04"/>
      </linearGradient>
    </defs>
    <path d="{area}" fill="url(#{grad_id})" stroke="none"/>
    <polyline points="{pts}" fill="none" stroke="{color}"
              stroke-width="1.6" vector-effect="non-scaling-stroke"
              stroke-linejoin="round" stroke-linecap="round"/>
  </svg>
</div>
"""


def _net_block(rx: list[float], tx: list[float]) -> str:
    color_dn = "#22d3ee"
    color_up = "#f472b6"
    dn_now = rx[-1] if rx else 0.0
    up_now = tx[-1] if tx else 0.0
    vmax = max(max(rx, default=0.0), max(tx, default=0.0), 1.0)
    pts_dn, area_dn = _series_paths(rx, vmax)
    pts_up, area_up = _series_paths(tx, vmax)
    return f"""
<div class="gf-rm-block">
  <div class="gf-rm-row">
    <span class="gf-rm-label">Network</span>
    <span>
      <span class="gf-rm-value" style="color:{color_dn};">↓ {dn_now:.1f}</span>
      <span class="gf-rm-value" style="color:{color_up};margin-left:5px;">↑ {up_now:.1f}</span>
      <span style="color:#7b8794;font-size:0.62rem;margin-left:3px;">Mbps</span>
    </span>
  </div>
  <svg class="gf-rm-svg" width="100%" height="{_CHART_H}"
       viewBox="0 0 {_CHART_W} {_CHART_H}" preserveAspectRatio="none">
    <defs>
      <linearGradient id="gf_net_dn" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="{color_dn}" stop-opacity="0.45"/>
        <stop offset="100%" stop-color="{color_dn}" stop-opacity="0.03"/>
      </linearGradient>
      <linearGradient id="gf_net_up" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="{color_up}" stop-opacity="0.45"/>
        <stop offset="100%" stop-color="{color_up}" stop-opacity="0.03"/>
      </linearGradient>
    </defs>
    <path d="{area_dn}" fill="url(#gf_net_dn)" stroke="none"/>
    <path d="{area_up}" fill="url(#gf_net_up)" stroke="none"/>
    <polyline points="{pts_dn}" fill="none" stroke="{color_dn}"
              stroke-width="1.6" vector-effect="non-scaling-stroke"
              stroke-linejoin="round" stroke-linecap="round"/>
    <polyline points="{pts_up}" fill="none" stroke="{color_up}"
              stroke-width="1.6" vector-effect="non-scaling-stroke"
              stroke-linejoin="round" stroke-linecap="round"/>
  </svg>
</div>
"""


# -- Mini pill + expanded panel ----------------------------------------------


_PALETTE = {
    "cpu": "#3b82f6",
    "ram": "#a855f7",
    "gpu": "#10b981",
    "vram": "#f59e0b",
}


def _mini_chip(color: str, value: str) -> str:
    return (
        '<span class="gf-rm-chip">'
        f'<span class="gf-rm-chip-dot" style="background:{color};"></span>'
        f'<span class="gf-rm-chip-val">{value}</span>'
        "</span>"
    )


def _mini_bar(snap: dict[str, list[float]], has_gpu: bool) -> str:
    cpu = snap["cpu"][-1] if snap["cpu"] else 0.0
    ram = snap["ram"][-1] if snap["ram"] else 0.0
    rx = snap["net_rx"][-1] if snap["net_rx"] else 0.0
    tx = snap["net_tx"][-1] if snap["net_tx"] else 0.0

    chips = [
        _mini_chip(_PALETTE["cpu"], f"{cpu:.0f}%"),
        _mini_chip(_PALETTE["ram"], f"{ram:.0f}%"),
    ]
    if has_gpu:
        gpu = snap["gpu"][-1] if snap["gpu"] else 0.0
        chips.append(_mini_chip(_PALETTE["gpu"], f"{gpu:.0f}%"))

    return (
        '<div class="gf-rm-mini" title="Click to toggle resource details">'
        + "".join(chips)
        + (
            '<span class="gf-rm-chip gf-rm-net">'
            f'<span style="color:#22d3ee;">↓ {rx:.1f}</span>'
            f'<span style="color:#f472b6;margin-left:4px;">↑ {tx:.1f}</span>'
            "</span>"
        )
        + '<span class="gf-rm-chevron">⌃</span>'
        + "</div>"
    )


def _full_panel(snap: dict[str, list[float]], has_gpu: bool) -> str:
    parts = [
        '<div class="gf-rm-panel">',
        '<div class="gf-rm-panel-title">System Resources</div>',
    ]
    parts.append(
        _chart_block(
            "CPU",
            snap["cpu"],
            snap["cpu"][-1] if snap["cpu"] else 0.0,
            "%",
            _PALETTE["cpu"],
            "gf_g_cpu",
        )
    )
    parts.append(
        _chart_block(
            "Memory",
            snap["ram"],
            snap["ram"][-1] if snap["ram"] else 0.0,
            "%",
            _PALETTE["ram"],
            "gf_g_ram",
        )
    )
    if has_gpu:
        parts.append(
            _chart_block(
                "GPU",
                snap["gpu"],
                snap["gpu"][-1] if snap["gpu"] else 0.0,
                "%",
                _PALETTE["gpu"],
                "gf_g_gpu",
            )
        )
        parts.append(
            _chart_block(
                "VRAM",
                snap["gpu_mem"],
                snap["gpu_mem"][-1] if snap["gpu_mem"] else 0.0,
                "%",
                _PALETTE["vram"],
                "gf_g_vram",
            )
        )
    parts.append(_net_block(snap["net_rx"], snap["net_tx"]))
    parts.append("</div>")
    return "".join(parts)


# CSS and JS for the floater pill, expanded panel, and click toggle
# live in ui/assets/resource_monitor.css and resource_monitor.js. See
# ui/chrome.py for the load_asset() helper and the why-not-inline rule.

def render_resource_monitor() -> None:
    """Render the floating resource monitor.

    Call once per script run from the top-level app entry — *not* inside any
    tab or sidebar container — so it stays visible on every tab.
    """
    sampler = _get_sampler()
    if sampler is None:
        return

    # One-shot JS injection. Re-runs on every script rerun but the script
    # itself is idempotent (binds with a sentinel attribute and only creates
    # the MutationObserver once via ``window.__gfRmObs``).
    components.html(
        f"<script>{load_asset('resource_monitor.js')}</script>",
        height=0,
        scrolling=False,
    )

    @st.fragment(run_every=_SAMPLE_INTERVAL_S)
    def _frag() -> None:
        snap = sampler.snapshot()
        html = (
            f"<style>{load_asset('resource_monitor.css')}</style>"
            + '<div class="gf-rm-floater">'
            + _full_panel(snap, sampler.has_gpu)
            + _mini_bar(snap, sampler.has_gpu)
            + "</div>"
        )
        st.markdown(html, unsafe_allow_html=True)

    _frag()
