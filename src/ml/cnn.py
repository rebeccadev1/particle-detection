"""Tiny two-channel crop CNN for the uncertain band of the FP cascade.

Weights are numpy arrays (no torch). Input is the unmarked raw+residual crop;
circled JPEGs are never used. Inference resizes 96×96 patches to 48×48.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

CNN_SIZE = 48
CNN_CHANNELS = 2


def _resize_batch(channels: np.ndarray, size: int = CNN_SIZE) -> np.ndarray:
    array = np.asarray(channels, dtype=np.float32)
    if array.ndim != 4 or array.shape[1] != CNN_CHANNELS:
        raise ValueError(f"Expected (N, 2, H, W), got {array.shape}")
    n, _, height, width = array.shape
    if height == size and width == size:
        return array
    out = np.empty((n, CNN_CHANNELS, size, size), dtype=np.float32)
    for i in range(n):
        for c in range(CNN_CHANNELS):
            out[i, c] = cv2.resize(
                array[i, c], (size, size), interpolation=cv2.INTER_AREA
            )
    return out


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    z = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def _im2col(x: np.ndarray, kh: int, kw: int, stride: int, pad: int) -> np.ndarray:
    n, c, h, w = x.shape
    if pad:
        x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
    out_h = (h + 2 * pad - kh) // stride + 1
    out_w = (w + 2 * pad - kw) // stride + 1
    windows = np.lib.stride_tricks.sliding_window_view(x, (kh, kw), axis=(2, 3))
    windows = windows[:, :, ::stride, ::stride]
    # (n, c, oh, ow, kh, kw) -> (n * oh * ow, c * kh * kw)
    return np.moveaxis(windows, 1, 3).reshape(n * out_h * out_w, c * kh * kw), out_h, out_w


def _col2im(
    cols: np.ndarray,
    n: int,
    c: int,
    h: int,
    w: int,
    kh: int,
    kw: int,
    stride: int,
    pad: int,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    x = np.zeros((n, c, h + 2 * pad, w + 2 * pad), dtype=np.float32)
    patch = cols.reshape(n, out_h, out_w, c, kh, kw)
    patch = np.moveaxis(patch, 3, 1)
    for iy in range(out_h):
        ys = iy * stride
        for ix in range(out_w):
            xs = ix * stride
            x[:, :, ys : ys + kh, xs : xs + kw] += patch[:, :, iy, ix]
    if pad:
        return x[:, :, pad : pad + h, pad : pad + w]
    return x


def _maxpool2(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n, c, h, w = x.shape
    h2, w2 = h // 2, w // 2
    trimmed = x[:, :, : h2 * 2, : w2 * 2]
    shaped = trimmed.reshape(n, c, h2, 2, w2, 2)
    pooled = shaped.max(axis=(3, 5))
    arg = shaped.reshape(n, c, h2, w2, 4).argmax(axis=-1)
    return pooled.astype(np.float32), arg.astype(np.int32)


def _maxpool2_backward(dout: np.ndarray, arg: np.ndarray, h: int, w: int) -> np.ndarray:
    n, c, h2, w2 = dout.shape
    dx = np.zeros((n, c, h2 * 2, w2 * 2), dtype=np.float32)
    flat = dx.reshape(n, c, h2, w2, 4)
    idx = arg.reshape(n, c, h2, w2)
    n_ix, c_ix, y_ix, x_ix = np.indices((n, c, h2, w2))
    flat[n_ix, c_ix, y_ix, x_ix, idx] = dout
    out = np.zeros((n, c, h, w), dtype=np.float32)
    out[:, :, : h2 * 2, : w2 * 2] = dx
    return out


@dataclass
class _Adam:
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    def __post_init__(self) -> None:
        self.m: dict[str, np.ndarray] = {}
        self.v: dict[str, np.ndarray] = {}
        self.t = 0

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        self.t += 1
        b1t = 1.0 - self.beta1**self.t
        b2t = 1.0 - self.beta2**self.t
        for name, value in params.items():
            g = grads[name]
            self.m[name] = self.beta1 * self.m.get(name, np.zeros_like(value)) + (
                1.0 - self.beta1
            ) * g
            self.v[name] = self.beta2 * self.v.get(name, np.zeros_like(value)) + (
                1.0 - self.beta2
            ) * (g * g)
            mhat = self.m[name] / b1t
            vhat = self.v[name] / b2t
            value -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


class TinyPatchCNN:
    """2×48×48 conv net: conv8 → pool → conv16 → pool → 32 → logit."""

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        gen = rng or np.random.default_rng(0)
        self.w1 = (gen.normal(0, 0.12, (8, 2, 3, 3))).astype(np.float32)
        self.b1 = np.zeros(8, dtype=np.float32)
        self.w2 = (gen.normal(0, 0.12, (16, 8, 3, 3))).astype(np.float32)
        self.b2 = np.zeros(16, dtype=np.float32)
        self.w3 = (gen.normal(0, 0.08, (32, 16 * 12 * 12))).astype(np.float32)
        self.b3 = np.zeros(32, dtype=np.float32)
        self.w4 = (gen.normal(0, 0.08, (1, 32))).astype(np.float32)
        self.b4 = np.zeros(1, dtype=np.float32)

    def _params(self) -> dict[str, np.ndarray]:
        return {
            "w1": self.w1,
            "b1": self.b1,
            "w2": self.w2,
            "b2": self.b2,
            "w3": self.w3,
            "b3": self.b3,
            "w4": self.w4,
            "b4": self.b4,
        }

    def copy(self) -> "TinyPatchCNN":
        other = TinyPatchCNN(np.random.default_rng(0))
        for name, value in self._params().items():
            setattr(other, name, value.copy())
        return other

    def _conv(self, x: np.ndarray, w: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n, c, h, width = x.shape
        col, oh, ow = _im2col(x, w.shape[2], w.shape[3], stride=1, pad=1)
        out = col @ w.reshape(w.shape[0], -1).T + b
        return out.reshape(n, oh, ow, w.shape[0]).transpose(0, 3, 1, 2), col

    def forward(self, x: np.ndarray) -> np.ndarray:
        logits, _cache = self._forward(x)
        return logits

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        c1, col1 = self._conv(x, self.w1, self.b1)
        a1 = _relu(c1)
        p1, arg1 = _maxpool2(a1)
        c2, col2 = self._conv(p1, self.w2, self.b2)
        a2 = _relu(c2)
        p2, arg2 = _maxpool2(a2)
        n = int(p2.shape[0])
        flat = p2.reshape(n, -1)
        h = _relu(flat @ self.w3.T + self.b3)
        logits = (h @ self.w4.T + self.b4).reshape(n)
        cache = {
            "x": x,
            "col1": col1,
            "c1": c1,
            "a1": a1,
            "p1": p1,
            "arg1": arg1,
            "col2": col2,
            "c2": c2,
            "a2": a2,
            "p2": p2,
            "arg2": arg2,
            "flat": flat,
            "h": h,
        }
        return logits, cache

    def _conv_backward(
        self,
        dout: np.ndarray,
        col: np.ndarray,
        w: np.ndarray,
        x_shape: tuple[int, ...],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n, cout, oh, ow = dout.shape
        dout_col = dout.transpose(0, 2, 3, 1).reshape(n * oh * ow, cout)
        dw = (dout_col.T @ col).reshape(w.shape)
        db = dout_col.sum(axis=0)
        dcol = dout_col @ w.reshape(cout, -1)
        n_in, cin, h, width = x_shape
        dx = _col2im(dcol, n_in, cin, h, width, w.shape[2], w.shape[3], 1, 1, oh, ow)
        return dx.astype(np.float32), dw.astype(np.float32), db.astype(np.float32)

    def _backward(
        self,
        dlogits: np.ndarray,
        cache: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        n = int(dlogits.shape[0])
        dh = dlogits.reshape(n, 1) @ self.w4
        dw4 = dlogits.reshape(n, 1).T @ cache["h"]
        db4 = np.array([float(dlogits.sum())], dtype=np.float32)
        dh_relu = dh * (cache["h"] > 0)
        dw3 = dh_relu.T @ cache["flat"]
        db3 = dh_relu.sum(axis=0)
        dflat = dh_relu @ self.w3
        dp2 = dflat.reshape(cache["p2"].shape)
        da2 = _maxpool2_backward(dp2, cache["arg2"], cache["a2"].shape[2], cache["a2"].shape[3])
        dc2 = da2 * (cache["c2"] > 0)
        dx2, dw2, db2 = self._conv_backward(dc2, cache["col2"], self.w2, cache["p1"].shape)
        da1 = _maxpool2_backward(dx2, cache["arg1"], cache["a1"].shape[2], cache["a1"].shape[3])
        dc1 = da1 * (cache["c1"] > 0)
        _dx1, dw1, db1 = self._conv_backward(dc1, cache["col1"], self.w1, cache["x"].shape)
        return {
            "w1": dw1,
            "b1": db1,
            "w2": dw2,
            "b2": db2,
            "w3": dw3.astype(np.float32),
            "b3": db3.astype(np.float32),
            "w4": dw4.astype(np.float32),
            "b4": db4.astype(np.float32),
        }

    def predict_proba(self, channels: np.ndarray, batch_size: int = 32) -> np.ndarray:
        if channels.shape[0] == 0:
            return np.empty((0,), dtype=np.float64)
        x = _resize_batch(channels)
        scores = []
        for start in range(0, x.shape[0], int(batch_size)):
            logits = self.forward(x[start : start + int(batch_size)])
            scores.append(_sigmoid(logits))
        return np.concatenate(scores).astype(np.float64)

    def fit(
        self,
        channels: np.ndarray,
        y: np.ndarray,
        epochs: int = 8,
        batch_size: int = 32,
        lr: float = 1e-3,
        rng: np.random.Generator | None = None,
    ) -> "TinyPatchCNN":
        gen = rng or np.random.default_rng(0)
        x = _resize_batch(channels)
        labels = np.asarray(y, dtype=np.float32).reshape(-1)
        n_pos = max(float(labels.sum()), 1.0)
        n_neg = max(float(len(labels) - labels.sum()), 1.0)
        pos_weight = n_neg / n_pos
        opt = _Adam(lr=lr)
        order = np.arange(x.shape[0])
        for _epoch in range(int(epochs)):
            gen.shuffle(order)
            for start in range(0, x.shape[0], int(batch_size)):
                idx = order[start : start + int(batch_size)]
                if idx.size == 0:
                    continue
                batch = x[idx]
                target = labels[idx]
                logits, cache = self._forward(batch)
                proba = _sigmoid(logits)
                # dL/dlogit for weighted BCE
                weight = np.where(target > 0.5, pos_weight, 1.0).astype(np.float32)
                dlogits = ((proba - target) * weight) / float(idx.size)
                grads = self._backward(dlogits.astype(np.float32), cache)
                opt.step(self._params(), grads)
        return self


def train_tiny_cnn(
    channels: np.ndarray,
    y: np.ndarray,
    epochs: int = 8,
    rng: np.random.Generator | None = None,
) -> TinyPatchCNN:
    model = TinyPatchCNN(rng)
    return model.fit(channels, y, epochs=epochs, rng=rng)
