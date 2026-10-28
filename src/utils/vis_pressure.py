"""
Shared pressure-map visualization utilities.

Used by both inference_tactile_parallel.py and inference_egodex.py so that
all inference outputs share an identical visual style.
"""
import cv2
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap

# Matches inference_tactile_parallel.py colormap exactly
PRESSURE_CMAP = LinearSegmentedColormap.from_list(
    'pressure', ['blue', 'cyan', 'yellow', 'red'])
PRESS_W: int = 720
PRESS_H: int = 360

# ── Physical glove sensor NaN masks (21x21) ──────────────────────────────────
# Extracted from TouchAnything HDF5 ground-truth pressure grids.
# True = no sensor at that position (NaN in GT data).
# Left and right hands differ because the glove layout mirrors between hands.
# These masks are FIXED across all trajectories / datasets.
_NAN_L = np.array([
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,1,1,1,1],
    [1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,1,1,1,1],
    [1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,1,1,1,1],
    [1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1],
    [1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1],
    [1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1],
    [1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1],
    [1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,1,1,1,1],
    [1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
], dtype=bool)

_NAN_R = np.array([
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1],
    [1,1,1,1,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1],
    [1,1,1,1,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1],
    [1,1,1,1,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1],
    [1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1],
    [1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1],
    [1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1],
    [1,1,1,1,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1],
    [1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1],
    [1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
], dtype=bool)

SENSOR_NAN_MASK_L: np.ndarray = _NAN_L   # True = no sensor
SENSOR_NAN_MASK_R: np.ndarray = _NAN_R


def pred_to_21(pred_s: np.ndarray, nan_mask: np.ndarray) -> np.ndarray:
    """
    Resize model prediction to 21x21 and apply the physical sensor NaN mask.

    Args:
        pred_s:   (S, S) float32 prediction in [0, 1]
        nan_mask: (21, 21) bool — True where no sensor exists

    Returns:
        (21, 21) float32, NaN at non-sensor positions
    """
    if pred_s.shape != (21, 21):
        pred_21 = cv2.resize(pred_s, (21, 21), interpolation=cv2.INTER_LINEAR)
    else:
        pred_21 = pred_s.copy()
    pred_21 = np.clip(pred_21, 0.0, 1.0)
    pred_21[nan_mask] = np.nan
    return pred_21


def fig_to_bgr(fig: Figure, w: int, h: int) -> np.ndarray:
    """Matplotlib Figure -> (h, w, 3) BGR numpy array."""
    fig.set_size_inches(w / 100, h / 100)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    img = buf.reshape(h, w, 4)[:, :, :3]
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def render_pressure_panel(left_grid: np.ndarray,
                          right_grid: np.ndarray,
                          title: str = 'Predicted Pressure',
                          w: int = PRESS_W,
                          h: int = PRESS_H,
                          vmax: float = 1.0,
                          global_max: float = None) -> np.ndarray:
    """
    Render a left/right hand pressure-map panel.

    Args:
        left_grid:  (21, 21) float32 — left-hand pressure grid; NaN = no sensor
        right_grid: (21, 21) float32 — right-hand pressure grid
        title:      panel title string
        w, h:       output image size in pixels
        vmax:       colorbar maximum value
        global_max: optional scalar annotated in the bottom-right corner

    Returns:
        (h, w, 3) BGR uint8 image
    """
    fig = Figure(facecolor='#1a1a2e')
    ax_l = fig.add_subplot(1, 2, 1)
    ax_r = fig.add_subplot(1, 2, 2)

    for ax, grid, label in [(ax_l, left_grid, 'Left Hand'),
                             (ax_r, right_grid, 'Right Hand')]:
        ax.set_facecolor('#111122')
        masked = np.ma.masked_invalid(grid.astype(np.float32))
        im = ax.imshow(masked, cmap=PRESSURE_CMAP, vmin=0, vmax=vmax,
                       interpolation='nearest', aspect='auto', origin='upper')
        ax.set_title(label, fontsize=9, color='#cccccc', pad=4)
        ax.axis('off')
        cbar_label = (f'Pressure (0-{vmax:.2f})' if vmax <= 1.0
                      else f'Pressure (0-{vmax:.0f})')
        cbar = fig.colorbar(im, ax=ax, label=cbar_label, shrink=0.7, fraction=0.04)
        cbar.ax.tick_params(labelsize=6, colors='#aaaaaa')
        cbar.ax.yaxis.label.set_size(7)
        cbar.ax.yaxis.label.set_color('#aaaaaa')
        valid = grid[~np.isnan(grid)]
        peak_thresh = vmax * 0.05
        if len(valid) > 0 and valid.max() > peak_thresh:
            pos = np.unravel_index(np.nanargmax(grid), grid.shape)
            ax.plot(pos[1], pos[0], 'w+', markersize=7,
                    markeredgewidth=1.5, alpha=0.9)

    fig.suptitle(title, fontsize=10, color='#eeeeee', y=0.99)
    if global_max is not None:
        fmt = '.2f' if global_max <= 1.0 else '.1f'
        fig.text(0.98, 0.02, f'max={global_max:{fmt}}',
                 ha='right', va='bottom', fontsize=7, color='#888888')
    fig.subplots_adjust(left=0.03, right=0.92, top=0.88, bottom=0.05, wspace=0.15)
    return fig_to_bgr(fig, w, h)
