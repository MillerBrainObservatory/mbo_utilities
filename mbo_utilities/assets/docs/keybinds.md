# Keyboard Shortcuts

## Navigation

| `Left` / `Right` | Step T (frames) |
| `Up` / `Down` | Step Z (z-plane) |
| `Shift` + arrow | Jump 10 |
| `Space` | Play / pause |

## File

| `o` | Open file(s) |
| `Shift + O` | Open folder (incl. Suite2p output dirs) |
| `s` | Save As |

## View

| `v` | Reset vmin/vmax (auto-contrast frame) |
| `Shift + V` | Toggle auto-contrast on Z-change |
| `c` | Toggle scan-phase correction |
| `Shift + C` | Toggle sub-pixel scan-phase |
| `m` | Metadata viewer |
| `p` / `Enter` | Toggle side panel |

## Help

| `h` / `F1` | Open this help |
| `k` | Keybind cheatsheet (this list) |

## Mouse

- **Scroll**: zoom
- **Drag**: pan
- **Right-click**: context menu

## Tip: subset selection

In the Selection popup (Run tab), use a small range to iterate fast:

- Timepoints `1:500` — first 500 frames only
- Z-planes `1:14:2` — every other plane
- Channels `1:1` — single channel from a multi-channel array

Both **Save As** and **Run -> Suite2p** respect the current selection.
