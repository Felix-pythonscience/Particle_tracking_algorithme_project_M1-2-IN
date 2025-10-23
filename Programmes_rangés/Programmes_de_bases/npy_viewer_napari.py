"""Simple napari-based viewer for 256x256 .npy images.

Usage:
    python npy_viewer_napari.py

Select a folder containing .npy images when prompted. Use Right/Left arrow keys to navigate.

Dependencies:
    pip install napari[all]

This script uses tkinter only to present a folder selection dialog, and relies on napari
for fast image display (OpenGL-backed).
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import numpy as np
import napari
import sys


def choose_folder():
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title='Select folder containing .npy images')
    root.destroy()
    return Path(folder) if folder else None


class NapariNpyViewer:
    def __init__(self, folder: Path):
        self.folder = folder
        # collect .npy files (recursive)
        self.files = sorted([p for p in folder.rglob('*.npy')])
        if not self.files:
            raise RuntimeError(f'No .npy files found in {folder}')
        self.index = 0

        # create napari viewer
        self.viewer = napari.Viewer()
        self.image_layer = None

        # load first image
        self.load(self.index)

        # key bindings
        # functions receive the viewer as first arg
        @self.viewer.bind_key('Right')
        def _next(viewer):
            self.next()

        @self.viewer.bind_key('Left')
        def _prev(viewer):
            self.prev()

        @self.viewer.bind_key('q')
        def _quit(viewer):
            viewer.close()

    def load(self, idx: int):
        path = self.files[idx]
        arr = np.load(path, allow_pickle=False)
        # warn if shape unexpected
        if arr.ndim != 2 or arr.shape != (256, 256):
            messagebox.showwarning('Shape warning', f'{path.name}: array shape {arr.shape} (expected (256,256))')
        if self.image_layer is None:
            self.image_layer = self.viewer.add_image(arr, name=path.name, colormap='gray')
        else:
            # update data in-place
            self.image_layer.data = arr
            self.image_layer.name = path.name
        # update title
        try:
            self.viewer.window.qt_viewer.setWindowTitle(f"{path.name} ({idx+1}/{len(self.files)})")
        except Exception:
            # fallback
            pass

    def next(self):
        if self.index < len(self.files) - 1:
            self.index += 1
            self.load(self.index)

    def prev(self):
        if self.index > 0:
            self.index -= 1
            self.load(self.index)


def main():
    folder = choose_folder()
    if folder is None:
        print('No folder selected; exiting.')
        return
    # Try starting napari viewer; on any error fall back to the Tkinter viewer
    try:
        NapariNpyViewer(folder)
        napari.run()
    except Exception as exc:
        # Inform the user and fallback to the matplotlib/tkinter viewer included in the package
        print('Napari viewer failed to start:', exc)
        try:
            from .npy_viewer import NpyViewer
        except Exception:
            # try relative import fallback when running as script
            from npy_viewer import NpyViewer
        messagebox.showwarning('Napari error', f'Napari failed to start and the Tkinter fallback will be used.\n\nError: {exc}')
        # Launch tkinter viewer
        app = NpyViewer()
        app.mainloop()


if __name__ == '__main__':
    main()
