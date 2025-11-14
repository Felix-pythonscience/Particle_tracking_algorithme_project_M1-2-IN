"""
PySide6 GUI skeleton for the manual classification tool.

This module provides a simple Qt application that permits:
- selecting a data file (your saved .npy cluster file or raw input),
- launching the optimized counter (compteur_particles_optimized) in a background thread,
- previewing the first cluster (if the selected .npy matches the project format),
- showing four result zones for alpha/electrons/muons/gamma counts.

Integration notes (where to extend):
- Replace preview extraction in `_try_preview` to use your slice/reader if you want
  to preview a slice built from a .t3pa file.
- If you want to run counting on an already-loaded image array instead of a file,
  modify `CountWorker.run()` to pass the image directly to `compteur_particles_optimized`.

UI behaviour summary:
- Press "Select file..." to pick a file. If it's a cluster .npy produced by the
  project's format, the first cluster pixels will be shown in the preview.
- Press "Run counting" to start the background job. The UI disables buttons while running
  and re-enables them when finished. Results are shown in the four result boxes.

"""

import numpy as np
from pathlib import Path
import sys
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from Programmes_de_bases.compteur import compteur_particles_optimized, compteur_particles

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QFileDialog, QGroupBox, QGridLayout, QSizePolicy
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
import os


def numpy_to_qimage(arr: np.ndarray) -> QImage:
    """Convert a 2D uint8 numpy array (H, W) to a QImage (Format_Grayscale8).

    The result is a copy so the underlying numpy array can be released safely.
    """
    h, w = arr.shape
    arr8 = np.ascontiguousarray(np.clip(arr, 0, 255).astype(np.uint8))
    return QImage(arr8.data, w, h, w, QImage.Format_Grayscale8).copy()


class CountWorker(QThread):
    """Worker thread that runs the optimized counting function.

    It emits `finished` with the results dict on success, or `error` with a message on failure.
    Running in a QThread keeps the GUI responsive for long jobs.
    """
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, file_path: str, discr_crit=None, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        # discrimination criteria passed to the counter (tweakable)
        self.discr_crit = discr_crit or {"electron_muon": {}}

    def run(self):
        # Called in the worker thread. Keep it minimal and return plain Python objects.
        try:
            res = compteur_particles_optimized(file=self.file_path,
                                              discrimination_criteria=self.discr_crit,
                                              progress_bar=False)
            self.finished.emit(res or {})
        except Exception as e:
            # Send readable error to the GUI thread
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    """Main application window.

    Layout overview:
    - Top row: Select file button, filename label, Run button
    - Middle: left = preview, right = 4 result zones (alpha/electrons/muons/gamma)
    - Bottom: status bar (simple QLabel)
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tri - GUI Qt (PySide6)")
        self.resize(1000, 700)

        # state
        self.selected_file = None
        self.worker = None

        # central widget + main vertical layout
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # Controls row
        ctrl_layout = QHBoxLayout()
        self.btn_select = QPushButton("Select file...")
        self.btn_select.clicked.connect(self.select_file)
        ctrl_layout.addWidget(self.btn_select)

        self.lbl_file = QLabel("No file selected")
        self.lbl_file.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        ctrl_layout.addWidget(self.lbl_file)

        self.btn_run = QPushButton("Run counting")
        self.btn_run.clicked.connect(self.run_counting)
        ctrl_layout.addWidget(self.btn_run)

        main_layout.addLayout(ctrl_layout)

        # Middle area: preview + results
        mid_layout = QHBoxLayout()

        # Preview box: shows a small 256x256 preview scaled up
        preview_box = QGroupBox("Image preview")
        preview_layout = QVBoxLayout(preview_box)
        self.lbl_preview = QLabel()
        self.lbl_preview.setFixedSize(512, 512)
        self.lbl_preview.setStyleSheet("background: #111; border: 1px solid #444")
        preview_layout.addWidget(self.lbl_preview, alignment=Qt.AlignCenter)
        mid_layout.addWidget(preview_box)

        # Results box: 4 labeled zones for the main counters
        results_box = QGroupBox("Results")
        results_layout = QGridLayout(results_box)

        self.result_labels = {}
        names = ["alpha", "electrons", "muons", "gamma"]
        for i, name in enumerate(names):
            g = QGroupBox(name.capitalize())
            v = QVBoxLayout(g)
            lbl = QLabel("-")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("font-size: 18px; font-weight: bold;")
            v.addWidget(lbl)
            results_layout.addWidget(g, i // 2, i % 2)
            self.result_labels[name] = lbl

        mid_layout.addWidget(results_box)
        main_layout.addLayout(mid_layout)

        # Simple status label
        self.status = QLabel("Ready")
        main_layout.addWidget(self.status)

    # ------------------ UI actions ------------------
    def select_file(self):
        """Open a file dialog and set the selected file.

        The preview routine expects the project's .npy format (arr[2] a dict with 'pixels' Nx2).
        Customize this to preview raw .t3pa files if needed by calling your reader/slicer.
        """
        p, _ = QFileDialog.getOpenFileName(self, "Select data file", os.getcwd(), "All files (*);;NumPy files (*.npy);;t3pa files (*.t3pa)")
        if not p:
            return
        self.selected_file = p
        self.lbl_file.setText(Path(p).name)
        self._try_preview(p)

    def _try_preview(self, path: str):
        """Try to make a quick 256x256 preview from a saved .npy cluster file.

        If the file isn't recognized, the preview area shows "No preview".
        """
        try:
            if path.lower().endswith('.npy'):
                arr = np.load(path, allow_pickle=True)
                if len(arr) > 2:
                    first = arr[2]
                    pixels = first.get('pixels', None) if isinstance(first, dict) else None
                    if pixels is not None:
                        img = np.zeros((256,256), dtype=np.uint8)
                        xs = np.clip(pixels[:,0].astype(int), 0, 255)
                        ys = np.clip(pixels[:,1].astype(int), 0, 255)
                        img[ys, xs] = 255
                        qimg = numpy_to_qimage(img)
                        pix = QPixmap.fromImage(qimg).scaled(self.lbl_preview.size(), Qt.KeepAspectRatio)
                        self.lbl_preview.setPixmap(pix)
                        return
        except Exception:
            # Any failure here is non-fatal for the GUI
            pass
        self.lbl_preview.clear()
        self.lbl_preview.setText("No preview")

    def run_counting(self):
        """Start the background worker to run compteur_particles_optimized.

        The GUI buttons are disabled while the worker runs. Results are handled
        in _on_finished or _on_error which are executed in the GUI thread.
        """
        if not self.selected_file:
            self.status.setText("Select a file first")
            return
        self.btn_run.setEnabled(False)
        self.btn_select.setEnabled(False)
        self.status.setText("Running...")
        self.worker = CountWorker(self.selected_file)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_finished(self, results: dict):
        """Update the result labels with the counts returned by the worker."""
        counts = results.get('Counts', {}) if isinstance(results, dict) else {}
        for k, lbl in self.result_labels.items():
            v = counts.get(k, '-')
            lbl.setText(str(v))
        self.status.setText("Done")
        self.btn_run.setEnabled(True)
        self.btn_select.setEnabled(True)

    def _on_error(self, msg: str):
        """Handle worker errors by reporting and re-enabling the UI."""
        self.status.setText(f"Error: {msg}")
        self.btn_run.setEnabled(True)
        self.btn_select.setEnabled(True)


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
import numpy as np
from pathlib import Path
import sys
from PySide6.QtWidgets import QApplication, QMainWindow
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from Programmes_de_bases.compteur import compteur_particles_optimized,compteur_particles





# empty_window.py



def main():
    app = QApplication(sys.argv)
    win = QMainWindow()
    win.setWindowTitle("Fenêtre PySide6 vide")
    win.resize(1400, 1000)      # largeur x hauteur en pixels
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()