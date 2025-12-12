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
    QHBoxLayout, QFileDialog, QGroupBox, QGridLayout, QSizePolicy, QProgressBar
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
    progress = Signal(float, str)   # progress (0.0-1.0), message
    def __init__(self, file_path: str, discr_crit=None, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        # discrimination criteria passed to the counter (tweakable)
        self.discr_crit = discr_crit or {"electron_muon": {}}

    def run(self):
        # Called in the worker thread. Keep it minimal and return plain Python objects.
        try:
            # build a progress callback that emits the Qt signal from the worker thread
            def cb(progress, message=""):
                try:
                    # ensure numeric progress between 0 and 1
                    p = float(progress) if progress is not None else 0.0
                    if p < 0.0: p = 0.0
                    if p > 1.0: p = 1.0
                except Exception:
                    p = 0.0
                self.progress.emit(p, str(message or ""))
            # pass QThread interruption checker as stop_requested
            res = compteur_particles_optimized(file=self.file_path,
                                              discrimination_criteria=self.discr_crit,
                                              progress_bar=False,
                                              progress_callback=cb,
                                              stop_requested=self.isInterruptionRequested)
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
        self.progress_phase = "slicing"
        self.last_pct = 0
        self.cancelled = False

        # central widget + main vertical layout
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        # réduire l'espacement vertical pour rapprocher les widgets (status / progressbar)
        main_layout.setSpacing(6)
        main_layout.setContentsMargins(8, 8, 8, 8)

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

        # Bouton d'annulation, actif uniquement pendant un traitement
        self.btn_cancel = QPushButton("Annuler")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel_counting)
        ctrl_layout.addWidget(self.btn_cancel)

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
        # augmenter la police et rendre le label plus compact (rapproché de la progress bar)
        self.status.setStyleSheet("font-size:14px; font-weight:600; color: #111;")
        # supprimer marges internes si présentes
        self.status.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.status, alignment=Qt.AlignLeft)

        # Zone de progression (slicing + comptage)
        self.progress_title = QLabel("Barres de progression")
        self.progress_title.setStyleSheet("font-size:12px; font-weight:600; color: #111;")
        main_layout.addWidget(self.progress_title, alignment=Qt.AlignLeft)

        self.slice_label = QLabel("Slicing")
        self.slice_label.setStyleSheet("font-size:12px; font-weight:600; color: #111; margin-top:4px;")
        main_layout.addWidget(self.slice_label, alignment=Qt.AlignLeft)

        self.slice_progress = QProgressBar()
        self.slice_progress.setRange(0, 100)
        self.slice_progress.setValue(0)
        self.slice_progress.setTextVisible(True)
        self.slice_progress.setFixedHeight(18)
        self.slice_progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #bbb;
                background: #ffffff;
                height: 18px;
                border-radius: 4px;
                color: #000000;
                padding: 0px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4caf50;
                margin: 0px;
            }
        """)
        self.slice_progress.setFormat("%p%")
        main_layout.addWidget(self.slice_progress)

        self.count_label = QLabel("Comptage")
        self.count_label.setStyleSheet("font-size:12px; font-weight:600; color: #111; margin-top:4px;")
        main_layout.addWidget(self.count_label, alignment=Qt.AlignLeft)

        self.count_progress = QProgressBar()
        self.count_progress.setRange(0, 100)
        self.count_progress.setValue(0)
        self.count_progress.setTextVisible(True)
        self.count_progress.setFixedHeight(18)
        self.count_progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #bbb;
                background: #ffffff;
                height: 18px;
                border-radius: 4px;
                color: #000000;
                padding: 0px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4caf50;
                margin: 0px;
            }
        """)
        self.count_progress.setFormat("%p%")
        main_layout.addWidget(self.count_progress)
        

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
        # reset progress bars
        for bar in (self.slice_progress, self.count_progress):
            bar.setValue(0)
            bar.setFormat("0%")
        self.progress_phase = "slicing"
        self.last_pct = 0
        self.cancelled = False
        self.worker = CountWorker(self.selected_file)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.progress.connect(self._on_progress)   # connect progress
        self.worker.start()
        self.btn_cancel.setEnabled(True)

    def cancel_counting(self):
        """Demande l'annulation du comptage en cours."""
        if self.worker is None or not self.worker.isRunning():
            return
        self.status.setText("Annulation en cours...")
        self.btn_cancel.setEnabled(False)
        try:
            self.worker.requestInterruption()
            # Optionnel : indiquer dans la barre de progression
            self.slice_progress.setFormat("Annulation demandée...")
            self.count_progress.setFormat("Annulation demandée...")
            self.cancelled = True
        except Exception:
            pass

    def _on_finished(self, results: dict):
        """Update the result labels with the counts returned by the worker."""
        counts = results.get('Counts', {}) if isinstance(results, dict) else {}
        for k, lbl in self.result_labels.items():
            v = counts.get(k, '-')
            lbl.setText(str(v))
        # Si annulé, on laisse les barres à leur valeur courante
        if self.cancelled:
            self.status.setText("Cancelled")
            self.slice_progress.setFormat("Annulé")
            self.count_progress.setFormat("Annulé")
        else:
            self.status.setText("Done")
            self.slice_progress.setValue(100)
            self.slice_progress.setFormat("Terminé")
            self.count_progress.setValue(100)
            self.count_progress.setFormat("Terminé")
        self.btn_run.setEnabled(True)
        self.btn_select.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.worker = None
        self.progress_phase = "slicing"
        self.last_pct = 0
        self.cancelled = False

    def _on_error(self, msg: str):
        """Handle worker errors by reporting and re-enabling the UI."""
        self.status.setText(f"Error: {msg}")
        self.btn_run.setEnabled(True)
        self.btn_select.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.worker = None
        self.progress_phase = "slicing"
        self.last_pct = 0
        self.cancelled = False

    def _on_progress(self, progress: float, message: str):
        """Update the status bar with progress from the worker.

        The message previously shown in self.status is now displayed inside the progress bar.
        self.status keeps higher-level state (Ready / Running / Done / Error).
        """
        # clamp & percentage
        try:
            p = max(0.0, min(1.0, float(progress)))
            pct = int(round(p * 100))
        except Exception:
            p = 0.0
            pct = 0

        # Déterminer la phase (slicing ou comptage)
        msg = str(message or "")
        lower_msg = msg.lower()
        slice_kw = ("slice", "slicing", "preproc", "pre-proc", "extraction", "cluster")
        count_kw = ("count", "compt", "analyse", "counting", "compteur", "particle")

        if any(k in lower_msg for k in slice_kw):
            self.progress_phase = "slicing"
        elif any(k in lower_msg for k in count_kw):
            self.progress_phase = "counting"

        target_bar = self.slice_progress if self.progress_phase == "slicing" else self.count_progress

        # Mettre à jour la barre choisie
        target_bar.setValue(pct)
        display_msg = f"{msg} ({pct}%)" if msg else f"{pct}%"
        target_bar.setFormat(display_msg)
        self.last_pct = pct

def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
