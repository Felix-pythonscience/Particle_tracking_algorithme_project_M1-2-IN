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
import traceback
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from Programmes_de_bases_GUI.compteur import compteur_particles_optimized, compteur_particles

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QFileDialog, QGroupBox, QGridLayout, QSizePolicy, QProgressBar, QCheckBox,
    QComboBox,QSpinBox,QDoubleSpinBox

)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from superqt import QRangeSlider
import os


def numpy_to_qimage(arr: np.ndarray) -> QImage:
    """Convert a 2D uint8 numpy array (H, W) to a QImage (Format_Grayscale8).

    The result is a copy so the underlying numpy array can be released safely.
    """
    h, w = arr.shape
    arr8 = np.ascontiguousarray(np.clip(arr, 0, 255).astype(np.uint8))
    return QImage(arr8.data, w, h, w, QImage.Format_Grayscale8).copy()

def numpy_to_qimage_rgb(arr: np.ndarray) -> QImage:
    """Convert a 3D RGB numpy array (H, W, 3) to a QImage (Format_RGB888).

    The result is a copy so the underlying numpy array can be released safely.
    """
    h, w = arr.shape[:2]
    arr8 = np.ascontiguousarray(np.clip(arr, 0, 255).astype(np.uint8))
    return QImage(arr8.data, w, h, w * 3, QImage.Format_RGB888).copy()


class CountWorker(QThread):
    """Worker thread that runs the optimized counting function.

    It emits `finished` with the results dict on success, or `error` with a message on failure.
    Running in a QThread keeps the GUI responsive for long jobs.
    """
    finished = Signal(dict)
    error = Signal(str)
    progress = Signal(float, str)   # progress (0.0-1.0), message
    def __init__(self, file_path: str, discr_crit=None, parent=None,
                time_window=150,time_min = None, time_max = None, return_slice = False, return_images = False,
                is_global_count = True, return_data_t_max=False, single_window=False, pre_sliced_images=None, pre_slice_times=None):
        super().__init__(parent)
        self.file_path = file_path
        self.time_window = time_window
        self.time_min = time_min
        self.time_max = time_max
        self.return_slice = return_slice  
        self.return_images = return_images
        self.return_data_t_max = return_data_t_max
        self.single_window = single_window
        self.pre_sliced_images = pre_sliced_images  # Pre-calculated slices for visualisation mode
        self.pre_slice_times = pre_slice_times  # Start times of pre-calculated slices
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
            
            # If pre-sliced images are provided, use them directly (visualisation mode with stored slices)
            if self.pre_sliced_images is not None:
                cb(0.5, "Processing pre-sliced images")
                # Filter slices that fall within [t_min, t_max] time range
                from Programmes_de_bases_GUI.compteur import compteur_particles
                import numpy as np
                
                filtered_slices = []
                if self.pre_slice_times is not None and len(self.pre_slice_times) > 0:
                    # Filter slices based on t_min and t_max
                    for img, t_start in zip(self.pre_sliced_images, self.pre_slice_times):
                        if self.time_min <= t_start < self.time_max:
                            filtered_slices.append(img)
                else:
                    # No time info, use all slices
                    filtered_slices = self.pre_sliced_images
                
                print(f"Filtered {len(filtered_slices)} slices from {len(self.pre_sliced_images)} total (t_min={self.time_min}, t_max={self.time_max})")
                
                totals = {"alpha": 0, "electrons": 0, "muons": 0, "gamma": 0}
                # Build composite RGB image where last particle wins
                composite_rgb = np.zeros((256, 256, 3), dtype=np.uint8) if self.return_images else None
                original_composite = np.zeros((256, 256), dtype=np.float32) if self.return_images else None
                
                for img in filtered_slices:
                    results = compteur_particles(img, is_slice=True,
                                return_images=self.return_images,
                                discrimination_criteria=self.discr_crit)
                    counts = results.get("Counts", {})
                    for k in totals.keys():
                        try:
                            totals[k] += int(counts.get(k, 0) or 0)
                        except Exception:
                            pass
                    
                    # Build composite image where last particle overwrites previous (last wins)
                    if self.return_images and results.get("Images") is not None:
                        slice_images = results.get("Images")
                        img_alpha = slice_images.get('alpha', np.zeros((256, 256)))
                        img_tracks = slice_images.get('tracks', np.zeros((256, 256)))
                        img_gamma = slice_images.get('gamma', np.zeros((256, 256)))
                        img_original = slice_images.get('original', np.zeros((256, 256)))
                        
                        # Update original composite (always accumulate for grayscale mode)
                        original_composite += img_original
                        
                        # For RGB mode: last particle wins (overwrites previous color)
                        # Priority order for simultaneous detections: gamma > tracks > alpha
                        alpha_pixels = img_alpha > 0
                        tracks_pixels = img_tracks > 0
                        gamma_pixels = img_gamma > 0
                        
                        # Apply colors in order (later ones overwrite earlier ones)
                        composite_rgb[alpha_pixels, 0] = 255  # Red for alpha
                        composite_rgb[alpha_pixels, 1] = 0
                        composite_rgb[alpha_pixels, 2] = 0
                        
                        composite_rgb[tracks_pixels, 0] = 0
                        composite_rgb[tracks_pixels, 1] = 0
                        composite_rgb[tracks_pixels, 2] = 255  # Blue for tracks
                        
                        composite_rgb[gamma_pixels, 0] = 0
                        composite_rgb[gamma_pixels, 1] = 255  # Green for gamma
                        composite_rgb[gamma_pixels, 2] = 0
                
                # Package results in the expected format
                if self.return_images:
                    accumulated_images = {
                        'alpha': (composite_rgb[:, :, 0] > 0).astype(np.float32),
                        'tracks': (composite_rgb[:, :, 2] > 0).astype(np.float32),
                        'gamma': (composite_rgb[:, :, 1] > 0).astype(np.float32),
                        'original': np.clip(original_composite, 0, 1)
                    }
                else:
                    accumulated_images = None
                    
                cb(1.0, "Counting complete")
                res = {"Counts": totals, "Images": accumulated_images if self.return_images else None}
            else:
                # Normal path: process from file
                res = compteur_particles_optimized(file=self.file_path,
                                                  discrimination_criteria=self.discr_crit,
                                                  progress_bar=False,
                                                  progress_callback=cb,
                                                  d_time=self.time_window,
                                                  t_min=self.time_min, t_max=self.time_max,
                                                  return_slice=self.return_slice,
                                                  return_images=self.return_images,
                                                  return_data_t_max=self.return_data_t_max,
                                                  single_window=self.single_window,
                                                  stop_requested=self.isInterruptionRequested)
            self.finished.emit(res or {})
        except Exception as e:
            # Print full traceback to console for debugging
            print("\n" + "="*60)
            print("ERROR in CountWorker:")
            traceback.print_exc()
            print("="*60 + "\n")
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
        self.is_global_count = True  # track which set of result labels to update (0=visualisation, 1=global)
        self.global_slices = None  # Store slices from global counting for reuse in visualisation
        self.global_slice_times = None  # Store start times of each slice

        #Counting options
        self.time_window_unit = "ticks"
        self.time_window = 150 # valeur par défaut de 150 ticks
        self.t_min = 0.0  # borne inférieure de la plage temporelle
        self.t_max = 1.0  # borne supérieure de la plage temporelle (sera mise à jour après comptage)
        self.data_t_max = 1.0  # valeur réelle du t_max du fichier (obtenue après lecture)
        self.use_color_code = False  # activer le code couleur pour les particules
        
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

        self.btn_run = QPushButton("Run global counting")
        self.btn_run.clicked.connect(self.run_global_counting)
        ctrl_layout.addWidget(self.btn_run)

        # Bouton d'annulation, actif uniquement pendant un traitement
        self.btn_cancel = QPushButton("Annuler")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel_counting)
        ctrl_layout.addWidget(self.btn_cancel)

        main_layout.addLayout(ctrl_layout)

        # Middle area: preview + results
        mid_layout = QHBoxLayout()

        # Preview box: shows a 256x256 image scaled up to 512x512
        preview_box = QGroupBox("Image preview")
        preview_layout = QVBoxLayout(preview_box)
        self.lbl_preview = QLabel()
        self.lbl_preview.setFixedSize(512, 512)
        self.lbl_preview.setStyleSheet("background: #111; border: 1px solid #444")
        self.lbl_preview.setScaledContents(True)
        preview_layout.addWidget(self.lbl_preview, alignment=Qt.AlignCenter)

        
        
       
        mid_layout.addWidget(preview_box)


        # BOX 4 : Options and results
        options_and_results_box = QGroupBox("Options and results")
        options_and_results_layout = QVBoxLayout(options_and_results_box)
                
                
        # Box 4.1 : time windows options for slicing
        time_windows_box = QGroupBox("Time windows")
        time_windows_layout = QVBoxLayout(time_windows_box)

        # -------- Nombre (time window) --------
        lbl_value = QLabel("Window size")
        time_windows_layout.addWidget(lbl_value)

        self.time_window_value = QDoubleSpinBox()
        self.time_window_value.setRange(1e-6, 10_000)
        self.time_window_value.setValue(150)  # valeur initiale
        time_windows_layout.addWidget(self.time_window_value)

        # Variable associée
        self.time_window = self.time_window_value.value()

        # Mise à jour automatique
        self.time_window_value.valueChanged.connect(self.on_time_window_changed)

        # -------- Menu déroulant --------
        lbl_unit = QLabel("unit")
        time_windows_layout.addWidget(lbl_unit)

        self.time_window_unit = QComboBox()
        self.time_window_unit.addItems(["ticks", "seconds (s)"])
        time_windows_layout.addWidget(self.time_window_unit)

        # Variable associée
        self.time_window_unit_value = self.time_window_unit.currentText()

        # Mise à jour automatique
        self.time_window_unit.currentTextChanged.connect(self.on_time_window_unit_changed)

        # Ajout au layout principal
        options_and_results_layout.addWidget(time_windows_box)
        options_and_results_layout.setStretch(0, 1)


        # Box 4.2 : Visulalisation options for results
        visualisation_box = QGroupBox("Visualisation")
        visualisation_layout = QVBoxLayout(visualisation_box)
        
        # Box 4.2.1 Options de couleurs et de choix de particules
        color_options_group = QGroupBox("Options de visualisation")
        color_options_layout = QVBoxLayout(color_options_group)
        # Bouton pour lancer la visualisation
        self.btn_visualisation = QPushButton("Visualisation")
        self.btn_visualisation.clicked.connect(self.run_visualisation_counting)
        color_options_layout.addWidget(self.btn_visualisation)
        
        # Checkbox pour activer le code couleur
        self.chk_color_code = QCheckBox("Code couleur pour particules")
        self.chk_color_code.setChecked(False)
        self.chk_color_code.stateChanged.connect(self.on_color_code_changed)
        color_options_layout.addWidget(self.chk_color_code)
        
        # Légende des couleurs
        legend_group = QGroupBox("Légende")
        legend_layout = QVBoxLayout(legend_group)
        legend_layout.setSpacing(2)
        legend_layout.setContentsMargins(4, 4, 4, 4)
        
        lbl_alpha = QLabel("🔴 Rouge : Alpha")
        lbl_alpha.setStyleSheet("color: red; font-weight: bold;")
        legend_layout.addWidget(lbl_alpha)
        
        lbl_gamma = QLabel("🟢 Vert : Gamma")
        lbl_gamma.setStyleSheet("color: green; font-weight: bold;")
        legend_layout.addWidget(lbl_gamma)
        
        lbl_tracks = QLabel("🔵 Bleu : Tracks (e⁻/μ)")
        lbl_tracks.setStyleSheet("color: blue; font-weight: bold;")
        legend_layout.addWidget(lbl_tracks)
        
        color_options_layout.addWidget(legend_group)
        color_options_layout.addStretch()

        visualisation_layout.addWidget(color_options_group)

        # box 4.2.2 Bornes temporelles pour la visualisation

        # Time range slider (t_min and t_max) avec deux curseurs sur une seule barre
        time_range_group = QGroupBox("Plage temporelle (t_min / t_max)")
        time_range_layout = QVBoxLayout(time_range_group)

        # Labels pour afficher les valeurs
        values_layout = QHBoxLayout()
        self.lbl_tmin_max_value = QLabel(f"t_min = {self.t_min:.3e} t_max = {self.t_max:.3e}")

        values_layout.addWidget(self.lbl_tmin_max_value)
        values_layout.addStretch()
        time_range_layout.addLayout(values_layout)

        # Range slider avec deux handles
        self.range_slider = QRangeSlider(Qt.Horizontal)
        self.range_slider.setRange(0, 1000)  # 0-1000 steps, mapped to 0.0-data_t_max
        self.range_slider.setValue((0, 1000))  # (min, max) initialement
        self.range_slider.valueChanged.connect(self.on_time_range_changed)
        time_range_layout.addWidget(self.range_slider)
        

        visualisation_layout.addWidget(time_range_group)
        
        options_and_results_layout.addWidget(visualisation_box)
        options_and_results_layout.setStretch(1, 2)

        # Results box: 4.3 labeled zones for the counters
        results_box = QGroupBox("Results")
        results_layout = QHBoxLayout(results_box)

        self.result_labels = {}
        names = ["alpha", "electrons", "muons", "gamma"]

        for name in names:
            g = QGroupBox(name.capitalize())
            v = QVBoxLayout(g)

            # Label Visualisation (bleu)
            lbl_visualisation = QLabel("Visualisation : -")
            lbl_visualisation.setAlignment(Qt.AlignCenter)
            lbl_visualisation.setStyleSheet(
                "font-size: 14px; font-weight: bold; color: blue;"
            )
            v.addWidget(lbl_visualisation)

            # Label Global (noir)
            lbl_global = QLabel("Global : -")
            lbl_global.setAlignment(Qt.AlignCenter)
            lbl_global.setStyleSheet(
                "font-size: 18px; font-weight: bold; color: black;"
            )
            v.addWidget(lbl_global)

            results_layout.addWidget(g)

            self.result_labels[name] = [lbl_visualisation, lbl_global]


        options_and_results_layout.addWidget(results_box)
        options_and_results_layout.setStretch(2, 1)

        mid_layout.addWidget(options_and_results_box)
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

    def on_time_window_changed(self, value):
        self.time_window = value * (1 if self.time_window_unit == "ticks" else 39806550.8)
        
    

    def on_time_window_unit_changed(self, unit):
        self.time_window_unit_value = unit
        self.time_window = self.time_window * (1 if unit == "ticks" else 39806550.8)

    def on_time_range_changed(self, values):
        """Update t_min and t_max from range slider (tuple of two values)."""
        min_val, max_val = values
        self.t_min = (min_val / 1000.0) * self.data_t_max
        self.t_max = (max_val / 1000.0) * self.data_t_max
        self.lbl_tmin_max_value.setText(f"t_min = {self.t_min:.3e}   t_max = {self.t_max:.3e}")
    
    def on_color_code_changed(self, state):
        """Active ou désactive le code couleur pour la visualisation."""
        self.use_color_code = bool(state)  # state est 0 (décoché) ou 2 (coché)
        print(f"Checkbox changée: state={state}, use_color_code={self.use_color_code}")

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

    def run_global_counting(self):
        """Start the background worker to run compteur_particles_optimized.

        The GUI buttons are disabled while the worker runs. Results are handled
        in _on_finished or _on_error which are executed in the GUI thread.
        """
        if not self.selected_file:
            self.status.setText("Select a file first")
            return
        
        # Reset stored slices to force re-slicing (e.g., if file changed)
        self.global_slices = None
        self.global_slice_times = None
        
        self.btn_run.setEnabled(False)
        self.btn_select.setEnabled(False)
        self.btn_visualisation.setEnabled(False)
        self.status.setText("Running...")
        # reset progress bars
        for bar in (self.slice_progress, self.count_progress):
            bar.setValue(0)
            bar.setFormat("0%")
        self.progress_phase = "slicing"
        self.last_pct = 0
        self.cancelled = False
        self.is_global_count = True
        # Use the same discrimination criteria for consistency
        discr_crit = {
            "electron_muon": {
                "solidity_threshold_alpha": 1,
                "eccentricity_threshold_alpha": 0.70,
                "area_threshold_alpha": 9,
                "eccentricity_threshold_muon": 0.99,
                "area_threshold_muon": 25,
            }
        }
        self.worker = CountWorker(self.selected_file, discr_crit=discr_crit, is_global_count=True,
                                  time_window=self.time_window, return_data_t_max=True,
                                  return_slice=True
                                )
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.progress.connect(self._on_progress)   # connect progress
        self.worker.start()
        self.btn_cancel.setEnabled(True)
    
    def run_visualisation_counting(self):
        """Lance le comptage sur une seule fenêtre [t_min, t_max] et affiche l'image composite."""
        if not self.selected_file:
            self.status.setText("Select a file first")
            return
        # Vérifier que data_t_max a été obtenu
        if self.data_t_max == 1.0:
            self.status.setText("Run Global counting first to get time range")
            return
        
        self.btn_run.setEnabled(False)
        self.btn_select.setEnabled(False)
        self.btn_visualisation.setEnabled(False)
        self.status.setText("Visualisation running...")
        # reset progress bars
        for bar in (self.slice_progress, self.count_progress):
            bar.setValue(0)
            bar.setFormat("0%")
        self.progress_phase = "slicing"
        self.last_pct = 0
        self.cancelled = False
        self.is_global_count = False  # mode visualisation
        
        # Use the same discrimination criteria for consistency
        discr_crit = {
            "electron_muon": {
                "solidity_threshold_alpha": 1,
                "eccentricity_threshold_alpha": 0.70,
                "area_threshold_alpha": 9,
                "eccentricity_threshold_muon": 0.99,
                "area_threshold_muon": 25,
            }
        }
        
        # Use stored slices if available (from global counting)
        if self.global_slices is not None:
            print(f"Using stored slices for visualisation")
            # Pas besoin d'utiliser le fichier, on utilise les slices pré-calculées
            self.worker = CountWorker(self.selected_file, discr_crit=discr_crit, is_global_count=False,
                                      return_images=True,  # récupérer les images filtrées
                                      time_min=self.t_min, time_max=self.t_max,  # Pass time bounds for filtering
                                      pre_sliced_images=self.global_slices,  # Utiliser les slices stockées
                                      pre_slice_times=self.global_slice_times)  # Utiliser les temps des slices
        else:
            # Fallback: calculer une seule fenêtre depuis le fichier
            print(f"No stored slices, computing single window from file")
            self.worker = CountWorker(self.selected_file, discr_crit=discr_crit, is_global_count=False,
                                      time_window=self.t_max - self.t_min,  # une seule fenêtre
                                      return_data_t_max=False,
                                      return_slice=False,
                                      return_images=True,  # récupérer les images filtrées
                                      time_min=self.t_min, time_max=self.t_max,
                                      single_window=True)  # mode fenêtre unique
        
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.progress.connect(self._on_progress)
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
        which_label = int(self.is_global_count)  # 0 pour Visualisation, 1 pour Global
        prefix = "Visualisation" if which_label == 0 else "Global"
        for category_name, label_pair in self.result_labels.items():
            count_value = counts.get(category_name, '-')
            label_pair[which_label].setText(f"{prefix} : {count_value}")
        
        # Si mode visualisation, afficher l'image composite dans preview
        if not self.is_global_count:
            print(f"Mode visualisation détecté, is_global_count={self.is_global_count}")
            images = results.get('Images', None)
            print(f"Images récupérées: {images is not None}")
            if images:
                print(f"Clés des images: {images.keys() if isinstance(images, dict) else 'pas un dict'}")
                composite, is_rgb = self._create_composite_image(images)
                print(f"Composite créé: shape={composite.shape if composite is not None else None}, is_rgb={is_rgb}")
                if composite is not None:
                    if is_rgb:
                        qimg = numpy_to_qimage_rgb(composite)
                    else:
                        qimg = numpy_to_qimage(composite)
                    pix = QPixmap.fromImage(qimg).scaled(self.lbl_preview.size(), Qt.KeepAspectRatio)
                    self.lbl_preview.setPixmap(pix)
                    print("Image affichée dans preview!")
                else:
                    print("Composite est None!")
            else:
                print("Pas d'images dans les résultats!")
        
        # Update slider ranges with real data t_max if available
        if 'data_t_max' in results and results['data_t_max'] is not None:
            self.data_t_max = float(results['data_t_max'])
            # Reset slider to use the full range
            self.t_min = 0.0
            self.t_max = self.data_t_max
            self.range_slider.setValue((0, 1000))
            self.lbl_tmin_max_value.setText(f"t_min = {self.t_min:.3e}   t_max = {self.t_max:.3e}")
        
        # If global count, store slices for reuse in visualisation
        if self.is_global_count:
            slices = results.get('Slices', None)
            slice_times = results.get('SliceTimes', None)
            if slices is not None:
                self.global_slices = slices
                self.global_slice_times = slice_times
                print(f"Stored {len(slices)} slices from global counting with {len(slice_times) if slice_times is not None else 0} times")
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
        self.btn_visualisation.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.worker = None
        self.progress_phase = "slicing"
        self.last_pct = 0
        self.cancelled = False

    def _create_composite_image(self, images: dict):
        """Crée une image composite RGB à partir des images filtrées.
        
        Args:
            images: Dict contenant 'alpha', 'tracks', 'gamma' (images 256x256)
        
        Returns:
            Tuple (image_array, is_rgb): image et booléen indiquant si c'est RGB
        """
        try:
            # Récupérer les images
            img_alpha = images.get('alpha', np.zeros((256, 256)))
            img_tracks = images.get('tracks', np.zeros((256, 256)))  # electrons + muons
            img_gamma = images.get('gamma', np.zeros((256, 256)))
            img_original = images.get('original', np.zeros((256, 256)))
            
            print(f"use_color_code = {self.use_color_code}")
            print(f"Alpha pixels non-nuls: {np.count_nonzero(img_alpha)}")
            print(f"Tracks pixels non-nuls: {np.count_nonzero(img_tracks)}")
            print(f"Gamma pixels non-nuls: {np.count_nonzero(img_gamma)}")
            print(f"Original pixels non-nuls: {np.count_nonzero(img_original)}")
            
            if self.use_color_code:
                print("Mode COULEUR activé!")
                # Code couleur: alpha=rouge, tracks=bleu, gamma=vert
                composite = np.zeros((256, 256, 3), dtype=np.uint8)
                composite[:, :, 0] = np.clip(img_alpha * 255, 0, 255).astype(np.uint8)  # Rouge pour alpha
                composite[:, :, 1] = np.clip(img_gamma * 255, 0, 255).astype(np.uint8)  # Vert pour gamma
                composite[:, :, 2] = np.clip(img_tracks * 255, 0, 255).astype(np.uint8)  # Bleu pour tracks
                print(f"Composite RGB créé, shape={composite.shape}, pixels non-nuls par canal: R={np.count_nonzero(composite[:,:,0])}, G={np.count_nonzero(composite[:,:,1])}, B={np.count_nonzero(composite[:,:,2])}")
                return composite, True  # Retourner l'image RGB
            else:
                print("Mode GRAYSCALE")
                # Grayscale: somme de toutes les détections
                return np.clip(img_original * 255, 0, 255).astype(np.uint8), False
        except Exception as e:
            print(f"Error creating composite image: {e}")
            traceback.print_exc()
            return None, False
    
    def _on_error(self, msg: str):
        """Handle worker errors by reporting and re-enabling the UI."""
        self.status.setText(f"Error: {msg}")
        self.btn_run.setEnabled(True)
        self.btn_select.setEnabled(True)
        self.btn_visualisation.setEnabled(True)
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
