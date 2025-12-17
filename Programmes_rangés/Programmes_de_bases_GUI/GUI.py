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
                
                totals = {"alpha": 0, "electrons": 0, "muons": 0, "gamma": 0}
                # Accumulate per-particle masks and original for grayscale preview
                accum_alpha = np.zeros((256, 256), dtype=np.float32) if self.return_images else None
                accum_electrons = np.zeros((256, 256), dtype=np.float32) if self.return_images else None
                accum_muons = np.zeros((256, 256), dtype=np.float32) if self.return_images else None
                accum_gamma = np.zeros((256, 256), dtype=np.float32) if self.return_images else None
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
                    
                    # Accumulate masks per particle type
                    if self.return_images and results.get("Images") is not None:
                        slice_images = results.get("Images")
                        img_alpha = slice_images.get('alpha', np.zeros((256, 256)))
                        img_electrons = slice_images.get('electrons', np.zeros((256, 256)))
                        img_muons = slice_images.get('muons', np.zeros((256, 256)))
                        img_gamma = slice_images.get('gamma', np.zeros((256, 256)))
                        img_original = slice_images.get('original', np.zeros((256, 256)))

                        # Accumulate masks (logical OR via max) and grayscale original
                        accum_alpha = np.maximum(accum_alpha, img_alpha)
                        accum_electrons = np.maximum(accum_electrons, img_electrons)
                        accum_muons = np.maximum(accum_muons, img_muons)
                        accum_gamma = np.maximum(accum_gamma, img_gamma)
                        original_composite += img_original
                
                # Package results in the expected format
                if self.return_images:
                    accumulated_images = {
                        'alpha': accum_alpha,
                        'electrons': accum_electrons,
                        'muons': accum_muons,
                        'gamma': accum_gamma,
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
        self.setWindowTitle("Particle Sorting - GUI (PySide6)")
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

        # Cancel button, active only during a run
        self.btn_cancel = QPushButton("Cancel")
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
        # Save preview button (saved PNG of currently displayed preview)
        btn_row = QHBoxLayout()
        self.save_preview_btn = QPushButton("Save image")
        self.save_preview_btn.setEnabled(False)
        self.save_preview_btn.clicked.connect(self._save_preview_image)
        btn_row.addWidget(self.save_preview_btn)
        btn_row.addStretch()
        preview_layout.addLayout(btn_row)

        
        
       
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
        # Ranges depend on unit: ticks -> [4 ticks (0.1 µs), 5e12 ticks]; seconds -> converted range
        self.ratio_ticks_per_second = 39806550.8  # 1 sec = 39806550.8 ticks
        self.time_window_value.setRange(4, 5e12)  # initial unit = ticks
        self.time_window_value.setDecimals(15)  # Allow very small decimal values (scientific notation)
        self.time_window_value.setValue(150)  # valeur initiale en ticks
        # Use a custom formatter for scientific notation
        self._setup_spinbox_scientific_notation(self.time_window_value)
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
        visualisation_box = QGroupBox("Visualization")
        visualisation_layout = QVBoxLayout(visualisation_box)
        
        # Box 4.2.1 Options de couleurs et de choix de particules
        color_options_group = QGroupBox("Visualization options")
        color_options_layout = QVBoxLayout(color_options_group)
        # Button to start visualization
        self.btn_visualisation = QPushButton("Visualize")
        self.btn_visualisation.clicked.connect(self.run_visualisation_counting)
        color_options_layout.addWidget(self.btn_visualisation)
        
        # Checkbox to enable particle color coding
        self.chk_color_code = QCheckBox("Particle color code")
        self.chk_color_code.setChecked(False)
        self.chk_color_code.stateChanged.connect(self.on_color_code_changed)
        color_options_layout.addWidget(self.chk_color_code)
        
        # Legend for colors
        legend_group = QGroupBox("Legend")
        legend_layout = QVBoxLayout(legend_group)
        legend_layout.setSpacing(2)
        legend_layout.setContentsMargins(4, 4, 4, 4)
        
        # Checkboxes + labels so the user can toggle visibility per particle type
        h_alpha = QHBoxLayout()
        self.chk_show_alpha = QCheckBox()
        self.chk_show_alpha.setChecked(True)
        self.chk_show_alpha.setToolTip("Show Alpha in visualization")
        lbl_alpha = QLabel("🟡 Yellow : Alpha")
        lbl_alpha.setStyleSheet("color: #dcbf00; font-weight: bold;")
        h_alpha.addWidget(self.chk_show_alpha)
        h_alpha.addWidget(lbl_alpha)
        h_alpha.addStretch()
        legend_layout.addLayout(h_alpha)

        h_muons = QHBoxLayout()
        self.chk_show_muons = QCheckBox()
        self.chk_show_muons.setChecked(True)
        self.chk_show_muons.setToolTip("Show Muons in visualization")
        lbl_muons = QLabel("🔴 Red : Muons")
        lbl_muons.setStyleSheet("color: red; font-weight: bold;")
        h_muons.addWidget(self.chk_show_muons)
        h_muons.addWidget(lbl_muons)
        h_muons.addStretch()
        legend_layout.addLayout(h_muons)

        h_elec = QHBoxLayout()
        self.chk_show_electrons = QCheckBox()
        self.chk_show_electrons.setChecked(True)
        self.chk_show_electrons.setToolTip("Show Electrons in visualization")
        lbl_elec = QLabel("🔵 Blue : Electrons")
        lbl_elec.setStyleSheet("color: blue; font-weight: bold;")
        h_elec.addWidget(self.chk_show_electrons)
        h_elec.addWidget(lbl_elec)
        h_elec.addStretch()
        legend_layout.addLayout(h_elec)

        h_gamma = QHBoxLayout()
        self.chk_show_gamma = QCheckBox()
        self.chk_show_gamma.setChecked(True)
        self.chk_show_gamma.setToolTip("Show Gamma in visualization")
        lbl_gamma = QLabel("🟢 Green : Gamma")
        lbl_gamma.setStyleSheet("color: green; font-weight: bold;")
        h_gamma.addWidget(self.chk_show_gamma)
        h_gamma.addWidget(lbl_gamma)
        h_gamma.addStretch()
        legend_layout.addLayout(h_gamma)
        # Connect checkbox changes to preview update
        try:
            self.chk_show_alpha.stateChanged.connect(self._on_legend_checkbox_changed)
            self.chk_show_muons.stateChanged.connect(self._on_legend_checkbox_changed)
            self.chk_show_electrons.stateChanged.connect(self._on_legend_checkbox_changed)
            self.chk_show_gamma.stateChanged.connect(self._on_legend_checkbox_changed)
        except Exception:
            pass
        
        color_options_layout.addWidget(legend_group)
        color_options_layout.addStretch()

        visualisation_layout.addWidget(color_options_group)

        # box 4.2.2 Bornes temporelles pour la visualisation

        # Time range slider (t_min and t_max) avec deux curseurs sur une seule barre
        time_range_group = QGroupBox("Time range (t_min / t_max)")
        time_range_layout = QVBoxLayout(time_range_group)

        # Labels pour afficher les valeurs
        values_layout = QHBoxLayout()
        self.lbl_tmin_max_value = QLabel(f"t_min = 0.000e+00 {self.time_window_unit.currentText()}   |   Duration = 0.000e+00 {self.time_window_unit.currentText()}   |   t_max = 0.000e+00 {self.time_window_unit.currentText()}")
        self.lbl_tmin_max_value.setAlignment(Qt.AlignCenter)
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


            # Label Visualization (blue)
            lbl_visualisation = QLabel("Visualization : -")
            lbl_visualisation.setAlignment(Qt.AlignCenter)
            lbl_visualisation.setStyleSheet(
                "font-size: 16px; font-weight: bold; color: blue;"
            )
            v.addWidget(lbl_visualisation)
            # Label Global (noir)
            lbl_global = QLabel("Global : -")
            lbl_global.setAlignment(Qt.AlignCenter)
            lbl_global.setStyleSheet(
                "font-size: 16px; font-weight: bold; color: black;"
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
        self.progress_title = QLabel("Progress")
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

        self.count_label = QLabel("Counting")
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
    def _setup_spinbox_scientific_notation(self, spinbox):
        """Setup a spinbox to display values in scientific notation."""
        spinbox.setPrefix("")
        spinbox.setSuffix("")
        # Create a custom validator/formatter by overriding text display
        # This is done via the spinbox's internal text update mechanism
        original_value_from_text = spinbox.valueFromText
        
        def text_from_value(val):
            # Format as scientific notation with 3 decimals
            if val == 0:
                return "0.000e+00"
            return f"{val:.3e}"
        
        # Monkey-patch the spinbox to use scientific notation for display
        spinbox.textFromValue = text_from_value
        spinbox.setValue(spinbox.value())  # Trigger update
    
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
        # update internal value (GUI spinbox changed)
        # Internally, self.time_window is always stored in ticks
        current_unit = self.time_window_unit.currentText() if hasattr(self.time_window_unit, "currentText") else getattr(self, "time_window_unit_value", "ticks")
        self.time_window = value if current_unit == "ticks" else value * self.ratio_ticks_per_second

    def on_time_window_unit_changed(self, unit):
        """Handle switching the time-window unit (ticks <-> seconds)."""
        old_unit = getattr(self, "time_window_unit_value", "ticks")
        if unit == old_unit:
            return

        current_value = self.time_window_value.value()

        # Convert the displayed value when switching units and update allowed ranges
        if old_unit == "ticks" and unit.startswith("seconds"):
            # ticks -> seconds: divide by ratio
            new_value = current_value / self.ratio_ticks_per_second
            # allowed range for seconds: 4 ticks / ratio to 5e12 ticks / ratio
            self.time_window_value.setRange(4 / self.ratio_ticks_per_second, 5e12 / self.ratio_ticks_per_second)
            self.time_window_value.setDecimals(15)
        elif old_unit.startswith("seconds") and unit == "ticks":
            # seconds -> ticks: multiply by ratio
            new_value = current_value * self.ratio_ticks_per_second
            self.time_window_value.setRange(4, 5e12)
            self.time_window_value.setDecimals(1)
        else:
            new_value = current_value

        # Clamp and set
        min_val = self.time_window_value.minimum()
        max_val = self.time_window_value.maximum()
        new_value = min(max(new_value, min_val), max_val)

        self.time_window_value.blockSignals(True)
        self.time_window_value.setValue(new_value)
        self.time_window_value.blockSignals(False)

        # Store unit and update internal ticks value and display
        self.time_window_unit_value = unit
        self.time_window = self.time_window_value.value() if unit == "ticks" else self.time_window_value.value() * self.ratio_ticks_per_second
        self._update_time_range_display(unit)

    def on_time_range_changed(self, values):
        """Update t_min and t_max from range slider (tuple of two values)."""
        min_val, max_val = values
        self.t_min = (min_val / 1000.0) * self.data_t_max
        self.t_max = (max_val / 1000.0) * self.data_t_max
        # Update display with current unit
        current_unit = self.time_window_unit.currentText() if hasattr(self.time_window_unit, "currentText") else "ticks"
        self._update_time_range_display(current_unit)

    def _update_time_range_display(self, unit):
        """Update the time range label display based on current unit."""
        if unit.startswith("seconds"):
            t_min_display = self.t_min / self.ratio_ticks_per_second
            t_max_display = self.t_max / self.ratio_ticks_per_second
            duration_display = (self.t_max - self.t_min) / self.ratio_ticks_per_second
        else:
            t_min_display = self.t_min
            t_max_display = self.t_max
            duration_display = self.t_max - self.t_min
        self.lbl_tmin_max_value.setText(
            f"t_min = {t_min_display:.3e} {unit}   |   "
            f"Duration = {duration_display:.3e} {unit}   |   "
            f"t_max = {t_max_display:.3e} {unit}"
        )
    
    def on_color_code_changed(self, state):
        """Enable or disable particle color coding for visualization."""
        self.use_color_code = bool(state)  # state is 0 (unchecked) or 2 (checked)
        # No debug print; state stored in self.use_color_code

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
                        try:
                            self.save_preview_btn.setEnabled(True)
                        except Exception:
                            pass
                        return
        except Exception:
            # Any failure here is non-fatal for the GUI
            pass
        self.lbl_preview.clear()
        self.lbl_preview.setText("No preview")
        try:
            self.save_preview_btn.setEnabled(False)
        except Exception:
            pass

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
        """Run counting on a single window [t_min, t_max] and display the composite image."""
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
        self.status.setText("Visualization running...")
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
            # Pas besoin d'utiliser le fichier, on utilise les slices pré-calculées
            self.worker = CountWorker(self.selected_file, discr_crit=discr_crit, is_global_count=False,
                                      return_images=True,  # récupérer les images filtrées
                                      time_min=self.t_min, time_max=self.t_max,  # Pass time bounds for filtering
                                      pre_sliced_images=self.global_slices,  # Utiliser les slices stockées
                                      pre_slice_times=self.global_slice_times)  # Utiliser les temps des slices
        else:
            # Fallback: calculer une seule fenêtre depuis le fichier
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
        """Request cancellation of the running counting job."""
        if self.worker is None or not self.worker.isRunning():
            return
        self.status.setText("Cancelling...")
        self.btn_cancel.setEnabled(False)
        try:
            self.worker.requestInterruption()
            # Optionally indicate in the progress bar
            self.slice_progress.setFormat("Cancellation requested...")
            self.count_progress.setFormat("Cancellation requested...")
            self.cancelled = True
        except Exception:
            pass

    def _on_finished(self, results: dict):
        """Update the result labels with the counts returned by the worker."""
        counts = results.get('Counts', {}) if isinstance(results, dict) else {}
        which_label = int(self.is_global_count)  # 0 for Visualization, 1 for Global
        prefix = "Visualization" if which_label == 0 else "Global"
        for category_name, label_pair in self.result_labels.items():
            count_value = counts.get(category_name, '-')
            label_pair[which_label].setText(f"{prefix} : {count_value}")
        
        # If in visualization mode, display the composite image in the preview
        if not self.is_global_count:
            images = results.get('Images', None)
            if images:
                composite, is_rgb = self._create_composite_image(images)
                if composite is not None:
                    if is_rgb:
                        qimg = numpy_to_qimage_rgb(composite)
                    else:
                        qimg = numpy_to_qimage(composite)
                    pix = QPixmap.fromImage(qimg).scaled(self.lbl_preview.size(), Qt.KeepAspectRatio)
                    self.lbl_preview.setPixmap(pix)
                # store last visualization images so checkboxes can update preview
                self._last_visualization_images = images
                try:
                    self.save_preview_btn.setEnabled(True)
                except Exception:
                    pass
        
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
        # If cancelled, leave bars at their current value
        if self.cancelled:
            self.status.setText("Cancelled")
            self.slice_progress.setFormat("Cancelled")
            self.count_progress.setFormat("Cancelled")
        else:
            self.status.setText("Done")
            self.slice_progress.setValue(100)
            self.slice_progress.setFormat("Finished")
            self.count_progress.setValue(100)
            self.count_progress.setFormat("Finished")
        self.btn_run.setEnabled(True)
        self.btn_select.setEnabled(True)
        self.btn_visualisation.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.worker = None
        self.progress_phase = "slicing"
        self.last_pct = 0
        self.cancelled = False

    def _create_composite_image(self, images: dict):
        """Create an RGB composite image from the filtered per-particle images.

        Args:
            images: Dict containing 'alpha', 'electrons', 'muons', 'gamma' (256x256 images)

        Returns:
            Tuple (image_array, is_rgb): NumPy image array and a boolean indicating if it's RGB
        """
        try:
            # Retrieve the images
            img_alpha = images.get('alpha', np.zeros((256, 256)))
            img_muons = images.get('muons', np.zeros((256, 256)))
            img_electrons = images.get('electrons', np.zeros((256, 256)))
            img_gamma = images.get('gamma', np.zeros((256, 256)))
            img_original = images.get('original', np.zeros((256, 256)))
            
            # Respect legend checkboxes (if present) — hide categories that are unchecked
            try:
                if not self.chk_show_alpha.isChecked():
                    img_alpha = np.zeros_like(img_alpha)
                if not self.chk_show_muons.isChecked():
                    img_muons = np.zeros_like(img_muons)
                if not self.chk_show_electrons.isChecked():
                    img_electrons = np.zeros_like(img_electrons)
                if not self.chk_show_gamma.isChecked():
                    img_gamma = np.zeros_like(img_gamma)
            except Exception:
                pass

            if self.use_color_code:
                # Color code: alpha=yellow (R+G), muons=red, electrons=blue, gamma=green
                composite = np.zeros((256, 256, 3), dtype=np.uint8)
                composite[:, :, 0] = np.clip(img_muons * 255, 0, 255).astype(np.uint8)  # red for muons
                composite[:, :, 1] = np.clip(img_gamma * 255, 0, 255).astype(np.uint8)   # green for gamma
                composite[:, :, 2] = np.clip(img_electrons * 255, 0, 255).astype(np.uint8) # blue for electrons
                # Add alpha as yellow (R + G)
                composite[:, :, 0] = np.clip(composite[:, :, 0].astype(np.int16) + (img_alpha * 255).astype(np.int16), 0, 255).astype(np.uint8)
                composite[:, :, 1] = np.clip(composite[:, :, 1].astype(np.int16) + (img_alpha * 255).astype(np.int16), 0, 255).astype(np.uint8)
                return composite, True
            else:
                # Grayscale: combine selected per-particle masks (respecting checkboxes)
                combined = np.zeros_like(img_alpha, dtype=np.float32)
                combined = np.clip(combined + img_alpha, 0, 1)
                combined = np.clip(combined + img_muons, 0, 1)
                combined = np.clip(combined + img_electrons, 0, 1)
                combined = np.clip(combined + img_gamma, 0, 1)
                # If nothing selected, fallback to original TOT image
                if not np.any(combined):
                    combined = img_original
                return np.clip(combined * 255, 0, 255).astype(np.uint8), False
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

    def _save_preview_image(self):
        """Open a save dialog and save the currently displayed preview as PNG."""
        try:
            pix = self.lbl_preview.pixmap()
            if pix is None:
                self.status.setText("No image to save")
                return
            start_dir = os.getcwd()
            path, _ = QFileDialog.getSaveFileName(self, "Save preview image", start_dir, "PNG Files (*.png)")
            if not path:
                return
            if not path.lower().endswith('.png'):
                path = path + '.png'
            success = pix.save(path, 'PNG')
            if success:
                self.status.setText(f"Saved image: {Path(path).name}")
            else:
                self.status.setText("Failed to save image")
        except Exception as e:
            print(f"Error saving preview: {e}")
            traceback.print_exc()
            self.status.setText("Error saving image")

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

    def _on_legend_checkbox_changed(self, _state):
        """Update preview when a legend checkbox changes (visualization mode)."""
        try:
            if not self.is_global_count and getattr(self, '_last_visualization_images', None) is not None:
                images = self._last_visualization_images
                composite, is_rgb = self._create_composite_image(images)
                if composite is not None:
                    if is_rgb:
                        qimg = numpy_to_qimage_rgb(composite)
                    else:
                        qimg = numpy_to_qimage(composite)
                    pix = QPixmap.fromImage(qimg).scaled(self.lbl_preview.size(), Qt.KeepAspectRatio)
                    self.lbl_preview.setPixmap(pix)
                    try:
                        self.save_preview_btn.setEnabled(True)
                    except Exception:
                        pass
        except Exception:
            # don't let preview updates break the UI
            pass

def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
