import numpy as np
from pathlib import Path
import tkinter as tk
import sys
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

try:
    # allow running as part of a package (preferred)
    from .read_file import read, slice, slice_Tot
    from .filtres import filtre_alpha, filtre_tracks
    from .plot_results import plot_results
    from .event_detector_v3 import event_counting_alpha, event_counting_electron_muon,event_counting_photon
    # diagnostic: indicate which import branch succeeded
except Exception:
    # fallback when the module is executed directly as a script
    # attempt to import from the same directory
    from read_file import read, slice, slice_Tot
    from filtres import filtre_alpha, filtre_tracks
    from plot_results import plot_results
    from event_detector_v3 import event_counting_alpha, event_counting_electron_muon,event_counting_photon
    # diagnostic: indicate fallback import used

