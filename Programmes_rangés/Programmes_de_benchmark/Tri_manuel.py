from pathlib import Path  # gestion de chemins OS-indépendants
import sys  # manipulation du sys.path pour importer depuis le repo
import numpy as np  # calcul numérique
import matplotlib.pyplot as plt  # tracés
from scipy.ndimage import label, sum as ndi_sum
from matplotlib.widgets import Button
from matplotlib.colors import ListedColormap
from datetime import datetime, timezone
import argparse
import os
import tkinter as tk
from tkinter import filedialog
# Ajoute le dossier parent du fichier courant au chemin Python afin de pouvoir
# importer les modules locaux (Programmes_de_bases) même si le script est lancé
# directement depuis ce dossier.
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

# Imports des fonctions du package local
from Programmes_de_bases.compteur import compteur_particles
from Programmes_de_bases.read_file import read


if __name__ == "__main__":
    # Arguments CLI
    parser = argparse.ArgumentParser(description='Tri manuel des clusters (GUI).')
    parser.add_argument('--folder', '-f', dest='folder', default=None,
                        help='Dossier contenant les fichiers .t3pa à analyser.')
    parser.add_argument('--file', '-F', dest='file', default=None,
                        help='Chemin vers un fichier .t3pa unique à traiter.')
    parser.add_argument('--pick', dest='pick', action='store_true', help='Ouvrir un explorateur pour sélectionner un fichier .t3pa.')
    parser.add_argument('--resume', dest='resume', action='store_true', help='Reprendre le tri depuis le fichier de comptage si présent.')
    parser.add_argument('--no-resume', dest='resume', action='store_false', help="Ne PAS reprendre, commencer depuis le début (ignore le CSV existant).")
    parser.set_defaults(resume=True)
    parser.add_argument('--reset', dest='reset', action='store_true', help='Si présent, supprime les fichiers de comptage existants avant de commencer (pour recommencer à zéro).')
    args = parser.parse_args()

    # Choix des fichiers à traiter : priorité --file, --pick, sinon --folder ou dossier par défaut
    if args.file:
        files = [args.file]
    elif args.pick:
        # ouvre un dialog pour sélectionner un fichier
        root = tk.Tk()
        root.withdraw()
        selected = filedialog.askopenfilename(title='Select .t3pa file', filetypes=[('t3pa files', '*.t3pa'), ('All files','*.*')])
        root.destroy()
        if not selected:
            print('Aucun fichier sélectionné, arrêt.')
            sys.exit(0)
        files = [selected]
    else:
        # Si aucun fichier explicitement fourni, on ouvre automatiquement le sélecteur au lancement
        root = tk.Tk()
        root.withdraw()
        selected = filedialog.askopenfilename(title='Select .t3pa file', filetypes=[('t3pa files', '*.t3pa'), ('All files','*.*')])
        root.destroy()
        if not selected:
            print('Aucun fichier sélectionné, arrêt.')
            sys.exit(0)
        files = [selected]

    # Utilise une file d'attente modifiable afin de pouvoir insérer des fichiers choisis pendant l'exécution
    pending_files = list(files)
    next_file = [None]

    # Initialisation des listes pour stocker les résultats


    # Flag global pour indiquer l'arrêt demandé via le bouton Stop
    stop_all = [False]

    # Boucle principale: on prend le premier fichier dans pending_files
    while pending_files:
        file = pending_files.pop(0)
        # initialisation par fichier
        alpha_counts = 0
        track_counts = 0
        gamma_counts = 0
        other_counts = 0
        processing_times = 0
        structure = np.ones((3, 3), dtype=int)

        # Lit les données et construit la liste de clusters pour TOUTES les fenêtres temporelles
        data_df = read(file)
        total_time = float(data_df.iloc[:, 1].max())
        # Durée de fenêtre par défaut (même règle que dans compteur_particles)
        d_time = total_time / 500.0

        clusters_list = []  # chaque élément: dict {mask, t, bbox, pixels}
        t = 0.0
        while t < total_time:
            # récupère les images pour la fenêtre temporelle [t, t+d_time)
            image = compteur_particles(file=data_df, t=t, d_time=d_time, plot=False, images_debug=True)
            labeled_matrix, num_clusters = label(image, structure=structure)
            if num_clusters > 0:
                for i in range(1, num_clusters + 1):
                    # ensure mask is an independent copy so later modifications
                    # or reuse of labeled_matrix won't change stored masks
                    mask = (labeled_matrix == i).copy()
                    total = int(mask.sum())
                    if total == 0:
                        continue
                    rows, cols = np.where(mask)
                    pad = 5
                    r0 = max(0, rows.min() - pad)
                    r1 = min(mask.shape[0], rows.max() + pad + 1)
                    c0 = max(0, cols.min() - pad)
                    c1 = min(mask.shape[1], cols.max() + pad + 1)
                    clusters_list.append({
                        'mask': mask,
                        't': float(t),
                        'bbox': (r0, r1, c0, c1),
                        'pixels': total
                    })
            t += d_time

        if not clusters_list:
            print(f"Fichier {file}: aucun cluster détecté sur toutes les fenêtres temporelles, skipped.")
            continue

        # Compteurs et état (ajout distinction muon/electron)
        counters = {"alpha": 0, "muon": 0, "electron": 0, "gamma": 0, "other": 0}
        # counts collected for the current cluster before committing with space
        pending_counts = {k: 0 for k in counters}
        cluster_index = [1]  # index 1-based dans clusters_list
        total_clusters = len(clusters_list)

        # Création de la figure/axes pour afficher les clusters et les boutons
        fig, ax_im = plt.subplots(figsize=(6, 6))
        plt.subplots_adjust(bottom=0.25)
        # Colormap: background white, lit pixels dark gray
        cmap_binary = ListedColormap(['#ffffff', '#444444'])
        img_display = ax_im.imshow(np.zeros((10, 10)), cmap=cmap_binary, vmin=0, vmax=1, interpolation='nearest', origin='upper')
        ax_im.set_facecolor('#ffffff')
        # We'll use minor ticks to draw a light grid over pixels; hide major ticks
        ax_im.set_xticks([])
        ax_im.set_yticks([])
        title = fig.suptitle(f"Fichier: {Path(file).name} — Cluster {cluster_index[0]}/{total_clusters}")

        # Axes des boutons (positions en pourcentage de la figure)
        ax_btn_alpha = plt.axes([0.05, 0.03, 0.18, 0.07])
        ax_btn_muon = plt.axes([0.25, 0.03, 0.18, 0.07])
        ax_btn_electron = plt.axes([0.45, 0.03, 0.18, 0.07])
        ax_btn_gamma = plt.axes([0.65, 0.03, 0.18, 0.07])
        ax_btn_other = plt.axes([0.05, 0.12, 0.16, 0.07])
        ax_btn_choose = plt.axes([0.23, 0.12, 0.16, 0.07])
        ax_btn_undo = plt.axes([0.41, 0.12, 0.16, 0.07])
        ax_btn_reset = plt.axes([0.59, 0.12, 0.16, 0.07])
        ax_btn_stop = plt.axes([0.77, 0.12, 0.18, 0.07])

        btn_alpha = Button(ax_btn_alpha, 'Alpha (a)')
        btn_muon = Button(ax_btn_muon, 'Muon (m)')
        btn_electron = Button(ax_btn_electron, 'Electron (e)')
        btn_gamma = Button(ax_btn_gamma, 'Gamma (g)')
        btn_other = Button(ax_btn_other, 'Other (o)')
        btn_choose = Button(ax_btn_choose, 'Choose file')
        btn_undo = Button(ax_btn_undo, 'Undo (u)')
        btn_reset = Button(ax_btn_reset, 'Reset')
        btn_stop = Button(ax_btn_stop, 'Stop (s)')

        # Affiche le cluster d'index i (1-based)
        def show_cluster(i):
            # i is 1-based index into clusters_list
            if i < 1 or i > total_clusters:
                # Aucun cluster à afficher: on garde la fenêtre ouverte et affiche un message
                img_display.set_data(np.zeros((10, 10)))
                ax_im.set_title("No cluster to display")
                title.set_text(f"Fichier: {Path(file).name} — Aucun cluster ({total_clusters}) — Appuyez sur Stop ou Choisir fichier")
                fig.canvas.draw_idle()
                return
            info = clusters_list[i - 1]
            mask = info['mask']
            r0, r1, c0, c1 = info['bbox']
            display = mask[r0:r1, c0:c1].astype(int)
            # update grid for the displayed patch: put a tick at every pixel edge
            # and make sure axis limits align exactly with pixel borders so the
            # grid lines coincide with pixel edges (important when zooming or
            # resizing the figure).
            h, w = display.shape
            # minor ticks at pixel boundaries (positions at -0.5, 0.5, 1.5, ...)
            ax_im.set_xticks(np.arange(-0.5, w, 1), minor=True)
            ax_im.set_yticks(np.arange(-0.5, h, 1), minor=True)
            # align axis limits with pixel borders (imshow uses pixels centered at
            # integer coords; borders are at +/-0.5). origin='upper' so invert y limits.
            ax_im.set_xlim(-0.5, w - 0.5)
            ax_im.set_ylim(h - 0.5, -0.5)
            ax_im.set_aspect('equal')
            ax_im.grid(which='minor', color='#dddddd', linestyle='-', linewidth=0.5)
            # hide major ticks/labels but keep minor grid visible
            ax_im.set_xticks([])
            ax_im.set_yticks([])
            ax_im.tick_params(which='major', bottom=False, left=False, labelbottom=False, labelleft=False)
            pixels = info['pixels']
            # Update the image shown and explicitly set its extent so pixel
            # boundaries align exactly with the grid lines. This prevents half-
            # pixel shifts when changing the image size and guarantees that one
            # grid square corresponds to one image pixel.
            img_display.set_data(display)
            # extent: left, right, bottom, top in data coords; using +-0.5
            # positions aligns pixel centers at integer coordinates and borders
            # at +-0.5
            img_display.set_extent((-0.5, w - 0.5, h - 0.5, -0.5))
            img_display.set_cmap(cmap_binary)
            img_display.set_clim(0, 1)
            img_display.set_interpolation('nearest')

            # recompute counts on the mask and on the displayed patch to detect
            # any mismatch (helps debugging intermittent count errors)
            pixels = int(mask.sum())
            displayed_pixels = int(np.count_nonzero(display))

            ax_im.set_title(f"Cluster {i}/{total_clusters} — pixels total: {pixels} (in-view: {displayed_pixels}) — t={info['t']:.3f}s")
            title.set_text(f"Fichier: {Path(file).name} — Cluster {i}/{total_clusters}")
            fig.canvas.draw_idle()

        # Fin de tri pour ce fichier: afficher récap et fermer la fenêtre
        def finish():
            print("\n=== Résumé tri manuel ===")
            print(f"Fichier: {file}")
            print(f"Alpha: {counters['alpha']}")
            print(f"Muon: {counters['muon']}")
            print(f"Electron: {counters['electron']}")
            print(f"Gamma: {counters['gamma']}")
            print(f"Other: {counters['other']}")
            plt.close(fig)

        # Callback stop: arrête le traitement global et ferme la fenêtre
        def stop_cb(event):
            stop_all[0] = True
            plt.close(fig)

        # Callback pour ouvrir un explorateur et insérer le fichier choisi en tête de la file
        def choose_cb(event):
            root = tk.Tk()
            root.withdraw()
            selected = filedialog.askopenfilename(title='Select .t3pa file', filetypes=[('t3pa files', '*.t3pa'), ('All files','*.*')])
            root.destroy()
            if selected:
                # insère en tête pour être traité immédiatement après la fermeture de la fenêtre
                pending_files.insert(0, selected)
                print(f"Fichier choisi: {selected}")
                plt.close(fig)

        # Callback undo: enlève la dernière entrée du fichier .npy et restaure les compteurs et l'index
        def undo_cb(event):
            if not comptage_path.exists():
                print("Aucun fichier de comptage pour annuler.")
                return
            try:
                arr = list(np.load(comptage_path, allow_pickle=True))
                # arr[0]=explanation, arr[1]=counters, arr[2:]=cluster entries
                if len(arr) <= 2:
                    print("Aucun enregistrement de cluster à annuler.")
                    return
                last = arr.pop()  # removed cluster entry (dict)

                # restore counters from arr[1] (snapshot after previous saved row) or zeros
                if len(arr) > 1 and isinstance(arr[1], dict):
                    prev_counters = arr[1]
                    for k in ('alpha', 'muon', 'electron', 'gamma', 'other'):
                        try:
                            counters[k] = int(prev_counters.get(k, 0))
                        except Exception:
                            counters[k] = 0
                else:
                    for k in ('alpha', 'muon', 'electron', 'gamma', 'other'):
                        counters[k] = 0

                # write back the trimmed array
                np.save(comptage_path, np.array(arr, dtype=object), allow_pickle=True)

                # After removing the last entry, display the cluster that was removed
                try:
                    last_cluster = int(last.get('cluster', 1))
                except Exception:
                    last_cluster = 1
                cluster_index[0] = last_cluster

                try:
                    show_cluster(cluster_index[0])
                except Exception:
                    pass
                print(f"Annulé: supprimé l'entrée du cluster {last.get('cluster')}, catégorie {last.get('category')}")
            except Exception as e:
                print(f"Erreur lors de l'annulation: {e}")

        # Callback reset: supprime le fichier CSV et remet compteurs et index à zéro/1
        def reset_cb(event):
            try:
                if comptage_path.exists():
                    os.remove(comptage_path)
                    print(f"Fichier de comptage supprimé: {comptage_path}")
            except Exception as e:
                print(f"Impossible de supprimer {comptage_path}: {e}")
            for k in ('alpha', 'muon', 'electron', 'gamma', 'other'):
                counters[k] = 0
            cluster_index[0] = 1
            try:
                show_cluster(cluster_index[0])
            except Exception:
                pass
            print("Réinitialisation des compteurs pour ce fichier.")

        # Sauvegarde du comptage dans un fichier binaire numpy (.npy) situé au même endroit que le fichier source
        comptage_path = Path(file).with_name(f"{Path(file).stem}_comptage.npy")
        EXPLANATION = (
            "FORMAT: [0]=explanation(str), [1]=counters(dict), [2:]=cluster entries(dict with keys 'cluster','category','pixels'(Nx2 np.int_), 'source_file')"
        )

       

        def save_count(cluster_num, pending_counts_snapshot, counters_snapshot):
            """Commit current cluster: store counts per cluster and pixels into .npy

            arr[0]=explanation, arr[1]=counters snapshot, arr[2:]=cluster entries
            each cluster entry: {'cluster':int,'counts':dict,'pixels':np.ndarray(N,2),'source_file':str}
            """
            # pixels as list of (row, col)
            try:
                mask = clusters_list[int(cluster_num) - 1]['mask']
                pixels_coords = np.column_stack(np.where(mask)).astype(np.int_)
            except Exception:
                pixels_coords = np.zeros((0, 2), dtype=np.int_)

            entry = {
                'cluster': int(cluster_num),
                'counts': {k: int(pending_counts_snapshot.get(k, 0)) for k in pending_counts_snapshot},
                'pixels': pixels_coords,
                'source_file': Path(file).name,
            }

            try:
                comptage_path.parent.mkdir(parents=True, exist_ok=True)
                if comptage_path.exists():
                    arr = list(np.load(comptage_path, allow_pickle=True))
                else:
                    # initialize array: explanation + counters snapshot
                    arr = [EXPLANATION, {k: counters_snapshot.get(k, 0) for k in ('alpha', 'muon', 'electron', 'gamma', 'other')}]

                # append entry and update counters snapshot
                arr.append(entry)
                arr[1] = {k: counters_snapshot.get(k, 0) for k in ('alpha', 'muon', 'electron', 'gamma', 'other')}

                np.save(comptage_path, np.array(arr, dtype=object), allow_pickle=True)
            except Exception as e:
                print(f"Erreur sauvegarde comptage (.npy): {e}")

        # Si le fichier de comptage existe déjà et que l'option resume est activée, on peut reprendre
        if args.resume and comptage_path.exists():
            try:
                arr = list(np.load(comptage_path, allow_pickle=True))
                # arr[0]=explanation, arr[1]=counters snapshot, arr[2:]=entries
                if len(arr) > 1 and isinstance(arr[1], dict):
                    last_counters = arr[1]
                    for k in ('alpha', 'muon', 'electron', 'gamma', 'other'):
                        try:
                            counters[k] = int(last_counters.get(k, 0))
                        except Exception:
                            counters[k] = 0
                else:
                    for k in ('alpha', 'muon', 'electron', 'gamma', 'other'):
                        counters[k] = 0

                if len(arr) > 2:
                    try:
                        last_cluster = int(arr[-1].get('cluster', 0))
                    except Exception:
                        last_cluster = 0
                else:
                    last_cluster = 0

                # Reprendre au cluster suivant (global index)
                cluster_index[0] = last_cluster + 1
                if cluster_index[0] > total_clusters:
                    print(f"Fichier {file} : tous les clusters ({total_clusters}) ont déjà été triés selon {comptage_path.name}, skipped.")
                    plt.close('all')
                    continue
                else:
                    print(f"Reprise du tri pour {Path(file).name} à partir du cluster {cluster_index[0]} (dernière entrée: cluster {last_cluster})")
            except Exception as e:
                print(f"Erreur lors de la lecture du fichier de reprise {comptage_path}: {e}")

        # Callbacks: increment pending_counts per category (buttons and keys)
        def incr_category(category):
            # increment pending count for current cluster and update title
            pending_counts[category] += 1
            pending_str = f"pending a={pending_counts['alpha']} e={pending_counts['electron']} m={pending_counts['muon']} g={pending_counts['gamma']} o={pending_counts['other']}"
            title.set_text(f"Fichier: {Path(file).name} — Cluster {cluster_index[0]}/{total_clusters} — {pending_str}")
            fig.canvas.draw_idle()

        def commit_cluster_and_advance():
            # commit current pending_counts for the current cluster and advance
            if cluster_index[0] > total_clusters:
                title.set_text(f"Fichier: {Path(file).name} — FINI ({total_clusters} clusters triés). Appuyez sur Stop ou Choisir fichier.")
                return
            current_cluster = int(cluster_index[0])
            # update global counters by adding pending_counts
            for k in pending_counts:
                try:
                    counters[k] += int(pending_counts.get(k, 0))
                except Exception:
                    pass
            # save
            save_count(current_cluster, pending_counts, counters)
            # reset pending counts
            for k in pending_counts:
                pending_counts[k] = 0
            # advance
            cluster_index[0] += 1
            if cluster_index[0] > total_clusters:
                img_display.set_data(np.zeros((10, 10)))
                title.set_text(f"Fichier: {Path(file).name} — FINI ({total_clusters} clusters triés). Appuyez sur Stop ou Choisir fichier.")
                ax_im.set_title("Aucun cluster restant")
                fig.canvas.draw_idle()
                return
            else:
                show_cluster(cluster_index[0])

        btn_alpha.on_clicked(lambda ev: incr_category('alpha'))
        btn_muon.on_clicked(lambda ev: incr_category('muon'))
        btn_electron.on_clicked(lambda ev: incr_category('electron'))
        btn_gamma.on_clicked(lambda ev: incr_category('gamma'))
        btn_other.on_clicked(lambda ev: incr_category('other'))
        btn_choose.on_clicked(choose_cb)
        btn_undo.on_clicked(undo_cb)
        btn_reset.on_clicked(reset_cb)
        btn_stop.on_clicked(stop_cb)

        # Raccourcis clavier pour les mêmes actions
        def on_key(event):
            key = getattr(event, 'key', None)
            if not key:
                return
            key = key.lower()
            if key == 'a':
                incr_category('alpha')
            elif key == 'm':
                incr_category('muon')
            elif key == 'e':
                incr_category('electron')
            elif key == 'g':
                incr_category('gamma')
            elif key == 'o':
                incr_category('other')
            elif key == ' ' or key == 'space':
                # commit pending counts for this cluster and advance
                commit_cluster_and_advance()
            elif key == 'u':
                undo_cb(event)
            elif key == 's':
                stop_cb(event)

        fig.canvas.mpl_connect('key_press_event', on_key)

        # Affichage initial
        show_cluster(cluster_index[0])
        plt.show()

        # Si l'utilisateur a appuyé sur Stop, on quitte la boucle principale
        if stop_all[0]:
            print("Stop demandé par l'utilisateur — arrêt du traitement des fichiers.")
            break


    



