import time  # pour mesurer les durées d'exécution
from pathlib import Path  # gestion de chemins OS-indépendants
import sys  # manipulation du sys.path pour importer depuis le repo
import numpy as np  # calcul numérique
import matplotlib.pyplot as plt  # tracés
from scipy.ndimage import label, sum as ndi_sum
from matplotlib.widgets import Button
from matplotlib.colors import ListedColormap
from datetime import datetime, timezone
import csv
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
        d_time = total_time / 100.0

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
        btn_reset = Button(ax_btn_reset, 'Reset (r)')
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

        # Callback undo: enlève la dernière entrée du CSV et restaure les compteurs et l'index
        def undo_cb(event):
            if not comptage_path.exists():
                print("Aucun fichier de comptage pour annuler.")
                return
            try:
                with open(comptage_path, 'r', encoding='utf-8') as f:
                    reader = list(csv.DictReader(f))
                if not reader:
                    print("Fichier de comptage vide, rien à annuler.")
                    return
                last = reader[-1]
                # compute previous snapshot (the row before the last), if any
                prev = reader[-2] if len(reader) > 1 else None
                # remove last entry
                remaining = reader[:-1]
                # réécrit le CSV en gardant l'ordre canonique des colonnes
                with open(comptage_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
                    writer.writeheader()
                    for row in remaining:
                        clean = {k: row.get(k, '') for k in CSV_FIELDNAMES}
                        writer.writerow(clean)

                # restore counters from prev or zeros
                if prev:
                    for k in ('alpha', 'muon', 'electron', 'gamma', 'other'):
                        try:
                            counters[k] = int(prev.get(k, 0))
                        except Exception:
                            counters[k] = 0
                else:
                    for k in ('alpha', 'muon', 'electron', 'gamma', 'other'):
                        counters[k] = 0

                # After removing the last entry, display the cluster that was
                # removed (last). Example: if currently at cluster 3, the last
                # saved row is for cluster 2; undo should remove row for 2 and
                # re-display cluster 2.
                try:
                    last_cluster = int(last.get('cluster', 1))
                except Exception:
                    last_cluster = 1
                cluster_index[0] = last_cluster

                # Show the cluster that was undone
                # last row contained cluster number before increment; after undo we want to show that cluster
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

        # Sauvegarde du comptage dans un fichier CSV situé au même endroit que le fichier source
        comptage_path = Path(file).with_name(f"{Path(file).stem}_comptage.csv")
        # définit un jeu de colonnes canonique (utilisé partout pour écriture/lecture)
        CSV_FIELDNAMES = ['timestamp', 'source_file', 'cluster', 'time', 'category', 'alpha', 'muon', 'electron', 'gamma', 'other']

        # Si demandé par l'utilisateur, supprimer le fichier de comptage existant pour recommencer
        if args.reset and comptage_path.exists():
            try:
                os.remove(comptage_path)
                print(f"Fichier de comptage supprimé (reset): {comptage_path}")
            except Exception as e:
                print(f"Impossible de supprimer {comptage_path}: {e}")

        def save_count(cluster_num, category, counters_snapshot):
            """Ajoute une ligne au fichier de comptage en utilisant DictWriter

            Colonnes: timestamp, source_file, cluster, time, category, alpha, muon, electron, gamma, other
            """
            time_of_cluster = clusters_list[int(cluster_num) - 1]['t'] if 1 <= int(cluster_num) <= len(clusters_list) else ''
            row = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'source_file': Path(file).name,
                'cluster': int(cluster_num),
                'time': f"{time_of_cluster:.6f}",
                'category': category,
                'alpha': counters_snapshot.get('alpha', 0),
                'muon': counters_snapshot.get('muon', 0),
                'electron': counters_snapshot.get('electron', 0),
                'gamma': counters_snapshot.get('gamma', 0),
                'other': counters_snapshot.get('other', 0),
            }
            try:
                comptage_path.parent.mkdir(parents=True, exist_ok=True)
                write_header = not comptage_path.exists()
                with open(comptage_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
                    if write_header:
                        writer.writeheader()
                    writer.writerow(row)
            except Exception as e:
                print(f"Erreur sauvegarde comptage: {e}")

        # Si le fichier de comptage existe déjà et que l'option resume est activée, on peut reprendre
        if args.resume and comptage_path.exists():
            try:
                with open(comptage_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                if rows:
                    # dernière entrée traitée
                    last = rows[-1]
                    try:
                        last_cluster = int(last.get('cluster', 0))
                    except Exception:
                        last_cluster = 0
                    # Restaurer les compteurs à partir de la dernière ligne
                    for k in ('alpha', 'muon', 'electron', 'gamma', 'other'):
                        try:
                            counters[k] = int(last.get(k, 0))
                        except Exception:
                            counters[k] = 0
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

        # Fabrica de callback pour incrémenter et avancer
        def make_callback(category):
            def _cb(event):
                # ignore presses if already past the end
                if cluster_index[0] > total_clusters:
                    # already finished for this file; keep window open until Stop
                    title.set_text(f"Fichier: {Path(file).name} — FINI ({total_clusters} clusters triés). Appuyez sur Stop ou Choisir fichier.")
                    return
                current_cluster = int(cluster_index[0])
                counters[category] += 1
                save_count(current_cluster, category, counters)
                cluster_index[0] += 1
                # si on a dépassé la fin, n'appeler pas finish() — on garde la fenêtre ouverte
                if cluster_index[0] > total_clusters:
                    # afficher message de fin mais ne pas fermer
                    img_display.set_data(np.zeros((10, 10)))
                    title.set_text(f"Fichier: {Path(file).name} — FINI ({total_clusters} clusters triés). Appuyez sur Stop ou Choisir fichier.")
                    ax_im.set_title("Aucun cluster restant")
                    fig.canvas.draw_idle()
                    return
                else:
                    show_cluster(cluster_index[0])
            return _cb

        btn_alpha.on_clicked(make_callback('alpha'))
        btn_muon.on_clicked(make_callback('muon'))
        btn_electron.on_clicked(make_callback('electron'))
        btn_gamma.on_clicked(make_callback('gamma'))
        btn_other.on_clicked(make_callback('other'))
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
                make_callback('alpha')(event)
            elif key == 'm':
                make_callback('muon')(event)
            elif key == 'e':
                make_callback('electron')(event)
            elif key == 'g':
                make_callback('gamma')(event)
            elif key == 'o':
                make_callback('other')(event)
            elif key == 'u':
                undo_cb(event)
            elif key == 'r':
                reset_cb(event)
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


    



