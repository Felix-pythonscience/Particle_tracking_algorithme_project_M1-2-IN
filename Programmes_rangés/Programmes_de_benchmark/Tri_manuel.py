import time  # pour mesurer les durées d'exécution
from pathlib import Path  # gestion de chemins OS-indépendants
import sys  # manipulation du sys.path pour importer depuis le repo
import numpy as np  # calcul numérique
import matplotlib.pyplot as plt  # tracés
from scipy.ndimage import label, sum as ndi_sum
from matplotlib.widgets import Button
from datetime import datetime
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

        # Récupère la matrice binaire/étiquetée depuis la fonction de comptage
        Data = compteur_particles(file=file, plot=False, images_debug=True)[0]
        labeled_matrix, num_clusters = label(Data, structure=structure)

        # Si aucun cluster détecté, on passe au fichier suivant
        if num_clusters == 0:
            print(f"Fichier {file}: aucun cluster détecté, skipped.")
            continue

        # Compteurs et état (ajout distinction muon/electron)
        counters = {"alpha": 0, "muon": 0, "electron": 0, "gamma": 0, "other": 0}
        cluster_index = [1]

        # Création de la figure/axes pour afficher les clusters et les boutons
        fig, ax_im = plt.subplots(figsize=(6, 6))
        plt.subplots_adjust(bottom=0.25)
        img_display = ax_im.imshow(np.zeros((10, 10)), cmap='gray')
        ax_im.axis('off')
        title = fig.suptitle(f"Fichier: {Path(file).name} — Cluster {cluster_index[0]}/{num_clusters}")

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
            mask = (labeled_matrix == i)
            total = int(mask.sum())
            if total == 0:
                display = np.zeros((10, 10), dtype=int)
            else:
                rows, cols = np.where(mask)
                pad = 5
                r0 = max(0, rows.min() - pad)
                r1 = min(mask.shape[0], rows.max() + pad + 1)
                c0 = max(0, cols.min() - pad)
                c1 = min(mask.shape[1], cols.max() + pad + 1)
                display = mask[r0:r1, c0:c1].astype(int)

            img_display.set_data(display)
            img_display.set_cmap('gray')
            img_display.set_clim(0, 1)
            ax_im.set_title(f"Cluster {i}/{num_clusters} — pixels: {total}")
            title.set_text(f"Fichier: {Path(file).name} — Cluster {i}/{num_clusters}")
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
                # compute previous snapshot
                prev = reader[-2] if len(reader) > 1 else None
                # remove last entry
                remaining = reader[:-1]
                fieldnames = ['timestamp', 'source_file', 'cluster', 'category', 'alpha', 'muon', 'electron', 'gamma', 'other']
                with open(comptage_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for row in remaining:
                        writer.writerow(row)

                # restore counters from prev or zeros
                if prev:
                    for k in ('alpha', 'muon', 'electron', 'gamma', 'other'):
                        try:
                            counters[k] = int(prev.get(k, 0))
                        except Exception:
                            counters[k] = 0
                    cluster_index[0] = int(prev.get('cluster', 0)) + 1 if prev.get('cluster') is not None else 1
                else:
                    for k in ('alpha', 'muon', 'electron', 'gamma', 'other'):
                        counters[k] = 0
                    cluster_index[0] = 1

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

        # Si demandé par l'utilisateur, supprimer le fichier de comptage existant pour recommencer
        if args.reset and comptage_path.exists():
            try:
                os.remove(comptage_path)
                print(f"Fichier de comptage supprimé (reset): {comptage_path}")
            except Exception as e:
                print(f"Impossible de supprimer {comptage_path}: {e}")

        def save_count(cluster_num, category, counters_snapshot):
            """Ajoute une ligne au fichier de comptage.

            Colonnes: timestamp, source_file, cluster, category, alpha, muon, electron, gamma, other
            """
            header = ['timestamp', 'source_file', 'cluster', 'category', 'alpha', 'muon', 'electron', 'gamma', 'other']
            row = [datetime.utcnow().isoformat(), Path(file).name, int(cluster_num), category,
                   counters_snapshot.get('alpha', 0), counters_snapshot.get('muon', 0),
                   counters_snapshot.get('electron', 0), counters_snapshot.get('gamma', 0),
                   counters_snapshot.get('other', 0)]
            try:
                comptage_path.parent.mkdir(parents=True, exist_ok=True)
                write_header = not comptage_path.exists()
                with open(comptage_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    if write_header:
                        writer.writerow(header)
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
                    # Reprendre au cluster suivant
                    cluster_index[0] = last_cluster + 1
                    if cluster_index[0] > num_clusters:
                        print(f"Fichier {file} : tous les clusters ({num_clusters}) ont déjà été triés selon {comptage_path.name}, skipped.")
                        plt.close('all')
                        continue
                    else:
                        print(f"Reprise du tri pour {Path(file).name} à partir du cluster {cluster_index[0]} (dernière entrée: cluster {last_cluster})")
            except Exception as e:
                print(f"Erreur lors de la lecture du fichier de reprise {comptage_path}: {e}")

        # Fabrica de callback pour incrémenter et avancer
        def make_callback(category):
            def _cb(event):
                current_cluster = int(cluster_index[0])
                counters[category] += 1
                save_count(current_cluster, category, counters)
                cluster_index[0] += 1
                if cluster_index[0] > num_clusters:
                    finish()
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


    



