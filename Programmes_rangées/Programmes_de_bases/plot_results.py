import matplotlib.pyplot as plt
import numpy as np
# Affichage : encapsulé dans une fonction et colorbar limitée à la première ligne
def plot_results(image_originale, image_alpha, image_tracks, image_without_tracks, image_couleur, figsize=(16, 12)):
    """Plot the detection images with a shared colorbar and grayscale second row.

    The top row shows the four images multiplied by `image_couleur` (for TOT-colored
    visualization). The bottom row shows the raw arrays in grayscale. A shared
    vertical colorbar is drawn on the right.

    Parameters
    ----------
    image_originale, image_alpha, image_tracks, image_without_tracks : ndarray
        2D arrays (256x256) with the binary or intensity data for each class.
    image_couleur : ndarray
        2D array (256x256) with TOT or intensity values used to color the top row.
    figsize : tuple, optional
        Figure size passed to matplotlib.
    """
    vmin = float(np.nanmin(image_couleur))
    vmax = float(np.nanmax(image_couleur))
    fig = plt.figure(figsize=figsize)
    # augmenter hspace et wspace pour plus d'espaces entre les images
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 0.06], height_ratios=[1, 1], hspace=0.45, wspace=0.35)

    # Axes pour la première ligne (images colorées)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[0, 3])

    # Axes pour la deuxième ligne (images sans couleur)
    ax1b = fig.add_subplot(gs[1, 0])
    ax2b = fig.add_subplot(gs[1, 1])
    ax3b = fig.add_subplot(gs[1, 2])
    ax4b = fig.add_subplot(gs[1, 3])

    # Axe pour la colorbar : seulement la première ligne (gs[0,4])
    cax = fig.add_subplot(gs[0, 4])

    # image avant filtre (avec couleurs)
    im1 = ax1.imshow(image_originale * image_couleur, cmap='viridis', vmin=vmin, vmax=vmax)
    ax1.set_title("image avant filtre")

    # image après filtre (alpha) - couleur
    im2 = ax2.imshow(image_alpha * image_couleur, cmap='viridis', vmin=vmin, vmax=vmax)
    ax2.set_title("image alpha")

    # image après filtre (tracks) - couleur
    im3 = ax3.imshow(image_tracks * image_couleur, cmap='viridis', vmin=vmin, vmax=vmax)
    ax3.set_title("image tracks")

    # image sans tracks - couleur
    im4 = ax4.imshow(image_without_tracks * image_couleur, cmap='viridis', vmin=vmin, vmax=vmax)
    ax4.set_title("image gammas")

    # Deuxième ligne : mêmes images mais en grayscale (pas de colorbar)
    im1b = ax1b.imshow(image_originale, cmap='gray')
    ax1b.set_title("image avant filtre ")

    im2b = ax2b.imshow(image_alpha, cmap='gray')
    ax2b.set_title("image alphas")

    im3b = ax3b.imshow(image_tracks, cmap='gray')
    ax3b.set_title("image tracks")

    im4b = ax4b.imshow(image_without_tracks, cmap='gray')
    ax4b.set_title("image gammas")

    # Colorbar attachée à l'axe cax (qui couvre uniquement la première ligne)
    cbar = fig.colorbar(im1, cax=cax, orientation='vertical')
    cax.yaxis.tick_right()
    cbar.set_label('TOT')

    plt.tight_layout()
    plt.show()