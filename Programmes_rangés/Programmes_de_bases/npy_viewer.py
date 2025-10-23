import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt


class NpyViewer(tk.Tk):
    """Simple viewer for 256x256 .npy images.

    Features:
    - Choose a root folder
    - Browse folder tree and list .npy files
    - Click a file to display the array as an image
    """

    def __init__(self):
        super().__init__()
        self.title('NPY Image Viewer')
        # larger default window for bigger display area
        self.geometry('1200x800')

        self.root_dir = None

        # UI layout
        self._build_controls()
        self._build_tree()
        self._build_image_area()

    def _build_controls(self):
        frame = ttk.Frame(self)
        frame.pack(fill='x', padx=6, pady=6)

        btn_open = ttk.Button(frame, text='Open folder', command=self.select_folder)
        btn_open.pack(side='left')

        btn_prev = ttk.Button(frame, text='◀ Prev', command=lambda: self._change_to_sibling(-1))
        btn_prev.pack(side='left', padx=4)
        btn_next = ttk.Button(frame, text='Next ▶', command=lambda: self._change_to_sibling(1))
        btn_next.pack(side='left', padx=4)

        self.path_label = ttk.Label(frame, text='No folder selected')
        self.path_label.pack(side='left', padx=8)
        # checkbuttons for four images and save button
        opts_frame = ttk.Frame(frame)
        opts_frame.pack(side='right')

        self.show_vars = {
            'original': tk.BooleanVar(value=True),
            'alpha': tk.BooleanVar(value=True),
            'tracks': tk.BooleanVar(value=True),
            'gamma': tk.BooleanVar(value=True),
        }

        cb1 = ttk.Checkbutton(opts_frame, text='Original', variable=self.show_vars['original'], command=self._update_from_options)
        cb2 = ttk.Checkbutton(opts_frame, text='Alpha', variable=self.show_vars['alpha'], command=self._update_from_options)
        cb3 = ttk.Checkbutton(opts_frame, text='Tracks', variable=self.show_vars['tracks'], command=self._update_from_options)
        cb4 = ttk.Checkbutton(opts_frame, text='Gamma', variable=self.show_vars['gamma'], command=self._update_from_options)
        cb1.pack(side='left', padx=2)
        cb2.pack(side='left', padx=2)
        cb3.pack(side='left', padx=2)
        cb4.pack(side='left', padx=2)

        btn_save = ttk.Button(opts_frame, text='Save view', command=self._save_view)
        btn_save.pack(side='left', padx=6)
        # keyboard bindings for left/right navigation
        self.bind('<Left>', lambda e: self._change_to_sibling(-1))
        self.bind('<Right>', lambda e: self._change_to_sibling(1))

    def _build_tree(self):
        pan = ttk.PanedWindow(self, orient='horizontal')
        pan.pack(fill='both', expand=True)

        left = ttk.Frame(pan, width=300)
        pan.add(left, weight=1)

        self.tree = ttk.Treeview(left)
        self.tree.pack(fill='both', expand=True, side='left')
        self.tree.bind('<<TreeviewSelect>>', self.on_tree_select)

        scrollbar = ttk.Scrollbar(left, orient='vertical', command=self.tree.yview)
        scrollbar.pack(side='right', fill='y')
        self.tree.configure(yscroll=scrollbar.set)

    def _build_image_area(self):
        # larger figure and slightly higher DPI for crisper plots
        self.fig, self.ax = plt.subplots(figsize=(8, 8), dpi=100)
        self.ax.axis('off')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(side='right', fill='both', expand=True)

    def select_folder(self):
        folder = filedialog.askdirectory()
        if not folder:
            return
        self.root_dir = Path(folder)
        self.path_label.config(text=str(self.root_dir))
        self.populate_tree()

    def populate_tree(self):
        self.tree.delete(*self.tree.get_children())
        if not self.root_dir:
            return
        root_id = self.tree.insert('', 'end', text=self.root_dir.name, open=True, values=(str(self.root_dir),))
        self._insert_subitems(root_id, self.root_dir)

    def _insert_subitems(self, parent_id, folder_path: Path):
        try:
            # avoid duplicate insertion: if this parent already has real children, skip
            existing_children = list(self.tree.get_children(parent_id))
            if existing_children:
                # if already populated with real entries (not just the placeholder), do nothing
                first_text = self.tree.item(existing_children[0], 'text')
                if first_text != 'loading...':
                    return

            existing_names = {self.tree.item(ch, 'text') for ch in existing_children}
            for p in sorted(folder_path.iterdir()):
                if p.is_dir():
                    # don't insert twice
                    if p.name in existing_names:
                        continue
                    node_id = self.tree.insert(parent_id, 'end', text=p.name, open=False, values=(str(p),))
                    # lazy populate (one level)
                    self.tree.insert(node_id, 'end', text='loading...')
                elif p.suffix.lower() == '.npy':
                    if p.name in existing_names:
                        continue
                    self.tree.insert(parent_id, 'end', text=p.name, values=(str(p),))
        except PermissionError:
            pass

    def on_tree_select(self, event):
        sel = self.tree.selection()
        if not sel:
            return
        node = sel[0]
        val = self.tree.item(node, 'values')
        if not val:
            return
        path = Path(val[0])
        # update last_selected_dir for navigation
        if path.is_dir():
            self.last_selected_dir = path
        else:
            self.last_selected_dir = path.parent
        # If the selected node is a directory and contains the four base .npy files,
        # automatically display them (user requested behaviour).
        if path.is_dir():
            expected = ["image_originale.npy", "image_alpha.npy", "image_tracks.npy", "image_gamma.npy"]
            sibling_paths = [path / name for name in expected]
            if all(p.exists() for p in sibling_paths):
                # display using the first file as entry point
                try:
                    self.display_npy(sibling_paths[0])
                except Exception as e:
                    messagebox.showerror('Error', f'Failed to display group: {e}')
                return
        if path.is_dir():
            # If already populated (and not only a 'loading...' placeholder) do nothing
            children = list(self.tree.get_children(node))
            if children:
                first = self.tree.item(children[0], 'text')
                if first != 'loading...':
                    return
            # remove placeholder and populate
            for ch in children:
                if self.tree.item(ch, 'text') == 'loading...':
                    self.tree.delete(ch)
            self._insert_subitems(node, path)
            return
        if path.suffix.lower() == '.npy':
            self.display_npy(path)

    def display_npy(self, path: Path):
        try:
            arr = np.load(path, allow_pickle=False)
        except Exception as e:
            messagebox.showerror('Error', f'Failed to load {path.name}: {e}')
            return
        # If the parent folder contains the four images saved by compteur, display them together
        parent = path.parent
        expected = ["image_originale.npy", "image_alpha.npy", "image_tracks.npy", "image_gamma.npy"]
        sibling_paths = [parent / name for name in expected]

        if all(p.exists() for p in sibling_paths):
            # load all arrays
            arrays = []
            for p in sibling_paths:
                try:
                    arrays.append(np.load(p, allow_pickle=False))
                except Exception as e:
                    messagebox.showerror('Error', f'Failed to load {p.name}: {e}')
                    return

            # compute display vmin/vmax across arrays
            try:
                vmax = max(a.max() for a in arrays)
                vmin = min(a.min() for a in arrays)
            except Exception:
                vmax = None
                vmin = None

            # create 2x2 grid
            self.current_group = dict(zip(['original', 'alpha', 'tracks', 'gamma'], arrays))
            self.current_group_titles = ['Original', 'Alpha', 'Tracks', 'Gamma']
            # redraw according to show_vars
            self._plot_group(vmin, vmax)
            return
            return

        # Fallback: single image display
        self.current_group = None
        if arr.ndim != 2 or arr.shape != (256, 256):
            messagebox.showwarning('Warning', f'Array shape {arr.shape} not (256,256)')
        self.fig.clf()
        self.ax = self.fig.subplots()
        if arr.max() != arr.min():
            self.ax.imshow(arr, cmap='gray', vmin=0, vmax=arr.max())
        else:
            self.ax.imshow(arr, cmap='gray')
        self.ax.axis('off')
        self.fig.suptitle(path.name)
        self.canvas.draw_idle()

    def _find_node_by_path(self, target: Path):
        # recursive search for a tree node with a given path value
        def recurse(node):
            val = self.tree.item(node, 'values')
            if val and Path(val[0]) == target:
                return node
            for ch in self.tree.get_children(node):
                res = recurse(ch)
                if res:
                    return res
            return None

        for root in self.tree.get_children(''):
            res = recurse(root)
            if res:
                return res
        return None

    def _change_to_sibling(self, direction: int):
        # direction: -1 for prev, +1 for next
        cur = getattr(self, 'last_selected_dir', None)
        if cur is None:
            # if nothing selected, try root
            cur = self.root_dir
        if cur is None:
            return
        parent = cur.parent
        if not parent.exists():
            return
        siblings = sorted([p for p in parent.iterdir() if p.is_dir()])
        if not siblings:
            return
        try:
            idx = siblings.index(cur)
        except ValueError:
            # if current not found, try to find nearest
            return
        new_idx = idx + direction
        if new_idx < 0 or new_idx >= len(siblings):
            return
        new_folder = siblings[new_idx]
        # set root to parent to ensure the tree shows the siblings
        self.root_dir = parent
        self.path_label.config(text=str(self.root_dir))
        self.populate_tree()
        node = self._find_node_by_path(new_folder)
        if node:
            self.tree.selection_set(node)
            self.tree.see(node)
            # trigger selection handler to expand/select
            self.on_tree_select(None)
            # update last_selected_dir
            self.last_selected_dir = new_folder

    def _plot_group(self, vmin, vmax):
        # plot the current group of 4 arrays according to the checkboxes
        arrays = [self.current_group[k] for k in ['original', 'alpha', 'tracks', 'gamma']]
        titles = ['Original', 'Alpha', 'Tracks', 'Gamma']
        self.fig.clf()
        axs = self.fig.subplots(1, 4)
        for ax, a, t, key in zip(axs.flat, arrays, titles, ['original', 'alpha', 'tracks', 'gamma']):
            if not self.show_vars[key].get():
                ax.axis('off')
                ax.set_title(t + ' (hidden)')
                continue
            if a.ndim != 2:
                ax.text(0.5, 0.5, f'shape {getattr(a, "shape", None)}', ha='center')
                ax.axis('off')
                continue
            if vmin is None or vmax is None or vmax == vmin:
                ax.imshow(a, cmap='gray')
            else:
                ax.imshow(a, cmap='gray', vmin=vmin, vmax=vmax)
            ax.set_title(t)
            ax.axis('off')
        self.fig.suptitle(' / '.join(titles))
        self.canvas.draw_idle()

    def _update_from_options(self):
        # redraw current group if present
        if getattr(self, 'current_group', None) is not None:
            try:
                vmax = max(a.max() for a in self.current_group.values())
                vmin = min(a.min() for a in self.current_group.values())
            except Exception:
                vmin = vmax = None
            self._plot_group(vmin, vmax)

    def _save_view(self):
        # Ask for filename and save current figure
        from tkinter import filedialog
        f = filedialog.asksaveasfilename(defaultextension='.png', filetypes=[('PNG', '*.png')], title='Save current view')
        if not f:
            return
        try:
            self.fig.savefig(f, dpi=300, bbox_inches='tight')
            messagebox.showinfo('Saved', f'Saved view to {f}')
        except Exception as e:
            messagebox.showerror('Error', f'Failed to save: {e}')


if __name__ == '__main__':
    app = NpyViewer()
    app.mainloop()
