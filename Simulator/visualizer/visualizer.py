from __future__ import annotations

import os
import re
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import ttk, messagebox

import numpy as np
import tifffile
from PIL import Image, ImageTk


RUN_PATTERN = re.compile(r"^2026_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}$")
SUPPORTED_IMAGE_SUFFIXES = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".npy"}
SUPPORTED_TEXT_SUFFIXES = {".csv", ".txt", ".json", ".log"}


@dataclass
class RunInfo:
    path: Path
    rel_parent: str
    file_count: int


class GeneratedFilesVisualizer:
    def __init__(self, root: tk.Tk, base_dir: Path):
        self.root = root
        self.base_dir = base_dir

        self.root.title("Visualizer - Fichiers générés")
        self.root.geometry("1450x850")

        self.runs: list[RunInfo] = []
        self.files_for_selected_run: list[Path] = []

        self.loaded_array: np.ndarray | None = None
        self.z_axis: int | None = None
        self.current_slice = 0

        self.preview_photo: ImageTk.PhotoImage | None = None

        self._build_ui()
        self.refresh_runs()

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill=tk.X)

        ttk.Label(top, text="Racine:", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
        self.base_label = ttk.Label(top, text=str(self.base_dir), foreground="#2c3e50")
        self.base_label.pack(side=tk.LEFT, padx=(6, 12))

        ttk.Button(top, text="🔄 Rafraîchir", command=self.refresh_runs).pack(side=tk.LEFT)

        self.summary_label = ttk.Label(top, text="")
        self.summary_label.pack(side=tk.RIGHT)

        body = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        left = ttk.Frame(body, padding=6)
        body.add(left, weight=2)

        ttk.Label(left, text="Dossiers détectés", font=("Arial", 11, "bold")).pack(anchor="w")

        columns = ("run", "location", "files")
        self.runs_tree = ttk.Treeview(left, columns=columns, show="headings", height=25)
        self.runs_tree.heading("run", text="Run")
        self.runs_tree.heading("location", text="Emplacement")
        self.runs_tree.heading("files", text="Nb fichiers")
        self.runs_tree.column("run", width=210, anchor="w")
        self.runs_tree.column("location", width=260, anchor="w")
        self.runs_tree.column("files", width=90, anchor="center")
        self.runs_tree.pack(fill=tk.BOTH, expand=True, pady=(6, 0))
        self.runs_tree.bind("<<TreeviewSelect>>", self.on_run_selected)

        middle = ttk.Frame(body, padding=6)
        body.add(middle, weight=2)

        ttk.Label(middle, text="Fichiers du run", font=("Arial", 11, "bold")).pack(anchor="w")

        self.files_list = tk.Listbox(middle, activestyle="none")
        self.files_list.pack(fill=tk.BOTH, expand=True, pady=(6, 0))
        self.files_list.bind("<<ListboxSelect>>", self.on_file_selected)

        right = ttk.Frame(body, padding=6)
        body.add(right, weight=3)

        ttk.Label(right, text="Prévisualisation", font=("Arial", 11, "bold")).pack(anchor="w")

        self.file_info_label = ttk.Label(right, text="Aucun fichier sélectionné", foreground="gray")
        self.file_info_label.pack(anchor="w", pady=(4, 2))

        self.preview_label = ttk.Label(right, text="Sélectionner un fichier")
        self.preview_label.pack(fill=tk.BOTH, expand=True)

        slider_row = ttk.Frame(right)
        slider_row.pack(fill=tk.X, pady=(6, 2))

        ttk.Label(slider_row, text="Plan Z:").pack(side=tk.LEFT)
        self.z_slider = ttk.Scale(
            slider_row,
            from_=0,
            to=0,
            orient=tk.HORIZONTAL,
            command=self.on_slider_changed,
            state="disabled",
        )
        self.z_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)

        self.z_label = ttk.Label(slider_row, text="-")
        self.z_label.pack(side=tk.LEFT)

        self.text_preview = tk.Text(right, height=12, wrap=tk.NONE)
        self.text_preview.pack(fill=tk.BOTH, expand=False)
        self.text_preview.insert("1.0", "Les aperçus CSV/TXT/JSON apparaissent ici.")
        self.text_preview.config(state=tk.DISABLED)

    def refresh_runs(self) -> None:
        self.runs = self._discover_runs(self.base_dir)

        for item_id in self.runs_tree.get_children():
            self.runs_tree.delete(item_id)

        for idx, run in enumerate(self.runs):
            self.runs_tree.insert(
                "",
                "end",
                iid=str(idx),
                values=(run.path.name, run.rel_parent, run.file_count),
            )

        self.summary_label.config(text=f"{len(self.runs)} dossier(s) trouvé(s)")

        self.files_list.delete(0, tk.END)
        self.files_for_selected_run = []
        self._clear_preview()

    def _discover_runs(self, root: Path) -> list[RunInfo]:
        runs: list[RunInfo] = []
        for current_root, dirnames, _ in os.walk(root):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for dirname in dirnames:
                if RUN_PATTERN.match(dirname):
                    full = Path(current_root) / dirname
                    file_count = sum(1 for p in full.rglob("*") if p.is_file())
                    rel_parent = str(full.parent.relative_to(root)) if full.parent != root else "."
                    runs.append(RunInfo(path=full, rel_parent=rel_parent, file_count=file_count))

        runs.sort(key=lambda r: r.path.name, reverse=True)
        return runs

    def on_run_selected(self, _event=None) -> None:
        selection = self.runs_tree.selection()
        if not selection:
            return

        run = self.runs[int(selection[0])]

        files = [p for p in run.path.rglob("*") if p.is_file()]
        files.sort(key=lambda p: str(p.relative_to(run.path)))

        self.files_for_selected_run = files
        self.files_list.delete(0, tk.END)

        for p in files:
            rel = p.relative_to(run.path)
            size_kb = p.stat().st_size / 1024
            self.files_list.insert(tk.END, f"{rel}  ({size_kb:.1f} KB)")

        self._clear_preview()

    def on_file_selected(self, _event=None) -> None:
        selected = self.files_list.curselection()
        if not selected:
            return

        path = self.files_for_selected_run[selected[0]]
        self.file_info_label.config(text=str(path))

        suffix = path.suffix.lower()
        if suffix in SUPPORTED_IMAGE_SUFFIXES:
            self._show_image_file(path)
        elif suffix in SUPPORTED_TEXT_SUFFIXES:
            self._show_text_file(path)
        else:
            self._clear_preview(message="Prévisualisation indisponible pour ce type de fichier")

    def _show_text_file(self, path: Path) -> None:
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            messagebox.showerror("Erreur", f"Lecture impossible:\n{exc}")
            return

        lines = content.splitlines()
        preview = "\n".join(lines[:250])
        if len(lines) > 250:
            preview += "\n\n... (fichier tronqué)"

        self._set_text_preview(preview)
        self._clear_image_only()

    def _show_image_file(self, path: Path) -> None:
        try:
            if path.suffix.lower() == ".npy":
                arr = np.load(path)
            elif path.suffix.lower() in {".tif", ".tiff"}:
                arr = tifffile.imread(path)
            else:
                arr = np.array(Image.open(path))

            arr = np.asarray(arr)
            if arr.size == 0:
                self._clear_preview(message="Fichier vide")
                return

            self.loaded_array = arr
            self.z_axis = self._detect_z_axis(arr)
            self.current_slice = 0

            if arr.ndim == 3 and self.z_axis is not None:
                z_len = arr.shape[self.z_axis]
                self.z_slider.config(from_=0, to=max(z_len - 1, 0), state="normal")
                self.z_slider.set(0)
                self.z_label.config(text=f"0 / {max(z_len - 1, 0)}")
            else:
                self.z_slider.config(from_=0, to=0, state="disabled")
                self.z_label.config(text="-")

            self._render_current_image()
            self._set_text_preview("")
        except Exception as exc:
            messagebox.showerror("Erreur", f"Impossible de charger ce fichier:\n{exc}")

    def on_slider_changed(self, value: str) -> None:
        if self.loaded_array is None or self.z_axis is None:
            return
        self.current_slice = int(float(value))
        z_len = self.loaded_array.shape[self.z_axis]
        self.z_label.config(text=f"{self.current_slice} / {z_len - 1}")
        self._render_current_image()

    def _render_current_image(self) -> None:
        if self.loaded_array is None:
            return

        arr = self.loaded_array
        view = self._extract_view(arr, self.current_slice)
        if view is None:
            self._clear_preview(message="Dimensions non supportées")
            return

        if view.ndim == 2:
            img_arr = self._to_uint8(view)
            image = Image.fromarray(img_arr, mode="L")
        elif view.ndim == 3 and view.shape[-1] in (3, 4):
            img_arr = self._to_uint8(view)
            mode = "RGBA" if img_arr.shape[-1] == 4 else "RGB"
            image = Image.fromarray(img_arr, mode=mode)
        else:
            self._clear_preview(message="Format d'image non supporté")
            return

        image.thumbnail((700, 500), Image.Resampling.LANCZOS)
        self.preview_photo = ImageTk.PhotoImage(image)
        self.preview_label.config(image=self.preview_photo, text="")

    def _extract_view(self, arr: np.ndarray, index: int) -> np.ndarray | None:
        if arr.ndim == 2:
            return arr

        if arr.ndim == 3:
            if self.z_axis is None:
                if arr.shape[-1] in (3, 4):
                    return arr
                return arr[0]

            if self.z_axis == 0:
                return arr[index, :, :]
            if self.z_axis == 1:
                return arr[:, index, :]
            if self.z_axis == 2:
                return arr[:, :, index]

        return None

    def _detect_z_axis(self, arr: np.ndarray) -> int | None:
        if arr.ndim != 3:
            return None

        if arr.shape[-1] in (3, 4):
            return None

        dims = list(arr.shape)
        return int(np.argmin(dims))

    @staticmethod
    def _to_uint8(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr)
        if arr.dtype == np.uint8:
            return arr

        finite_mask = np.isfinite(arr)
        if not finite_mask.any():
            return np.zeros(arr.shape, dtype=np.uint8)

        vmin = float(arr[finite_mask].min())
        vmax = float(arr[finite_mask].max())
        if vmax <= vmin:
            return np.zeros(arr.shape, dtype=np.uint8)

        scaled = (arr - vmin) / (vmax - vmin)
        scaled = np.clip(scaled * 255.0, 0, 255)
        return scaled.astype(np.uint8)

    def _clear_preview(self, message: str = "Sélectionner un fichier") -> None:
        self.loaded_array = None
        self.z_axis = None
        self.current_slice = 0
        self.z_slider.config(from_=0, to=0, state="disabled")
        self.z_label.config(text="-")
        self.preview_label.config(image="", text=message)
        self.preview_photo = None
        self._set_text_preview("Les aperçus CSV/TXT/JSON apparaissent ici.")
        self.file_info_label.config(text="Aucun fichier sélectionné", foreground="gray")

    def _clear_image_only(self) -> None:
        self.loaded_array = None
        self.z_axis = None
        self.current_slice = 0
        self.z_slider.config(from_=0, to=0, state="disabled")
        self.z_label.config(text="-")
        self.preview_label.config(image="", text="Prévisualisation texte")
        self.preview_photo = None

    def _set_text_preview(self, text: str) -> None:
        self.text_preview.config(state=tk.NORMAL)
        self.text_preview.delete("1.0", tk.END)
        self.text_preview.insert("1.0", text)
        self.text_preview.config(state=tk.DISABLED)


def main() -> None:
    script_dir = Path(__file__).resolve().parent.parent
    results_dir = script_dir / "results"
    
    if not results_dir.exists():
        results_dir.mkdir(parents=True, exist_ok=True)
    
    root = tk.Tk()
    GeneratedFilesVisualizer(root, results_dir)
    root.mainloop()


if __name__ == "__main__":
    main()
