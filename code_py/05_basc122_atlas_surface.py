#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Auto-converted from the Jupyter notebook ``map.ipynb``.

Code cells are reproduced verbatim. Markdown cells are kept as comments.
Jupyter shell-magic lines (if any) are commented out so the file is valid
Python; they are preserved for reference. Cell boundaries are marked with
``# In[...]`` to match the original notebook ordering.
"""


# In[5]:

from nilearn import datasets

atlas = datasets.fetch_atlas_basc_multiscale_2015(
    resolution=122,
    version="sym"   # optional, but good to state explicitly
)

print("Atlas keys:", atlas.keys())
print("Maps path:", atlas["maps"])
print("Number of labels:", len(atlas["labels"]))
print("First 10 labels:", atlas["labels"][:10])

# In[7]:

import matplotlib.pyplot as plt
from nilearn import datasets, plotting, surface

# -------------------------------------------------
# 1. Fetch BASC 122 atlas
# -------------------------------------------------
atlas = datasets.fetch_atlas_basc_multiscale_2015(
    resolution=122,
    version="sym"
)
basc_122 = atlas["maps"]

print("Atlas path:", basc_122)
print("N labels including background:", len(atlas["labels"]))

# -------------------------------------------------
# 2. Fetch fsaverage surface
# -------------------------------------------------
fsaverage = datasets.fetch_surf_fsaverage()

# -------------------------------------------------
# 3. Project volume atlas to surface
#    Use nearest_most_frequent for deterministic atlas labels
# -------------------------------------------------
texture_left = surface.vol_to_surf(
    basc_122,
    fsaverage.pial_left,
    interpolation="nearest_most_frequent"
)

texture_right = surface.vol_to_surf(
    basc_122,
    fsaverage.pial_right,
    interpolation="nearest_most_frequent"
)

# -------------------------------------------------
# 4. Create a real 4-panel figure
# -------------------------------------------------
fig, axes = plt.subplots(
    2, 2,
    figsize=(14, 10),
    subplot_kw={"projection": "3d"}
)

views = [
    ("left", "lateral", fsaverage.infl_left, fsaverage.sulc_left, texture_left, "Left lateral"),
    ("left", "medial",  fsaverage.infl_left, fsaverage.sulc_left, texture_left, "Left medial"),
    ("right", "lateral", fsaverage.infl_right, fsaverage.sulc_right, texture_right, "Right lateral"),
    ("right", "medial",  fsaverage.infl_right, fsaverage.sulc_right, texture_right, "Right medial"),
]

for ax, (hemi, view, surf_mesh, sulc, texture, title) in zip(axes.ravel(), views):
    plotting.plot_surf_roi(
        surf_mesh=surf_mesh,
        roi_map=texture,
        hemi=hemi,
        view=view,
        bg_map=sulc,
        bg_on_data=True,
        cmap="tab20",
        colorbar=False,
        axes=ax,
        title=title
    )

plt.suptitle("BASC 122 functional atlas", fontsize=16, y=0.98)
plt.tight_layout()
plt.savefig("BASC122_surface_4views.png", dpi=600, bbox_inches="tight", facecolor="white")
plt.show()

# In[8]:

import matplotlib.pyplot as plt
from nilearn import datasets, plotting, image, surface

# -----------------------------
# 1. Fetch BASC atlas at 122 ROIs
# -----------------------------
atlas = datasets.fetch_atlas_basc_multiscale_2015()
basc_122 = atlas.scale122

print("Atlas file:", basc_122)

# -----------------------------
# 2. Plot directly on cortical surface
# -----------------------------
fig = plt.figure(figsize=(14, 10))

plotting.plot_img_on_surf(
    basc_122,
    views=["lateral", "medial"],
    hemispheres=["left", "right"],
    colorbar=False,
    threshold=0,
    bg_on_data=True,
    darkness=0.5,
    cmap="tab20",   # good categorical look for parcels
    figure=fig
)

plt.suptitle("BASC 122 atlas", fontsize=18, y=0.98)
plt.savefig("basc122_surface_publication.pdf", dpi=1000, bbox_inches="tight")
plt.show()

# In[9]:

from nilearn import plotting
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 8))

display = plotting.plot_roi(
    basc_122,
    display_mode="ortho",
    draw_cross=False,
    black_bg=False,
    cmap="tab20",
    annotate=False,
    dim=-0.3,
    cut_coords=(0, -20, 20),
    figure=fig
)

plt.savefig("basc122_ortho_publication.png", dpi=600, bbox_inches="tight")
plt.show()

# In[13]:

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from nilearn import datasets, plotting, surface

# -------------------------------------------------
# 1. Fetch BASC 122 atlas
# -------------------------------------------------
atlas = datasets.fetch_atlas_basc_multiscale_2015(
    resolution=122,
    version="sym"
)
basc_122 = atlas["maps"]

# -------------------------------------------------
# 2. Fetch fsaverage inflated surface
# -------------------------------------------------
fsaverage = datasets.fetch_surf_fsaverage()

# -------------------------------------------------
# 3. Project atlas to surface
# -------------------------------------------------
texture_left = surface.vol_to_surf(
    basc_122,
    fsaverage.pial_left,
    interpolation="nearest_most_frequent"
)

texture_right = surface.vol_to_surf(
    basc_122,
    fsaverage.pial_right,
    interpolation="nearest_most_frequent"
)

# -------------------------------------------------
# 4. Build categorical colormap
# -------------------------------------------------
n_labels = 123
base = plt.cm.nipy_spectral(np.linspace(0, 1, n_labels))
base[0] = [1, 1, 1, 0]
parcel_cmap = ListedColormap(base)

# -------------------------------------------------
# 5. Create figure
# -------------------------------------------------
fig = plt.figure(figsize=(14, 8), facecolor="white")

gs = fig.add_gridspec(
    2, 2,
    left=0.02, right=0.98,
    bottom=0.04, top=0.93,
    wspace=0.02, hspace=0.08
)

axes = [
    fig.add_subplot(gs[0, 0], projection="3d"),
    fig.add_subplot(gs[0, 1], projection="3d"),
    fig.add_subplot(gs[1, 0], projection="3d"),
    fig.add_subplot(gs[1, 1], projection="3d"),
]

panels = [
    ("left",  "lateral", fsaverage.infl_left,  fsaverage.sulc_left,  texture_left,  "Left lateral"),
    ("right", "lateral", fsaverage.infl_right, fsaverage.sulc_right, texture_right, "Right lateral"),
    ("left",  "medial",  fsaverage.infl_left,  fsaverage.sulc_left,  texture_left,  "Left medial"),
    ("right", "medial",  fsaverage.infl_right, fsaverage.sulc_right, texture_right, "Right medial"),
]

for ax, (hemi, view, mesh, sulc, texture, title) in zip(axes, panels):
    plotting.plot_surf_roi(
        surf_mesh=mesh,
        roi_map=texture,
        hemi=hemi,
        view=view,
        bg_map=sulc,
        bg_on_data=True,
        cmap=parcel_cmap,
        colorbar=False,
        axes=ax,
        title=title
    )

fig.suptitle("BASC-122 functional parcellation", fontsize=18, fontweight="bold")

plt.savefig(
    "BASC122_high_impact_style.png",
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)
plt.show()

# In[15]:

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from nilearn import datasets, plotting, surface, image

# -------------------------------------------------
# 1. Load BASC-122 atlas
# -------------------------------------------------
atlas = datasets.fetch_atlas_basc_multiscale_2015(
    resolution=122,
    version="sym"
)
basc_122 = atlas["maps"]

# -------------------------------------------------
# 2. Choose the parcel IDs you want to highlight
#    Replace these with your own important BASC regions
# -------------------------------------------------
selected_labels = [10, 25, 48, 71, 95]

# -------------------------------------------------
# 3. Keep only those labels in the volume
#    Remap them to 1..N so each gets its own color
# -------------------------------------------------
img = image.load_img(basc_122)
data = img.get_fdata()

masked = np.zeros_like(data, dtype=np.int32)
for new_id, old_id in enumerate(selected_labels, start=1):
    masked[data == old_id] = new_id

masked_img = image.new_img_like(img, masked)

# -------------------------------------------------
# 4. Fetch inflated cortical surface
# -------------------------------------------------
fsaverage = datasets.fetch_surf_fsaverage()

# -------------------------------------------------
# 5. Project to surface
#    nearest_most_frequent is the right choice for label atlases
# -------------------------------------------------
texture_left = surface.vol_to_surf(
    masked_img,
    fsaverage.pial_left,
    interpolation="nearest_most_frequent"
)

texture_right = surface.vol_to_surf(
    masked_img,
    fsaverage.pial_right,
    interpolation="nearest_most_frequent"
)

# -------------------------------------------------
# 6. Build a discrete colormap
#    0 = transparent background
# -------------------------------------------------
colors = np.array([
    [1.0, 1.0, 1.0, 0.0],   # background
    [0.85, 0.15, 0.15, 1.0],# red
    [0.15, 0.35, 0.85, 1.0],# blue
    [0.15, 0.70, 0.30, 1.0],# green
    [0.90, 0.55, 0.10, 1.0],# orange
    [0.60, 0.20, 0.75, 1.0],# purple
])

roi_cmap = ListedColormap(colors[:len(selected_labels)+1])

# -------------------------------------------------
# 7. Plot multiple perspectives
# -------------------------------------------------
fig = plt.figure(figsize=(12, 8), facecolor="white")
gs = fig.add_gridspec(2, 2, wspace=0.01, hspace=0.03)

axes = [
    fig.add_subplot(gs[0, 0], projection="3d"),
    fig.add_subplot(gs[0, 1], projection="3d"),
    fig.add_subplot(gs[1, 0], projection="3d"),
    fig.add_subplot(gs[1, 1], projection="3d"),
]

views = [
    ("left", "lateral", fsaverage.infl_left, fsaverage.sulc_left, texture_left,  "Left lateral"),
    ("right", "lateral", fsaverage.infl_right, fsaverage.sulc_right, texture_right, "Right lateral"),
    ("left", "medial", fsaverage.infl_left, fsaverage.sulc_left, texture_left, "Left medial"),
    ("right", "medial", fsaverage.infl_right, fsaverage.sulc_right, texture_right, "Right medial"),
]

for ax, (hemi, view, mesh, sulc, tex, title) in zip(axes, views):
    plotting.plot_surf_roi(
        surf_mesh=mesh,
        roi_map=tex,
        hemi=hemi,
        view=view,
        bg_map=sulc,
        bg_on_data=True,
        cmap=roi_cmap,
        colorbar=False,
        axes=ax,
        title=title
    )

plt.suptitle("Selected BASC-122 regions on inflated cortical surface", fontsize=15)
plt.savefig("BASC122_selected_regions_4views.png", dpi=600, bbox_inches="tight", facecolor="white")
plt.show()

# In[18]:

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from nilearn.datasets import (
    fetch_atlas_basc_multiscale_2015,
    load_fsaverage,
    load_fsaverage_data,
)
from nilearn.surface import SurfaceImage
from nilearn.plotting import plot_surf, show

# -------------------------------------------------
# 1. Load BASC-122 atlas
# -------------------------------------------------
atlas = fetch_atlas_basc_multiscale_2015(resolution=122, version="sym")
atlas_img = nib.load(atlas["maps"])
atlas_data = atlas_img.get_fdata()

# -------------------------------------------------
# 2. Select BASC labels to highlight
#    Change these to the regions you want
# -------------------------------------------------
selected_labels = [10, 25, 48, 71, 95, 110]

masked_data = np.zeros_like(atlas_data, dtype=np.int16)
for new_idx, old_label in enumerate(selected_labels, start=1):
    masked_data[atlas_data == old_label] = new_idx

masked_img = nib.Nifti1Image(masked_data, atlas_img.affine, atlas_img.header)

# -------------------------------------------------
# 3. Load fine fsaverage mesh
# -------------------------------------------------
fsaverage = load_fsaverage(mesh="fsaverage")

# -------------------------------------------------
# 4. Load sulcal background
# -------------------------------------------------
fsaverage_sulcal = load_fsaverage_data(
    mesh="fsaverage",
    data_type="sulcal",
    mesh_type="inflated",
)

# -------------------------------------------------
# 5. Project BASC volume to pial surface
# -------------------------------------------------
surf_img = SurfaceImage.from_volume(
    mesh=fsaverage["pial"],
    volume_img=masked_img,
    interpolation="nearest_most_frequent",
)

surf_left = np.asarray(surf_img.data.parts["left"])
surf_right = np.asarray(surf_img.data.parts["right"])

sulc_left = np.asarray(fsaverage_sulcal.data.parts["left"])
sulc_right = np.asarray(fsaverage_sulcal.data.parts["right"])

# -------------------------------------------------
# 6. Discrete colors
# -------------------------------------------------
region_colors = [
    (0.85, 0.20, 0.20, 1.0),  # red
    (0.20, 0.35, 0.85, 1.0),  # blue
    (0.20, 0.70, 0.35, 1.0),  # green
    (0.95, 0.55, 0.10, 1.0),  # orange
    (0.60, 0.25, 0.75, 1.0),  # purple
    (0.95, 0.80, 0.20, 1.0),  # yellow
]
cmap = ListedColormap(region_colors[:len(selected_labels)])

# -------------------------------------------------
# 7. Plot four views
# -------------------------------------------------
fig = plt.figure(figsize=(10, 8), facecolor="white")
gs = fig.add_gridspec(2, 2, wspace=0.01, hspace=0.01)

axes = [
    fig.add_subplot(gs[0, 0], projection="3d"),
    fig.add_subplot(gs[0, 1], projection="3d"),
    fig.add_subplot(gs[1, 0], projection="3d"),
    fig.add_subplot(gs[1, 1], projection="3d"),
]

plot_surf(
    surf_mesh=fsaverage["inflated"],
    surf_map=surf_left,
    hemi="left",
    view="lateral",
    bg_map=sulc_left,
    cmap=cmap,
    vmin=1,
    vmax=len(selected_labels),
    threshold=0.5,
    colorbar=False,
    bg_on_data=True,
    axes=axes[0],
    title="Left lateral",
)

plot_surf(
    surf_mesh=fsaverage["inflated"],
    surf_map=surf_right,
    hemi="right",
    view="lateral",
    bg_map=sulc_right,
    cmap=cmap,
    vmin=1,
    vmax=len(selected_labels),
    threshold=0.5,
    colorbar=False,
    bg_on_data=True,
    axes=axes[1],
    title="Right lateral",
)

plot_surf(
    surf_mesh=fsaverage["inflated"],
    surf_map=surf_left,
    hemi="left",
    view="medial",
    bg_map=sulc_left,
    cmap=cmap,
    vmin=1,
    vmax=len(selected_labels),
    threshold=0.5,
    colorbar=False,
    bg_on_data=True,
    axes=axes[2],
    title="Left medial",
)

plot_surf(
    surf_mesh=fsaverage["inflated"],
    surf_map=surf_right,
    hemi="right",
    view="medial",
    bg_map=sulc_right,
    cmap=cmap,
    vmin=1,
    vmax=len(selected_labels),
    threshold=0.5,
    colorbar=False,
    bg_on_data=True,
    axes=axes[3],
    title="Right medial",
)

plt.suptitle(
    "Selected BASC-122 regions on fine fsaverage surface",
    fontsize=15,
    fontweight="bold"
)
plt.savefig(
    "BASC122_fine_surface_selected_regions.png",
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)
show()

# In[19]:

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from nilearn.datasets import (
    fetch_atlas_basc_multiscale_2015,
    load_fsaverage,
    load_fsaverage_data,
)
from nilearn.surface import SurfaceImage
from nilearn.plotting import plot_surf, show

# -------------------------------------------------
# 1. Load full BASC-122 atlas
# -------------------------------------------------
atlas = fetch_atlas_basc_multiscale_2015(resolution=122, version="sym")
atlas_img = nib.load(atlas["maps"])

# -------------------------------------------------
# 2. Load fine fsaverage mesh
# -------------------------------------------------
fsaverage = load_fsaverage(mesh="fsaverage")

# -------------------------------------------------
# 3. Load sulcal background for inflated surface
# -------------------------------------------------
fsaverage_sulcal = load_fsaverage_data(
    mesh="fsaverage",
    data_type="sulcal",
    mesh_type="inflated",
)

# -------------------------------------------------
# 4. Project full atlas to the cortical surface
# -------------------------------------------------
surf_img = SurfaceImage.from_volume(
    mesh=fsaverage["pial"],
    volume_img=atlas_img,
    interpolation="nearest_most_frequent",
)

surf_left = np.asarray(surf_img.data.parts["left"])
surf_right = np.asarray(surf_img.data.parts["right"])

sulc_left = np.asarray(fsaverage_sulcal.data.parts["left"])
sulc_right = np.asarray(fsaverage_sulcal.data.parts["right"])

# -------------------------------------------------
# 5. Build a large discrete colormap for 122 parcels
#    label 0 = background
# -------------------------------------------------
n_regions = 122
colors = plt.cm.gist_ncar(np.linspace(0, 1, n_regions + 1))
colors[0] = [0, 0, 0, 0]  # transparent background
cmap = ListedColormap(colors)

# -------------------------------------------------
# 6. Plot four views in the style you liked
# -------------------------------------------------
fig = plt.figure(figsize=(10, 8), facecolor="white")
gs = fig.add_gridspec(2, 2, wspace=0.01, hspace=0.01)

axes = [
    fig.add_subplot(gs[0, 0], projection="3d"),
    fig.add_subplot(gs[0, 1], projection="3d"),
    fig.add_subplot(gs[1, 0], projection="3d"),
    fig.add_subplot(gs[1, 1], projection="3d"),
]

plot_surf(
    surf_mesh=fsaverage["inflated"],
    surf_map=surf_left,
    hemi="left",
    view="lateral",
    bg_map=sulc_left,
    cmap=cmap,
    vmin=1,
    vmax=122,
    threshold=0.5,
    colorbar=False,
    bg_on_data=True,
    axes=axes[0],
    title="Left lateral",
)

plot_surf(
    surf_mesh=fsaverage["inflated"],
    surf_map=surf_right,
    hemi="right",
    view="lateral",
    bg_map=sulc_right,
    cmap=cmap,
    vmin=1,
    vmax=122,
    threshold=0.5,
    colorbar=False,
    bg_on_data=True,
    axes=axes[1],
    title="Right lateral",
)

plot_surf(
    surf_mesh=fsaverage["inflated"],
    surf_map=surf_left,
    hemi="left",
    view="medial",
    bg_map=sulc_left,
    cmap=cmap,
    vmin=1,
    vmax=122,
    threshold=0.5,
    colorbar=False,
    bg_on_data=True,
    axes=axes[2],
    title="Left medial",
)

plot_surf(
    surf_mesh=fsaverage["inflated"],
    surf_map=surf_right,
    hemi="right",
    view="medial",
    bg_map=sulc_right,
    cmap=cmap,
    vmin=1,
    vmax=122,
    threshold=0.5,
    colorbar=False,
    bg_on_data=True,
    axes=axes[3],
    title="Right medial",
)

plt.suptitle(
    "BASC-122 atlas on fine fsaverage surface",
    fontsize=15,
    fontweight="bold"
)

plt.savefig(
    "BASC122_all_regions_fine_surface.png",
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)

show()

# In[14]:

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from nilearn import datasets, plotting, surface

# -------------------------------------------------
# 1. Fetch BASC 122 atlas
# -------------------------------------------------
atlas = datasets.fetch_atlas_basc_multiscale_2015(
    resolution=122,
    version="sym"
)
basc_122 = atlas["maps"]

print("Atlas path:", basc_122)
print("Number of labels:", len(atlas["labels"]))

# -------------------------------------------------
# 2. Fetch fsaverage inflated surface
# -------------------------------------------------
fsaverage = datasets.fetch_surf_fsaverage()

# -------------------------------------------------
# 3. Project atlas to surface
# -------------------------------------------------
texture_left = surface.vol_to_surf(
    basc_122,
    fsaverage.pial_left,
    interpolation="nearest_most_frequent"
)

texture_right = surface.vol_to_surf(
    basc_122,
    fsaverage.pial_right,
    interpolation="nearest_most_frequent"
)

# -------------------------------------------------
# 4. Create a categorical colormap
# -------------------------------------------------
n_labels = 123  # 0–122
colors = plt.cm.nipy_spectral(np.linspace(0, 1, n_labels))
colors[0] = [1, 1, 1, 0]  # Transparent background
parcel_cmap = ListedColormap(colors)

# -------------------------------------------------
# 5. Create a publication-style multi-panel layout
# -------------------------------------------------
fig = plt.figure(figsize=(14, 8), facecolor="white")
gs = fig.add_gridspec(2, 2, wspace=0.02, hspace=0.02)

axes = [
    fig.add_subplot(gs[0, 0], projection="3d"),
    fig.add_subplot(gs[0, 1], projection="3d"),
    fig.add_subplot(gs[1, 0], projection="3d"),
    fig.add_subplot(gs[1, 1], projection="3d"),
]

panels = [
    ("left", "lateral", fsaverage.infl_left, fsaverage.sulc_left, texture_left, "Left lateral"),
    ("right", "lateral", fsaverage.infl_right, fsaverage.sulc_right, texture_right, "Right lateral"),
    ("left", "medial", fsaverage.infl_left, fsaverage.sulc_left, texture_left, "Left medial"),
    ("right", "medial", fsaverage.infl_right, fsaverage.sulc_right, texture_right, "Right medial"),
]

for ax, (hemi, view, mesh, sulc, texture, title) in zip(axes, panels):
    plotting.plot_surf_roi(
        surf_mesh=mesh,
        roi_map=texture,
        hemi=hemi,
        view=view,
        bg_map=sulc,
        bg_on_data=True,
        cmap=parcel_cmap,
        colorbar=False,
        axes=ax,
        title=title
    )

# Main title
fig.suptitle(
    "BASC-122 Functional Brain Parcellation",
    fontsize=16,
    fontweight="bold"
)

# Save high-resolution figure
plt.savefig(
    "BASC122_high_impact_style.png",
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()

# In[24]:

import numpy as np
import nibabel as nib
from nilearn import plotting, image, datasets

atlas = datasets.fetch_atlas_basc_multiscale_2015(resolution=122, version="sym")
atlas_img = nib.load(atlas["maps"])
atlas_data = atlas_img.get_fdata()

missing_labels = [9, 10, 11, 14, 18, 46, 54, 62, 65, 90, 108, 121]

missing_mask = np.isin(atlas_data, missing_labels).astype(int)
missing_img = image.new_img_like(atlas_img, missing_mask)

plotting.plot_roi(
    missing_img,
    title="BASC-122 parcels not represented on cortical surface",
    cmap="Reds",
    display_mode="ortho",
    cut_coords=(0, -65, -43),
    draw_cross=False,
    annotate=True,
    black_bg=False,
    colorbar=True,
)

plotting.show()

# In[9]:

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.ndimage import center_of_mass

from nilearn import datasets, surface, plotting

# -------------------------------------------------
# 1. Load BASC-122 atlas
# -------------------------------------------------
atlas = datasets.fetch_atlas_basc_multiscale_2015(
    resolution=122,
    version="sym"
)
atlas_img = nib.load(atlas["maps"])
atlas_data = atlas_img.get_fdata().astype(int)
affine = atlas_img.affine

# -------------------------------------------------
# 2. Assign each BASC parcel to an approximate group
#    based on centroid in MNI space
# -------------------------------------------------
# group IDs:
# 0 background
# 1 left frontal
# 2 right frontal
# 3 left parietal
# 4 right parietal
# 5 left temporal
# 6 right temporal
# 7 left occipital
# 8 right occipital
# 9 deep/other

grouped_data = np.zeros_like(atlas_data, dtype=np.int16)

parcel_to_group = {}

for label in range(1, 123):
    mask = (atlas_data == label)
    if not np.any(mask):
        continue

    com_voxel = center_of_mass(mask)
    x, y, z = nib.affines.apply_affine(affine, com_voxel)

    left = x < 0

    # very simple approximate anatomical grouping in MNI space
    if y > 20:
        group = 1 if left else 2   # frontal
    elif y < -55:
        group = 7 if left else 8   # occipital
    elif z < 0 and -55 <= y <= 20:
        group = 5 if left else 6   # temporal
    elif z >= 0 and -55 <= y <= 20:
        group = 3 if left else 4   # parietal
    else:
        group = 9                  # deep/other

    grouped_data[mask] = group
    parcel_to_group[label] = group

grouped_img = nib.Nifti1Image(grouped_data, affine, atlas_img.header)

# -------------------------------------------------
# 3. Load fsaverage surface
# -------------------------------------------------
fsaverage = datasets.fetch_surf_fsaverage()

# -------------------------------------------------
# 4. Project grouped atlas to surface
# -------------------------------------------------
texture_left = surface.vol_to_surf(
    grouped_img,
    fsaverage.pial_left,
    interpolation="nearest_most_frequent"
)

texture_right = surface.vol_to_surf(
    grouped_img,
    fsaverage.pial_right,
    interpolation="nearest_most_frequent"
)

# -------------------------------------------------
# 5. Define colors for groups
# -------------------------------------------------
colors = np.array([
    [1.00, 1.00, 1.00, 0.00],  # 0 background transparent
    [0.80, 0.20, 0.20, 1.00],  # 1 left frontal
    [0.95, 0.45, 0.45, 1.00],  # 2 right frontal
    [0.20, 0.45, 0.85, 1.00],  # 3 left parietal
    [0.45, 0.65, 0.95, 1.00],  # 4 right parietal
    [0.20, 0.70, 0.35, 1.00],  # 5 left temporal
    [0.50, 0.85, 0.60, 1.00],  # 6 right temporal
    [0.60, 0.25, 0.75, 1.00],  # 7 left occipital
    [0.80, 0.55, 0.90, 1.00],  # 8 right occipital
    [0.55, 0.55, 0.55, 1.00],  # 9 deep/other
])

group_cmap = ListedColormap(colors)

# -------------------------------------------------
# 6. Plot four 3D views
# -------------------------------------------------
fig = plt.figure(figsize=(12, 8), facecolor="white")
gs = fig.add_gridspec(2, 2, wspace=0.02, hspace=0.02)

axes = [
    fig.add_subplot(gs[0, 0], projection="3d"),
    fig.add_subplot(gs[0, 1], projection="3d"),
    fig.add_subplot(gs[1, 0], projection="3d"),
    fig.add_subplot(gs[1, 1], projection="3d"),
]

views = [
    ("left", "lateral", fsaverage.infl_left, fsaverage.sulc_left, texture_left, "Left lateral"),
    ("right", "lateral", fsaverage.infl_right, fsaverage.sulc_right, texture_right, "Right lateral"),
    ("left", "medial", fsaverage.infl_left, fsaverage.sulc_left, texture_left, "Left medial"),
    ("right", "medial", fsaverage.infl_right, fsaverage.sulc_right, texture_right, "Right medial"),
]

for ax, (hemi, view, mesh, sulc, tex, title) in zip(axes, views):
    plotting.plot_surf_roi(
        surf_mesh=mesh,
        roi_map=tex,
        hemi=hemi,
        view=view,
        bg_map=sulc,
        bg_on_data=True,
        cmap=group_cmap,
        colorbar=False,
        axes=ax,
        title=title
    )

fig.suptitle(
    "BASC-122 grouped into approximate cortical lobes",
    fontsize=16,
    fontweight="bold"
)

plt.savefig(
    "BASC122_grouped_lobes_surface.png",
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)
plt.show()

# In[10]:

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap

from nilearn.datasets import (
    fetch_atlas_basc_multiscale_2015,
    load_fsaverage,
    load_fsaverage_data,
)
from nilearn.surface import SurfaceImage
from nilearn.plotting import plot_surf, show

# -------------------------------------------------
# 1. Load BASC-122 atlas
# -------------------------------------------------
atlas = fetch_atlas_basc_multiscale_2015(
    resolution=122,
    version="sym"
)
atlas_img = nib.load(atlas["maps"])

# Optional: save lookup table with parcel IDs/names
lut = atlas["lut"]
lut.to_csv("BASC122_LUT.csv", index=False)

# -------------------------------------------------
# 2. Load fine fsaverage mesh
# -------------------------------------------------
fsaverage = load_fsaverage(mesh="fsaverage")

# -------------------------------------------------
# 3. Load sulcal background for inflated surface
# -------------------------------------------------
fsaverage_sulcal = load_fsaverage_data(
    mesh="fsaverage",
    data_type="sulcal",
    mesh_type="inflated",
)

# -------------------------------------------------
# 4. Project full BASC-122 atlas to the surface
# -------------------------------------------------
surf_img = SurfaceImage.from_volume(
    mesh=fsaverage["pial"],
    volume_img=atlas_img,
    interpolation="nearest_most_frequent",
)

surf_left = np.asarray(surf_img.data.parts["left"])
surf_right = np.asarray(surf_img.data.parts["right"])

sulc_left = np.asarray(fsaverage_sulcal.data.parts["left"])
sulc_right = np.asarray(fsaverage_sulcal.data.parts["right"])

# -------------------------------------------------
# 5. Check how many labels appear on the surface
# -------------------------------------------------
left_labels = np.unique(surf_left.astype(int))
left_labels = left_labels[left_labels > 0]

right_labels = np.unique(surf_right.astype(int))
right_labels = right_labels[right_labels > 0]

all_surface_labels = np.union1d(left_labels, right_labels)

print("Number of labels on left surface:", len(left_labels))
print("Number of labels on right surface:", len(right_labels))
print("Total unique labels across both hemispheres:", len(all_surface_labels))

# -------------------------------------------------
# 6. Build a categorical colormap for 122 parcels
# -------------------------------------------------
n_labels = 123  # includes 0 as background
base = plt.cm.nipy_spectral(np.linspace(0, 1, n_labels))
base[0] = [1, 1, 1, 0]  # transparent background
parcel_cmap = ListedColormap(base)

# -------------------------------------------------
# 7. Create figure layout
# -------------------------------------------------
fig = plt.figure(figsize=(12, 10), facecolor="white")
gs = fig.add_gridspec(
    2, 2,
    left=0.03, right=0.97,
    bottom=0.12, top=0.90,
    wspace=0.04, hspace=0.08
)

axes = [
    fig.add_subplot(gs[0, 0], projection="3d"),
    fig.add_subplot(gs[0, 1], projection="3d"),
    fig.add_subplot(gs[1, 0], projection="3d"),
    fig.add_subplot(gs[1, 1], projection="3d"),
]

# -------------------------------------------------
# 8. Plot the four views
# -------------------------------------------------
plot_surf(
    surf_mesh=fsaverage["inflated"],
    surf_map=surf_left,
    hemi="left",
    view="lateral",
    bg_map=sulc_left,
    cmap=parcel_cmap,
    vmin=1,
    vmax=122,
    threshold=0.5,
    colorbar=False,
    bg_on_data=True,
    axes=axes[0],
    title="Left lateral",
)

plot_surf(
    surf_mesh=fsaverage["inflated"],
    surf_map=surf_right,
    hemi="right",
    view="lateral",
    bg_map=sulc_right,
    cmap=parcel_cmap,
    vmin=1,
    vmax=122,
    threshold=0.5,
    colorbar=False,
    bg_on_data=True,
    axes=axes[1],
    title="Right lateral",
)

plot_surf(
    surf_mesh=fsaverage["inflated"],
    surf_map=surf_left,
    hemi="left",
    view="medial",
    bg_map=sulc_left,
    cmap=parcel_cmap,
    vmin=1,
    vmax=122,
    threshold=0.5,
    colorbar=False,
    bg_on_data=True,
    axes=axes[2],
    title="Left medial",
)

plot_surf(
    surf_mesh=fsaverage["inflated"],
    surf_map=surf_right,
    hemi="right",
    view="medial",
    bg_map=sulc_right,
    cmap=parcel_cmap,
    vmin=1,
    vmax=122,
    threshold=0.5,
    colorbar=False,
    bg_on_data=True,
    axes=axes[3],
    title="Right medial",
)

# -------------------------------------------------
# 9. Add figure title
# -------------------------------------------------
fig.suptitle(
    "BASC-122 atlas on fine fsaverage surface",
    fontsize=20,
    fontweight="bold"
)

# -------------------------------------------------
# 10. Add colorbar legend
# -------------------------------------------------
norm = mpl.colors.Normalize(vmin=1, vmax=122)
sm = mpl.cm.ScalarMappable(cmap=parcel_cmap, norm=norm)
sm.set_array([])

cbar = fig.colorbar(
    sm,
    ax=axes,
    orientation="horizontal",
    fraction=0.03,
    pad=0.04
)

cbar.set_label("BASC-122 parcel index", fontsize=12)
cbar.ax.tick_params(labelsize=9)

# You can control how many ticks appear:
cbar.set_ticks([1, 20, 40, 60, 80, 100, 122])

# -------------------------------------------------
# 11. Save outputs
# -------------------------------------------------
png_file = "BASC122_surface_with_colorbar.png"
pdf_file = "BASC122_surface_with_colorbar.pdf"

plt.savefig(png_file, dpi=600, bbox_inches="tight", facecolor="white")
plt.savefig(pdf_file, dpi=600, bbox_inches="tight", facecolor="white")

print("Saved:", png_file)
print("Saved:", pdf_file)
print("Saved LUT:", "BASC122_LUT.csv")

plt.show()

# In[2]:

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from nilearn import datasets, surface
from surfplot import Plot

atlas = datasets.fetch_atlas_basc_multiscale_2015(resolution=122, version="sym")
basc_122 = atlas["maps"]

fsavg = datasets.fetch_surf_fsaverage()

surf_left = surface.vol_to_surf(
    basc_122, fsavg.pial_left, interpolation="nearest_most_frequent"
).astype(int)

surf_right = surface.vol_to_surf(
    basc_122, fsavg.pial_right, interpolation="nearest_most_frequent"
).astype(int)

colors = plt.cm.nipy_spectral(np.linspace(0, 1, 123))
colors[0] = [0, 0, 0, 0]
cmap = ListedColormap(colors)

p = Plot(
    fsavg.infl_left,
    fsavg.infl_right,
    layout="grid",
    views=["lateral", "medial"],
    size=(1100, 850),
    zoom=1.4,
    background=(1, 1, 1),
    brightness=0.6,
)

p.add_layer(
    {"left": fsavg.sulc_left, "right": fsavg.sulc_right},
    cmap="binary_r",
    cbar=False
)

p.add_layer(
    {"left": surf_left, "right": surf_right},
    cmap=cmap,
    color_range=(1, 122),
    zero_transparent=True,
    cbar=False
)

p.add_layer(
    {"left": surf_left, "right": surf_right},
    cmap="gray",
    color_range=(1, 122),
    zero_transparent=True,
    cbar=False,
    as_outline=True
)

fig = p.build()
fig.suptitle("BASC-122 Functional Brain Parcellation", fontsize=22, fontweight="bold", y=0.98)
plt.show()


# In[1]:

# Code source: Dan Gale
# License: BSD 3 clause

from surfplot import Plot
from surfplot.datasets import load_example_data
from neuromaps.datasets import fetch_fslr

surfaces = fetch_fslr()
lh, rh = surfaces['inflated']

p = Plot(lh, rh)

# shading
lh_sulc, rh_sulc = surfaces['sulc']
p.add_layer({'left': lh_sulc, 'right': rh_sulc}, cmap='binary_r', cbar=False)

color_range = (0, 12)

# add default mode association stats
default = load_example_data(join=True)
p.add_layer(default, cmap='Blues_r', color_range=color_range,
            cbar_label='Default mode')

# add frontoparietal assocation stats
fronto = load_example_data('frontoparietal', join=True)
p.add_layer(fronto, cmap='Greens_r', color_range=color_range,
            cbar_label='Frontoparietal')

# create a clean looking set of colorbars. Only show labels for outer colorbar,
# given that both colorbars have the same range.
cbar_kws = dict(outer_labels_only=True, pad=.02, n_ticks=2, decimals=0)
fig = p.build(cbar_kws=cbar_kws)
# add units to colorbar
fig.axes[1].set_xlabel('z', labelpad=-11, fontstyle='italic')
fig.show()

# In[1]:

import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass

import netplotbrain
from nilearn.datasets import fetch_atlas_basc_multiscale_2015

# -------------------------------------------------
# 1. Load BASC-122 atlas
# -------------------------------------------------
atlas = fetch_atlas_basc_multiscale_2015(resolution=122, version="sym")
img = nib.load(atlas["maps"])
data = img.get_fdata().astype(int)
affine = img.affine

# -------------------------------------------------
# 2. Compute centroid of each label in world (MNI) coordinates
# -------------------------------------------------
labels = np.unique(data)
labels = labels[labels > 0]

rows = []
for label in labels:
    mask = (data == label)
    if not np.any(mask):
        continue

    # centroid in voxel space
    com_voxel = center_of_mass(mask)

    # convert to MNI/world coordinates
    xyz = nib.affines.apply_affine(affine, com_voxel)

    rows.append({
        "id": int(label),
        "x": float(xyz[0]),
        "y": float(xyz[1]),
        "z": float(xyz[2]),
        "label": f"BASC-{label}"
    })

nodes = pd.DataFrame(rows)

print("Number of nodes:", len(nodes))
print(nodes.head())

# -------------------------------------------------
# 3. Assign colors
# -------------------------------------------------
cmap = plt.cm.gist_ncar
colors = [cmap(i / 122) for i in range(122)]
nodes["color"] = colors[:len(nodes)]
nodes["size"] = 40  # adjust if needed

# -------------------------------------------------
# 4. Plot with netplotbrain
# -------------------------------------------------
fig = plt.figure(figsize=(12, 8))

views = ["S", "I", "A", "P"]  # you can change these
titles = ["Superior", "Inferior", "Anterior", "Posterior"]

for i, (view, title) in enumerate(zip(views, titles), start=1):
    ax = fig.add_subplot(2, 2, i, projection="3d")
    netplotbrain.plot(
        nodes=nodes[["x", "y", "z"]],
        nodes_df=nodes,
        node_color="color",
        node_size="size",
        template="MNI152NLin2009cAsym",
        template_style="surface",
        template_alpha=0.12,
        node_alpha=1.0,
        view=view,
        fig=fig,
        ax=ax,
        title=title,
        arrowaxis=None,
    )

plt.suptitle("BASC-122 regions shown as centroids", fontsize=18, fontweight="bold")
plt.tight_layout()
plt.savefig("BASC122_netplotbrain_centroids.png", dpi=600, bbox_inches="tight")
plt.show()

# In[7]:

# Code source: Dan Gale
# License: BSD 3 clause

from surfplot import Plot
from surfplot.datasets import load_example_data
from neuromaps.datasets import fetch_fslr

surfaces = fetch_fslr()
lh, rh = surfaces['inflated']

p = Plot(lh, rh)

# shading
lh_sulc, rh_sulc = surfaces['sulc']
p.add_layer({'left': lh_sulc, 'right': rh_sulc}, cmap='binary_r', cbar=False)

color_range = (0, 12)

# add default mode association stats
default = load_example_data(join=True)
p.add_layer(default, cmap='Blues_r', color_range=color_range,
            cbar_label='Default mode')

# add frontoparietal assocation stats
fronto = load_example_data('frontoparietal', join=True)
p.add_layer(fronto, cmap='Greens_r', color_range=color_range,
            cbar_label='Frontoparietal')

# create a clean looking set of colorbars. Only show labels for outer colorbar,
# given that both colorbars have the same range.
cbar_kws = dict(outer_labels_only=True, pad=.02, n_ticks=2, decimals=0)
fig = p.build(cbar_kws=cbar_kws)
# add units to colorbar
fig.axes[1].set_xlabel('z', labelpad=-11, fontstyle='italic')
fig.show()

# In[20]:

import numpy as np

left_labels = np.unique(surf_left.astype(int))
right_labels = np.unique(surf_right.astype(int))

# remove background 0
left_labels = left_labels[left_labels > 0]
right_labels = right_labels[right_labels > 0]

all_surface_labels = np.union1d(left_labels, right_labels)

print("Number of labels on left surface:", len(left_labels))
print("Left labels:", left_labels)

print("Number of labels on right surface:", len(right_labels))
print("Right labels:", right_labels)

print("Total unique labels across both hemispheres:", len(all_surface_labels))
print("All surface labels:", all_surface_labels)

# In[21]:

atlas_data = nib.load(basc_122).get_fdata().astype(int)
volume_labels = np.unique(atlas_data)
volume_labels = volume_labels[volume_labels > 0]

print("Labels in volume:", len(volume_labels))
print(volume_labels)

# In[22]:

missing_labels = np.setdiff1d(volume_labels, all_surface_labels)
print("Missing after projection:", missing_labels)
print("Number missing:", len(missing_labels))

# In[23]:

import numpy as np
import nibabel as nib
from nilearn import plotting, image, datasets

# Load atlas
atlas = datasets.fetch_atlas_basc_multiscale_2015(
    resolution=122, version="sym"
)
atlas_img = nib.load(atlas["maps"])
atlas_data = atlas_img.get_fdata()

# Missing labels
missing_labels = [9, 10, 11, 14, 18, 46, 54, 62, 65, 90, 108, 121]

# Create mask of missing regions
missing_mask = np.isin(atlas_data, missing_labels).astype(int)
missing_img = image.new_img_like(atlas_img, missing_mask)

# Plot
plotting.plot_roi(
    missing_img,
    title="BASC-122 Regions Not Represented on the Cortical Surface",
    cmap="Reds",
    draw_cross=False,
    black_bg=False
)

plotting.show()

# In[11]:

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from nilearn import datasets, plotting, surface

# -------------------------------------------------
# 1. Fetch BASC 122 atlas
# -------------------------------------------------
atlas = datasets.fetch_atlas_basc_multiscale_2015(
    resolution=122,
    version="sym"
)
basc_122 = atlas["maps"]

# -------------------------------------------------
# 2. Fetch fsaverage inflated surface
# -------------------------------------------------
fsaverage = datasets.fetch_surf_fsaverage()

# -------------------------------------------------
# 3. Project atlas to surface
#    Use nearest_most_frequent for label atlases
# -------------------------------------------------
texture_left = surface.vol_to_surf(
    basc_122,
    fsaverage.pial_left,
    interpolation="nearest_most_frequent"
)

texture_right = surface.vol_to_surf(
    basc_122,
    fsaverage.pial_right,
    interpolation="nearest_most_frequent"
)

# -------------------------------------------------
# 4. Build a large categorical colormap
#    Background (0) = transparent/white-like
# -------------------------------------------------
n_labels = 123  # 0 to 122
base = plt.cm.nipy_spectral(np.linspace(0, 1, n_labels))
base[0] = [1, 1, 1, 0]  # background transparent
parcel_cmap = ListedColormap(base)

# -------------------------------------------------
# 5. Create a publication-style multi-panel layout
# -------------------------------------------------
fig = plt.figure(figsize=(14, 8), facecolor="white")

gs = fig.add_gridspec(
    2, 2,
    left=0.02, right=0.98,
    bottom=0.04, top=0.93,
    wspace=0.02, hspace=0.08
)

axes = [
    fig.add_subplot(gs[0, 0], projection="3d"),
    fig.add_subplot(gs[0, 1], projection="3d"),
    fig.add_subplot(gs[1, 0], projection="3d"),
    fig.add_subplot(gs[1, 1], projection="3d"),
]

panels = [
    ("left",  "lateral", fsaverage.infl_left,  fsaverage.sulc_left,  texture_left,  "Left lateral"),
    ("right", "lateral", fsaverage.infl_right, fsaverage.sulc_right, texture_right, "Right lateral"),
    ("left",  "medial",  fsaverage.infl_left,  fsaverage.sulc_left,  texture_left,  "Left medial"),
    ("right", "medial",  fsaverage.infl_right, fsaverage.sulc_right, texture_right, "Right medial"),
]

for ax, (hemi, view, mesh, sulc, texture, title) in zip(axes, panels):
    plotting.plot_surf_roi(
        surf_mesh=mesh,
        roi_map=texture,
        hemi=hemi,
        view=view,
        bg_map=sulc,
        bg_on_data=True,
        darkness=0.65,       # stronger gray background, closer to your example
        cmap=parcel_cmap,
        colorbar=False,
        axes=ax,
        title=title
    )

# Optional main title
fig.suptitle("BASC-122 functional parcellation", fontsize=18, fontweight="bold")

plt.savefig(
    "BASC122_high_impact_style.png",
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)
plt.show()

# In[ ]:



# In[12]:

import numpy as np
from nilearn import image, surface, plotting, datasets
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

atlas = datasets.fetch_atlas_basc_multiscale_2015(resolution=122, version="sym")
basc_122 = atlas["maps"]
fsaverage = datasets.fetch_surf_fsaverage()

# Example: keep only labels 10, 25, 48
img = image.load_img(basc_122)
data = img.get_fdata()

keep = [10, 25, 48]
masked = np.where(np.isin(data, keep), data, 0)

masked_img = image.new_img_like(img, masked)

texture_left = surface.vol_to_surf(
    masked_img, fsaverage.pial_left, interpolation="nearest_most_frequent"
)
texture_right = surface.vol_to_surf(
    masked_img, fsaverage.pial_right, interpolation="nearest_most_frequent"
)

# Simple 4-color map: background transparent + highlighted parcels
colors = np.array([
    [1, 1, 1, 0],      # background
    [0.85, 0.15, 0.15, 1],
    [0.15, 0.35, 0.85, 1],
    [0.15, 0.70, 0.30, 1],
    [0.80, 0.50, 0.10, 1],
])
cmap = ListedColormap(colors)

fig, axes = plt.subplots(1, 2, figsize=(10, 4), subplot_kw={"projection": "3d"})

plotting.plot_surf_roi(
    fsaverage.infl_left, texture_left,
    hemi="left", view="lateral",
    bg_map=fsaverage.sulc_left,
    bg_on_data=True, darkness=0.75,
    cmap=cmap, colorbar=False, axes=axes[0]
)

plotting.plot_surf_roi(
    fsaverage.infl_right, texture_right,
    hemi="right", view="lateral",
    bg_map=fsaverage.sulc_right,
    bg_on_data=True, darkness=0.75,
    cmap=cmap, colorbar=False, axes=axes[1]
)

plt.savefig("BASC122_selected_regions.png", dpi=600, bbox_inches="tight")
plt.show()

# In[ ]:
