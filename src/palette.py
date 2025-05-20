# Consistent color palette (color-blind-friendly, scientific)
color_palette = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
]

# Broad cell type color map
label_colors = {
    "GABAergic": color_palette[0],  # blue
    "Glutamatergic": color_palette[1],      # orange
    "Other": color_palette[2],          # green
}

# Optional: Apply to matplotlib globally (in notebooks)
def set_global_palette():
    import matplotlib.pyplot as plt
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=color_palette)
