"""cellplot — the notebook plotting convention (PROTOCOL: every cell ends with a representative
plot; glanceable without reading the code; inline AND saved to the notebook's own plots/ folder).

Libraries: plotly (time series / equity / interactive forms) + seaborn (statistical forms), both
styled from ONE validated palette (dataviz-skill reference instance, light mode — saved PNGs are
viewed on white). Inline rendering uses static PNG (no JS dependency, glanceable in any viewer).

Usage in a notebook (cwd = the notebook's folder, e.g. notebooks/overnight_rf/):
    import sys; sys.path.insert(0, r"C:\\Users\\User\\backtest_engine\\backtest_engine2")
    from cellplot import cellplot, PAL, GRAY, DIV, setup; setup()
    ...
    cellplot(fig, 4, "val_tournament")        # -> inline + plots/cell_04_val_tournament.png
"""
import os

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# dataviz reference palette (light mode) — fixed categorical order, never cycled/re-sorted
PAL = ["#2a78d6", "#008300", "#e87ba4", "#eda100", "#1baf7a", "#eb6834", "#4a3aa7", "#e34948"]
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
GRID = "#e5e4e0"
GRAY = "#9b9a94"                       # comparison/baseline series (non-categorical role)
DIV = ["#2a78d6", "#f0efec", "#e34948"]   # diverging: blue <- neutral gray -> red

PLOTS_DIR = "plots"


def setup():
    sns.set_theme(style="whitegrid", palette=PAL)
    matplotlib.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": TEXT, "axes.labelcolor": TEXT2, "xtick.color": TEXT2, "ytick.color": TEXT2,
        "axes.edgecolor": GRID, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 12, "axes.titleweight": "bold", "figure.dpi": 110,
        "lines.linewidth": 1.6, "font.size": 10,
    })
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
        tpl = go.layout.Template()
        tpl.layout = dict(colorway=PAL, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
                          font=dict(color=TEXT, size=12),
                          xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID),
                          yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID),
                          margin=dict(l=60, r=30, t=60, b=45), showlegend=True)
        pio.templates["cellplot"] = tpl
        pio.templates.default = "cellplot"
    except Exception:
        pass


def cellplot(fig, cell, slug, width=980, height=430):
    """Render inline (static) AND save plots/cell_NN_slug.png. Accepts a plotly Figure or a
    matplotlib Figure/Axes. Returns the saved path."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, f"cell_{cell:02d}_{slug}.png")
    if hasattr(fig, "write_image"):                       # plotly
        # titles don't wrap in plotly — break long ones so they never overflow the canvas
        t = fig.layout.title.text if fig.layout.title else None
        if t and len(t) > 88 and "<br>" not in t:
            cut = t.rfind("; ", 0, 88)
            cut = cut if cut > 40 else t.rfind(" ", 0, 88)
            if cut > 40:
                fig.update_layout(title_text=t[:cut] + "<br><sup>" + t[cut:].lstrip("; ") + "</sup>")
        fig.update_layout(title_font_size=15, title_x=0.02, title_xanchor="left")
        fig.write_image(path, width=width, height=height, scale=2)
        try:
            from IPython.display import Image, display
            display(Image(filename=path))
        except Exception:
            pass
    else:                                                 # matplotlib/seaborn
        f = fig.figure if hasattr(fig, "figure") and not hasattr(fig, "savefig") else fig
        f.savefig(path, bbox_inches="tight")
        plt.show()
        plt.close(f)
    print(f"[plot] {path}")
    return path
