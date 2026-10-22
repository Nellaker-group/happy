INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
SURFACE = "#ffffff"
NEUTRAL_FILL = "#e4e3df"
BAND = "#d9d8d4"
LINE = "#2a78d6"

# Colour-blind safe categorical colours in a fixed order
STRUCTURE_COLOURS = {
    "UVein": "#2a78d6",
    "UArtery": "#eb6834",
    "Allantois": "#1baf7a",
    "VitDuct": "#eda100",
    "VitVes": "#e87ba4",
}
STRUCTURE_NAMES = {
    "UVein": "Vein",
    "UArtery": "Artery",
    "Allantois": "Allantoic duct",
    "VitDuct": "Vitelline duct",
    "VitVes": "Vitelline vessel",
}
STRUCTURE_SHORT = {"UVein": "V", "UArtery": "A", "Allantois": "Al", "VitDuct": "VD", "VitVes": "VV"}


def style_axes(ax):
    """Recessive axes: no top or right spine, light solid gridlines behind the data."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
