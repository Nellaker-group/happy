from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Cell:
    label: str
    name: str
    colour: str
    id: int
    structural_id: int

    def __str__(self):
        return f"{self.label}"


@dataclass(frozen=True)
class Tissue:
    label: str
    name: str
    colour: str
    id: int

    def __str__(self):
        return f"{self.label}"


class Organ:
    def __init__(self, cells: List[Cell], tissues: List[Tissue]):
        self.cells = cells
        self.tissues = tissues

    def cell_by_id(self, i: int):
        return self.cells[i]

    def cell_by_label(self, label):
        labels = {cell.label: cell.id for cell in self.cells}
        return self.cells[labels[label]]

    def tissue_by_label(self, label):
        labels = {tissue.label: tissue.id for tissue in self.tissues}
        return self.tissues[labels[label]]


PLACENTA = Organ(
    [
        Cell("CYT", "Cytotrophoblast", "#00E307", 0, 2),
        Cell("FIB", "Fibroblast", "#C80B2A", 1, 4),
        Cell("HOF", "Hofbauer Cell", "#FFDC3D", 2, 8),
        Cell("SYN", "Syncytiotrophoblast", "#009FFA", 3, 0),
        Cell("VEN", "Vascular Endothelial", "#FF6E3A", 4, 6),
        Cell("MAT", "Maternal Decidua", "#008169", 5, 10),
        Cell("VMY", "Vascular Myocyte", "#6A0213", 6, 5),
        Cell("WBC", "Leukocyte", "#003C86", 7, 9),
        Cell("MES", "Mesenchymal Cell", "#FF71FD", 8, 7),
        Cell("EVT", "Extra Villus Trophoblast", "#FFCFE2", 9, 3),
        Cell("KNT", "Syncytial Knot", "#7CFFFA", 10, 1),
    ],
    [
        Tissue("Unlabelled", "Unlabelled", "#000000", 0),
        Tissue("Sprout", "Villus Sprout", "#ff3cfe", 1),
        Tissue("MVilli", "Mesenchymal Villi", "#f60239", 2),
        Tissue("TVilli", "Terminal Villi", "#ff6e3a", 3),
        Tissue("ImIVilli", "Immature Intermediate Villi", "#5a000f", 4),
        Tissue("MIVilli", "Mature Intermediate Villi", "#ffac3b", 5),
        Tissue("AVilli", "Anchoring Villi", "#ffcfe2", 6),
        Tissue("SVilli", "Stem Villi", "#ffdc3d", 7),
        Tissue("Chorion", "Chorionic Plate", "#005a01", 8),
        Tissue("Maternal", "Basal Plate/Septum", "#00cba7", 9),
        Tissue("Inflam", "Inflammatory Response", "#7cfffa", 10),
        Tissue("Fibrin", "Fibrin", "#0079fa", 11),
        Tissue("Avascular", "Avascular Villi", "#450270", 12),
    ],
)
PLACENTA_CORD = Organ(
    [
        Cell("EPI", "Epithelial Cell", "#ff0000", 0, 0),
        Cell("FIB", "Fibroblast", "#7b03fc", 1, 1),
        Cell("MAC", "Macrophage", "#979903", 2, 2),
        Cell("VEN", "Vascular Endothelial", "#734c0e", 3, 3),
        Cell("VMY", "Vascular Myocyte", "#cc6633", 4, 4),
        Cell("WBC", "White Blood Cell", "#2f3ec7", 5, 5),
        Cell("MES", "Mesenchymal Cell", "#ff00ff", 6, 6),
    ],
    [],
)


def get_organ(organ_name):
    organ_dicts = {
        "placenta": PLACENTA,
        "placenta_cord": PLACENTA_CORD,
    }
    return organ_dicts[organ_name]
