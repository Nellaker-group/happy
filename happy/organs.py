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
CORD = Organ(
    [
        Cell("EPI", "Epithelial Cell", "#332288", 0, 4),
        Cell("VEN", "Vascular Endothelial", "#88CCEE", 1, 2),
        Cell("MES", "Mesenchymal Cell", "#882255", 2, 1),
        Cell("FIB", "Fibroblast", "#AA4499", 3, 1),
        Cell("MFB", "Myofibroblast", "#CC6677", 4, 1),
        Cell("VMY", "Vascular Myocyte", "#661100", 5, 1),
        Cell("WBC", "Leukocyte", "#DDCC77", 6, 3),
    ],
    [
        Tissue("Unlabelled", "Unlabelled", "#bab0ac", 0),
        Tissue("AmnEpi", "Amniotic Epithelium", "#4e79a7", 1),
        Tissue("SWJelly", "Subamniotic Wharton's Jelly", "#b07aa1", 2),
        Tissue("IWJelly", "Intermediate Wharton's Jelly", "#ff9da7", 3),
        Tissue("PWJelly", "Perivascular Wharton's Jelly", "#76b7b2", 4),
        Tissue("UArtery", "Umbilical Artery", "#edc948", 5),
        Tissue("UVein", "Umbilical Vein", "#59a14f", 6),
        Tissue("Allantois", "Allantoic Duct Remnant", "#9c755f", 7),
        Tissue("VitDuct", "Vitelline Duct Remnant", "#f28e2b", 8),
        Tissue("VitVes", "Vitelline Vessel Remnant", "#e15759", 9),
    ],
)
MEMBRANE = Organ(
    [
        Cell("EPI", "Epithelial Cell", "#332288", 0, 2),
        Cell("VEN", "Vascular Endothelial", "#88CCEE", 1, 2),
        Cell("MES", "Mesenchymal Cell", "#882255", 2, 1),
        Cell("FIB", "Fibroblast", "#AA4499", 3, 1),
        Cell("MFB", "Myofibroblast", "#CC6677", 4, 1),
        Cell("VMY", "Vascular Myocyte", "#661100", 5, 1),
        Cell("CYT", "Cytotrophoblast", "#999933", 6, 0),
        Cell("EVT", "Extra Villus Trophoblast", "#117733", 7, 0),
        Cell("MAT", "Maternal Decidua", "#44AA99", 8, 3),
        Cell("WBC", "Leukocyte", "#DDCC77", 9, 4),
    ],
    [
        Tissue("Unlabelled", "Unlabelled", "#000000", 0),
        Tissue("Amnion", "Amnion", "#CC79A7", 1),
        Tissue("CMembrane", "Chorion Laeve", "#56B4E9", 2),
        Tissue("RVilli", "Villous Remnant", "#0072B2", 3),
        Tissue("Decidua", "Decidua Capsularis", "#E69F00", 4),
        Tissue("SArtery", "Spiral Artery", "#D55E00", 5),
    ],
)



def get_organ(organ_name):
    organ_dicts = {
        "placenta": PLACENTA,
        "cord": CORD,
        "membrane": MEMBRANE,
    }
    return organ_dicts[organ_name]
