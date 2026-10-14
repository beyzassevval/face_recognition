"""
Grafiklerde ortak stil ve renkler.

Renk kişiye sabit bağlıdır (sıralamaya göre değil): aynı kişi tüm grafiklerde aynı
renk + aynı marker şeklindedir. İlk 4 renk, dağılım grafiklerinde tüm çiftler için
renk körlüğü (CVD) ayrımını geçen doğrulanmış bir palettir; 5. kişi nötr koyu gri
alır. Her kişinin ayrıca farklı bir marker şekli vardır, yani kimlik yalnızca renge
bağlı değildir. LFW (distractor) kişileri içi boş gri marker ile çizilir.
"""
import matplotlib.pyplot as plt

OWN_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#3d3c39"]
OWN_MARKERS = ["o", "s", "^", "D", "P", "v", "X", "*"]
LFW_COLOR = "#8a8984"
LFW_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">", "p"]

CONFIG_COLORS = {
    "A_scrfd_arcface": "#2a78d6",
    "B_mtcnn_facenet": "#eb6834",
    "C_scrfd_facenet": "#1baf7a",
}


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "savefig.facecolor": "#fcfcfb",
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": "#0b0b0b",
        "axes.grid": True,
        "grid.color": "#e8e7e1",
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": "#52514e",
        "ytick.color": "#52514e",
        "font.size": 9,
    })
