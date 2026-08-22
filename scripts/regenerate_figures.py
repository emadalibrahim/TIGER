from pathlib import Path
import sys

repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo))

from src.tiger_model import TigerModel

if __name__ == "__main__":
    model = TigerModel(repo)
    model.plot_lce()
    model.plot_tco()
    model.plot_lce_vs_tco()
    model.plot_avoided_cost()
    model.plot_contours()
    model.plot_tornado()
    model.plot_fleet()
    model.plot_charging_infrastructure()
