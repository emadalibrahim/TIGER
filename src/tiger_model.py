from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, ConnectionPatch
from matplotlib.colors import ListedColormap, BoundaryNorm


VEHICLE_FILES = {
    "trucks": "trucks.json",
    "cars": "cars.json",
    "trucks_light": "trucks_light.json",
    "bus": "bus.json",
    "scooters": "scooters.json",
}

VEHICLE_LABELS = {
    "trucks": "Trucks",
    "cars": "Cars",
    "trucks_light": "Light Trucks",
    "bus": "Buses",
    "scooters": "Scooters",
}

BATTERY_KWH = {
    ("trucks", "electric"): 732.0,
    ("trucks", "hydrogen"): 285.0,
    ("cars", "hybrid"): 1.6,
    ("cars", "phev"): 17.0,
    ("cars", "electric"): 26.0,
    ("trucks_light", "electric"): 31.0,
    ("bus", "electric"): 324.0,
}

# Vehicle manufacturing + battery replacement emissions, kg CO2e per vehicle.
# These preserve the manuscript boundary and values while making the replacement explicit.
VEHICLE_PRODUCTION_KG = {
    ("trucks", "diesel"): 56110.0,
    ("trucks", "electric"): 168400.0 * 0.33 + 732.0 * 61.0,
    ("trucks", "hydrogen"): 70150.0,
    ("cars", "gasoline"): 8000.0,
    ("cars", "hybrid"): 10400.0,
    ("cars", "phev"): 12000.0,
    ("cars", "electric"): 13600.0,
    ("trucks_light", "gasoline"): 9980.0,
    ("trucks_light", "electric"): 16170.0,
    ("bus", "diesel"): 37459.0,
    ("bus", "electric"): 50297.0,
    ("scooters", "gasoline"): 260.0,
    ("scooters", "electric"): 401.0,
}

BATTERY_REPLACEMENT_EMISSIONS_KG = {
    ("trucks", "diesel"): 0.0,
    ("trucks", "electric"): 732.0 * 61.0,
    ("trucks", "hydrogen"): 285.0 * 61.0,
    ("cars", "gasoline"): 0.0,
    ("cars", "hybrid"): 1.6 * 61.0,
    ("cars", "phev"): 17.0 * 61.0,
    ("cars", "electric"): 26.0 * 61.0,
    ("trucks_light", "gasoline"): 0.0,
    ("trucks_light", "electric"): 31.0 * 61.0,
    ("bus", "diesel"): 0.0,
    ("bus", "electric"): 324.0 * 69.0,
    ("scooters", "gasoline"): 0.0,
    ("scooters", "electric"): 79.3,
}

FLEET_SIZE = {
    "scooters": 14_611_708,
    "cars": 7_122_350,
    "trucks_light": 941_000,
    "bus": 30_127,
    "trucks": 167_088,
}

ANNUAL_SALES = {
    "scooters": 695_670,
    "cars": 457_837,
    "trucks_light": 71_380,
    "bus": 1_800,
    "trucks": 9_700,
}

# Matches the revised Methods section, including the corrected light-truck value.
FLEET_VKT = {
    "scooters": 6_400,
    "cars": 12_179,
    "trucks_light": 11_509,
    "bus": 58_793,
    "trucks": 33_388,
}

OBSERVED_EV = {
    "scooters": {2015: 52010, 2016: 71846, 2017: 114013, 2018: 194633, 2019: 359934, 2020: 455764, 2021: 546438, 2022: 630223, 2023: 703879, 2024: 774651},
    "cars": {2015: 240, 2016: 286, 2017: 779, 2018: 1152, 2019: 3597, 2020: 8487, 2021: 14290, 2022: 28062, 2023: 49149, 2024: 81740},
    "trucks_light": {2015: 16, 2016: 7, 2017: 5, 2018: 9, 2019: 6, 2020: 13, 2021: 77, 2022: 89, 2023: 180, 2024: 601},
    "bus": {2015: 168, 2016: 249, 2017: 313, 2018: 514, 2019: 521, 2020: 612, 2021: 800, 2022: 1222, 2023: 1883, 2024: 1994},
    "trucks": {2015: 0, 2016: 0, 2017: 0, 2018: 0, 2019: 0, 2020: 0, 2021: 0, 2022: 1, 2023: 3, 2024: 9},
}

SALES_SHARE_ANCHORS = {
    "scooters": {2024: 0.07, 2030: 0.35, 2035: 0.70, 2040: 1.00, 2050: 1.00},
    "cars": {2024: 0.06, 2030: 0.30, 2035: 0.60, 2040: 1.00, 2050: 1.00},
    "trucks_light": {2024: 0.01, 2035: 0.30, 2040: 0.60, 2050: 1.00},
    "bus": {2024: 0.20, 2030: 1.00, 2050: 1.00},
    "trucks": {2024: 0.00, 2040: 0.30, 2050: 0.60},
}


@dataclass
class TCOResult:
    fuel: np.ndarray
    labor: np.ndarray
    capital: np.ndarray
    other_opex: np.ndarray
    total: np.ndarray


@dataclass
class LCEResult:
    vehicle: np.ndarray
    fuel_production: np.ndarray
    fuel_use: np.ndarray
    total: np.ndarray


class TigerModel:
    """Transport cost, LCA, scenario, and fleet model.
    """

    def __init__(self, repo_root: str | Path):
        self.root = Path(repo_root).resolve()
        self.data_dir = self.root / "data"
        self.fig_dirs = [self.root / "figs", self.root / "figures"]
        for d in self.fig_dirs:
            d.mkdir(parents=True, exist_ok=True)

        with open(self.data_dir / "scenario_config.json") as f:
            self.cfg = json.load(f)
        self.factors = pd.read_csv(self.data_dir / "technology_factors.csv")
        self.atb_vehicle_battery_costs = pd.read_csv(self.data_dir / "atb_selected_vehicle_battery_costs.csv")
        self.phev_mode_factors = pd.read_csv(self.data_dir / "phev_atb_mode_efficiency.csv")
        self.scooter_nmc_costs = pd.read_csv(self.data_dir / "scooter_nmc_pack_cost_trajectory.csv")
        self.vehicles: Dict[str, dict] = {}
        for name, filename in VEHICLE_FILES.items():
            with open(self.data_dir / filename) as f:
                self.vehicles[name] = json.load(f)

        self.lifetime = int(self.cfg["lifetime_years"])
        self.discount_rate = float(self.cfg["discount_rate"])
        self.depreciation_rate = float(self.cfg["depreciation_rate"])
        self.replacement_year = int(self.cfg["battery_replacement_year"])
        self.battery_cost_2022_to_2023 = float(self.cfg.get("battery_cost_2022_to_2023", 304.702 / 292.655))
        self.uf = float(self.cfg["phev_utility_factor"])
        self.fx = float(self.cfg["usd_to_ntd"])
        self.n_mc = int(self.cfg["monte_carlo_draws"])
        self.seed = int(self.cfg["monte_carlo_seed"])
        self.ec = self.cfg["energy_constants"]
        self.fuel_prices = self.cfg["fuel_prices_ntd"]

    # --------------------------- general helpers ---------------------------
    @staticmethod
    def _interp(anchors: dict, year: float) -> float:
        x = np.array(sorted(float(k) for k in anchors), dtype=float)
        y = np.array([float(anchors[str(int(k))] if str(int(k)) in anchors else anchors[int(k)]) for k in x], dtype=float)
        return float(np.interp(year, x, y))

    @staticmethod
    def _triangular(rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.triangular(0.0, 0.5, 1.0, n)

    def _bundle(self, vehicle: str, powertrain: str, u: np.ndarray | float) -> dict:
        lo = self.vehicles[vehicle][f"{powertrain}_low"]
        hi = self.vehicles[vehicle][f"{powertrain}_high"]
        out = {}
        for key in ["fuel_efficiency", "vehicle_cost", "labor_cost", "maintenance_and_repair_cost", "insurance_cost", "fuel_fee_rate", "subsidy"]:
            a = float(lo[key] or 0.0)
            b = float(hi[key] or 0.0)
            out[key] = a + (b - a) * u
        # VKT is treated as a class-specific activity parameter, not correlated with the vehicle bundle.
        out["VKT"] = float(FLEET_VKT.get(vehicle, lo["VKT"]))
        # Scooter battery-swapping cost is already encoded directly as a per-km range.
        if lo.get("fuel_cost") is not None and hi.get("fuel_cost") is not None:
            a = float(lo["fuel_cost"])
            b = float(hi["fuel_cost"])
            out["fuel_cost"] = a + (b - a) * u
        else:
            out["fuel_cost"] = None
        return out

    def factor_for_trajectory(self, trajectory: str, vehicle: str, powertrain: str, year: int, column: str) -> float:
        """Return a technology multiplier for an explicit trajectory label."""
        if trajectory == "Static" or year <= 2025:
            return 1.0
        sub = self.factors[
            (self.factors.scenario == trajectory)
            & (self.factors.vehicle == vehicle)
            & (self.factors.powertrain == powertrain)
        ].sort_values("year")
        if sub.empty:
            return 1.0
        return float(np.interp(year, sub.year.to_numpy(), sub[column].to_numpy()))

    def factor(self, scenario: str, vehicle: str, powertrain: str, year: int, column: str) -> float:
        """Return the technology multiplier for a manuscript scenario.

        Conservative freezes 2025 vehicle purchase cost and energy use. Reference
        uses the manuscript ATB Baseline trajectory (the raw 2024 ATB Conservative vehicle
        trajectory), and Optimistic uses ATB Advanced. Scooter rows remain literature-based
        but follow the same scenario mapping.
        """
        trajectory = self.cfg["technology_trajectory_map"][scenario]
        return self.factor_for_trajectory(trajectory, vehicle, powertrain, year, column)

    def phev_mode_energy_factor(self, scenario: str, year: int, mode: str) -> float:
        """PHEV-specific charge-sustaining or charge-depleting energy-use factor.

        The selected ATB record is the Midsize Gasoline Power Split PHEV with
        30-mile electric range. PHEV efficiency scenarios use the raw ATB
        Constant / Mid / Advanced mode-specific trajectories for study
        Conservative / Reference / Optimistic, respectively. Fuel-mode energy
        use scales inversely with charge-sustaining fuel economy (mi/gge),
        while electric-mode energy use scales directly with charge-depleting
        electricity consumption (Wh/mi). Each raw trajectory is normalized to
        its own 2025 value so the Taiwan-specific 2025 model inputs remain the
        anchor.
        """
        raw_scenario = self.cfg.get("phev_efficiency_trajectory_map", {
            "Conservative": "Constant",
            "Reference": "Mid",
            "Optimistic": "Advanced",
        })[scenario]
        if mode not in ("fuel", "electric"):
            raise ValueError("PHEV mode must be 'fuel' or 'electric'.")
        column = (
            "fuel_mode_energy_factor_vs_2025"
            if mode == "fuel"
            else "electric_mode_energy_factor_vs_2025"
        )
        sub = self.phev_mode_factors[
            self.phev_mode_factors.raw_atb_scenario == raw_scenario
        ].sort_values("year")
        if sub.empty:
            raise ValueError(f"No PHEV ATB mode-efficiency trajectory for {raw_scenario}")
        y = min(max(int(year), int(sub.year.min())), int(sub.year.max()))
        return float(np.interp(y, sub.year.to_numpy(dtype=float), sub[column].to_numpy(dtype=float)))

    def phev_mode_efficiencies(self, scenario: str, year: int, u_vehicle: np.ndarray | float = 0.5):
        """Return charge-sustaining and electric-mode PHEV energy use (kWh/km).

        The 2025 mode-specific energy-use anchors retain the existing Taiwan
        model inputs used for charge-sustaining (hybrid proxy) and electric
        (BEV proxy) operation. Only their future evolution is replaced by the
        PHEV-specific ATB CS/CD trajectories.
        """
        hev = self._bundle("cars", "hybrid", u_vehicle)
        bev = self._bundle("cars", "electric", u_vehicle)
        fuel_base = np.asarray(hev["fuel_efficiency"], dtype=float)
        electric_base = np.asarray(bev["fuel_efficiency"], dtype=float)
        fuel = fuel_base * self.phev_mode_energy_factor(scenario, year, "fuel")
        electric = electric_base * self.phev_mode_energy_factor(scenario, year, "electric")
        return fuel, electric

    def atb_vehicle_battery_pack_cost_2022usd_per_kwh(self, vehicle: str, powertrain: str, scenario: str, year: int) -> float:
        """Pack-level battery cost from the exact selected 2024 Transportation ATB vehicle.

        Battery replacement costs use the ATB battery-cost metric attached to the
        selected vehicle configuration, rather than a generic BEV/PHEV/HEV curve.
        Study scenarios map to raw ATB battery scenarios as:
        Conservative -> Conservative, Reference -> Mid, Optimistic -> Advanced.
        Values are linearly interpolated between ATB milestone years.
        """
        raw_scenario = {
            "Conservative": "Conservative",
            "Reference": "Mid",
            "Optimistic": "Advanced",
        }[scenario]
        df = self.atb_vehicle_battery_costs
        sub = df[(df.vehicle == vehicle) &
                 (df.powertrain == powertrain) &
                 (df.raw_atb_scenario == raw_scenario)].sort_values("year")
        if sub.empty:
            raise ValueError(f"No selected-vehicle ATB battery trajectory for {vehicle}/{powertrain}/{raw_scenario}")
        y = min(max(int(year), int(sub.year.min())), int(sub.year.max()))
        return float(np.interp(y,
                               sub.year.to_numpy(dtype=float),
                               sub.pack_cost_2022usd_per_kwh.to_numpy(dtype=float)))

    def scooter_nmc_pack_cost_usd_per_kwh(self, year: int) -> float:
        """NMC pack-cost trajectory used only for electric-scooter swapping costs."""
        sub = self.scooter_nmc_costs.sort_values("year")
        y = min(max(int(year), int(sub.year.min())), int(sub.year.max()))
        return float(np.interp(y, sub.year.to_numpy(dtype=float),
                               sub.pack_cost_usd_per_kwh.to_numpy(dtype=float)))

    def scooter_swap_cost_usd_per_km(self, base_cost_usd_per_km, scenario: str, year: int):
        """Update only the battery-capital component of the swapping service cost.

        The observed 2025 swapping fee is retained as the base. Conservative holds
        that fee constant. Reference and Optimistic replace the 2025 battery-capital
        contribution with the future NMC pack cost using an effective network inventory
        of 2.2 batteries per scooter, a 1.3-kWh battery, and an equivalent
        10-year/64,000-km battery service life. Non-battery service costs are therefore
        held constant instead of scaling the entire swapping fee.
        """
        base = np.asarray(base_cost_usd_per_km, dtype=float)
        if scenario == "Conservative" or year <= 2025:
            return base
        if scenario not in ("Reference", "Optimistic"):
            raise ValueError(scenario)
        cap_kwh = float(self.cfg["scooter_battery_capacity_kwh"])
        inventory = float(self.cfg["scooter_effective_battery_inventory_per_scooter"])
        lifetime_km = float(self.cfg["scooter_battery_lifetime_km"])
        c0 = self.scooter_nmc_pack_cost_usd_per_kwh(2025)
        ct = self.scooter_nmc_pack_cost_usd_per_kwh(min(int(year), 2050))
        return base + inventory * cap_kwh * (ct - c0) / lifetime_km

    def save_scooter_swapping_trajectory(self):
        """Write the scenario/year swapping-cost trajectory used by the TCO model."""
        lo = float(self._bundle("scooters", "electric", 0.0)["fuel_cost"])
        hi = float(self._bundle("scooters", "electric", 1.0)["fuel_cost"])
        rows = []
        c2025 = self.scooter_nmc_pack_cost_usd_per_kwh(2025)
        for scenario in ("Conservative", "Reference", "Optimistic"):
            for year in range(2025, 2051):
                c = self.scooter_nmc_pack_cost_usd_per_kwh(year)
                low = float(self.scooter_swap_cost_usd_per_km(lo, scenario, year))
                high = float(self.scooter_swap_cost_usd_per_km(hi, scenario, year))
                rows.append({
                    "scenario": scenario,
                    "year": year,
                    "nmc_pack_cost_usd_per_kwh": c,
                    "nmc_pack_cost_change_vs_2025_usd_per_kwh": c - c2025,
                    "effective_battery_inventory_per_scooter": float(self.cfg["scooter_effective_battery_inventory_per_scooter"]),
                    "battery_capacity_kwh": float(self.cfg["scooter_battery_capacity_kwh"]),
                    "battery_lifetime_km": float(self.cfg["scooter_battery_lifetime_km"]),
                    "swap_cost_low_usd_per_km": low,
                    "swap_cost_high_usd_per_km": high,
                    "swap_cost_mid_usd_per_km": 0.5 * (low + high),
                })
        self._save_table(pd.DataFrame(rows), "scooter_battery_swapping_cost_trajectory.csv")

    def grid_ci(self, scenario: str, year: int) -> float:
        anchors = self.cfg["grid_ci_kgco2_per_kwh"][scenario]
        return self._interp(anchors, year)

    def h2_ci(self, pathway: str, year: int) -> float:
        return self._interp(self.cfg["hydrogen_ci_kgco2_per_kg"][pathway], year)

    def h2_dispensing(self, scenario: str, year: int) -> float:
        return self._interp(self.cfg["hydrogen_dispensing_ntd_per_kg"][scenario], year)

    def h2_production_mid(self, pathway: str) -> float:
        lo, hi = self.cfg["hydrogen_production_ntd_per_kg"][pathway]
        return 0.5 * (float(lo) + float(hi))

    # ------------------------------- LCA -----------------------------------
    def _vehicle_cycle_g_per_km(self, vehicle: str, powertrain: str) -> float:
        kg = VEHICLE_PRODUCTION_KG[(vehicle, powertrain)] + BATTERY_REPLACEMENT_EMISSIONS_KG[(vehicle, powertrain)]
        vkt = float(FLEET_VKT[vehicle])
        return kg * 1000.0 / (self.lifetime * vkt)

    def lce(self, vehicle: str, powertrain: str, year: int = 2025, scenario: str = "Reference",
            pathway: Optional[str] = None, u: np.ndarray | float = 0.5, utility_factor: Optional[float] = None) -> LCEResult:
        uf = self.uf if utility_factor is None else utility_factor
        b = self._bundle(vehicle, powertrain, u)
        eff_factor = self.factor(scenario, vehicle, powertrain, year, "energy_use_factor")
        eff = np.asarray(b["fuel_efficiency"], dtype=float) * eff_factor
        vehicle_g = np.asarray(self._vehicle_cycle_g_per_km(vehicle, powertrain)) + np.zeros_like(eff)

        if vehicle == "cars" and powertrain == "phev":
            fuel_eff, electric_eff = self.phev_mode_efficiencies(scenario, year, u)
            liters = (1.0 - uf) * fuel_eff / float(self.ec["gasoline_kwh_per_l"])
            fuel_prod = (
                liters * float(self.ec["gasoline_upstream_kgco2_per_l"]) * 1000.0
                + uf * electric_eff * self.grid_ci(scenario, year) * 1000.0
            )
            fuel_use = liters * (float(self.ec["gasoline_ci_kgco2_per_l"]) - float(self.ec["gasoline_upstream_kgco2_per_l"])) * 1000.0
        elif powertrain in ("gasoline", "hybrid"):
            liters = eff / float(self.ec["gasoline_kwh_per_l"])
            fuel_prod = liters * float(self.ec["gasoline_upstream_kgco2_per_l"]) * 1000.0
            fuel_use = liters * (float(self.ec["gasoline_ci_kgco2_per_l"]) - float(self.ec["gasoline_upstream_kgco2_per_l"])) * 1000.0
        elif powertrain == "diesel":
            liters = eff / float(self.ec["diesel_kwh_per_l"])
            fuel_prod = liters * float(self.ec["diesel_upstream_kgco2_per_l"]) * 1000.0
            fuel_use = liters * (float(self.ec["diesel_ci_kgco2_per_l"]) - float(self.ec["diesel_upstream_kgco2_per_l"])) * 1000.0
        elif powertrain == "electric":
            fuel_prod = eff * self.grid_ci(scenario, year) * 1000.0
            fuel_use = np.zeros_like(eff)
        elif powertrain == "hydrogen":
            if pathway not in ("blue", "green"):
                raise ValueError("Hydrogen pathway must be 'blue' or 'green'.")
            fuel_prod = eff / float(self.ec["hydrogen_kwh_per_kg"]) * self.h2_ci(pathway, year) * 1000.0
            fuel_use = np.zeros_like(eff)
        else:
            raise ValueError((vehicle, powertrain))

        total = vehicle_g + fuel_prod + fuel_use
        return LCEResult(vehicle=vehicle_g, fuel_production=fuel_prod, fuel_use=fuel_use, total=total)

    # ------------------------------- TCO -----------------------------------
    def _replacement_cost_usd(self, vehicle: str, powertrain: str, year: int, scenario: str) -> float:
        """Battery replacement cost using the exact selected ATB vehicle trajectory.

        One replacement occurs in ownership year 10. The model retains the
        Taiwan/TIGER battery capacity for each vehicle, but values that capacity
        using the pack-level 2022 USD/kWh battery cost attached to the selected
        2024 Transportation ATB vehicle configuration. Study Conservative,
        Reference, and Optimistic use raw ATB Conservative, Mid, and Advanced
        battery trajectories, respectively.
        """
        kwh = BATTERY_KWH.get((vehicle, powertrain), 0.0)
        if kwh == 0.0:
            return 0.0
        replacement_calendar_year = min(int(year) + self.replacement_year, 2050)
        pack_2022 = self.atb_vehicle_battery_pack_cost_2022usd_per_kwh(
            vehicle, powertrain, scenario, replacement_calendar_year
        )
        return kwh * pack_2022 * self.battery_cost_2022_to_2023

    def _energy_cost_usd_per_km(self, vehicle: str, powertrain: str, year: int, scenario: str,
                                b: dict, u_vehicle: np.ndarray | float, u_price: np.ndarray | float,
                                pathway: Optional[str], utility_factor: float) -> np.ndarray:
        eff = np.asarray(b["fuel_efficiency"], dtype=float) * self.factor(scenario, vehicle, powertrain, year, "energy_use_factor")
        if vehicle == "scooters" and powertrain == "electric" and b["fuel_cost"] is not None:
            # Battery swapping remains a per-km service cost, but its future
            # trajectory follows battery-pack cost learning.
            return self.scooter_swap_cost_usd_per_km(b["fuel_cost"], scenario, year)
        if powertrain in ("gasoline", "hybrid"):
            lo, hi = self.fuel_prices["gasoline_per_l"]
            p = (float(lo) + (float(hi) - float(lo)) * u_price) / self.fx
            return eff / float(self.ec["gasoline_kwh_per_l"]) * p
        if powertrain == "diesel":
            lo, hi = self.fuel_prices["diesel_per_l"]
            p = (float(lo) + (float(hi) - float(lo)) * u_price) / self.fx
            return eff / float(self.ec["diesel_kwh_per_l"]) * p
        if powertrain == "electric":
            lo, hi = self.fuel_prices["electricity_per_kwh"]
            p = (float(lo) + (float(hi) - float(lo)) * u_price) / self.fx
            return eff * p
        if vehicle == "cars" and powertrain == "phev":
            fuel_eff, electric_eff = self.phev_mode_efficiencies(scenario, year, u_vehicle)
            gl, gh = self.fuel_prices["gasoline_per_l"]
            el, eh = self.fuel_prices["electricity_per_kwh"]
            gp = (float(gl) + (float(gh) - float(gl)) * u_price) / self.fx
            ep = (float(el) + (float(eh) - float(el)) * u_price) / self.fx
            return (
                (1.0 - utility_factor) * fuel_eff / float(self.ec["gasoline_kwh_per_l"]) * gp
                + utility_factor * electric_eff * ep
            )
        if powertrain == "hydrogen":
            if pathway not in ("blue", "green"):
                raise ValueError("Hydrogen pathway must be 'blue' or 'green'.")
            lo, hi = self.cfg["hydrogen_production_ntd_per_kg"][pathway]
            prod = float(lo) + (float(hi) - float(lo)) * u_price
            delivered = prod + self.h2_dispensing(scenario, year)
            return eff / float(self.ec["hydrogen_kwh_per_kg"]) * delivered / self.fx
        raise ValueError((vehicle, powertrain))

    def tco(self, vehicle: str, powertrain: str, year: int = 2025, scenario: str = "Reference",
            pathway: Optional[str] = None, u_vehicle: np.ndarray | float = 0.5,
            u_price: np.ndarray | float = 0.5, utility_factor: Optional[float] = None) -> TCOResult:
        uf = self.uf if utility_factor is None else utility_factor
        b = self._bundle(vehicle, powertrain, u_vehicle)
        vkt = float(FLEET_VKT[vehicle])
        cost_factor = self.factor(scenario, vehicle, powertrain, year, "purchase_cost_factor")
        vehicle_cost = np.asarray(b["vehicle_cost"], dtype=float) * cost_factor
        energy = self._energy_cost_usd_per_km(vehicle, powertrain, year, scenario, b, u_vehicle, u_price, pathway, uf)

        fee = np.asarray(b["fuel_fee_rate"], dtype=float)
        if vehicle == "cars" and powertrain == "phev":
            # JSON PHEV fuel fee is calibrated at UF=0.5 and 2025 fuel-mode
            # energy use. Scale it with both combustion-mode travel and the
            # PHEV-specific charge-sustaining efficiency trajectory.
            fee = (
                fee
                * ((1.0 - uf) / (1.0 - self.uf))
                * self.phev_mode_energy_factor(scenario, year, "fuel")
            )
        elif powertrain in ("gasoline", "hybrid", "diesel"):
            fee = fee * self.factor(scenario, vehicle, powertrain, year, "energy_use_factor")
        else:
            fee = np.zeros_like(np.asarray(energy))

        mr = np.asarray(b["maintenance_and_repair_cost"], dtype=float)
        insurance = np.asarray(b["insurance_cost"], dtype=float)
        labor_per_km = np.asarray(b["labor_cost"], dtype=float)

        years = np.arange(1, self.lifetime + 1, dtype=float)
        pv_sum = float(np.sum(1.0 / (1.0 + self.discount_rate) ** years))
        total_distance = vkt * self.lifetime

        fuel = (energy + fee) * pv_sum / self.lifetime
        other = (mr + insurance) * pv_sum / self.lifetime
        labor = labor_per_km * pv_sum / self.lifetime

        residual = vehicle_cost * (1.0 - self.depreciation_rate) ** self.lifetime
        replacement = self._replacement_cost_usd(vehicle, powertrain, year, scenario)
        capital = (vehicle_cost + replacement / (1.0 + self.discount_rate) ** self.replacement_year - residual / (1.0 + self.discount_rate) ** self.lifetime) / total_distance

        fuel_ntd = fuel * self.fx
        other_ntd = other * self.fx
        labor_ntd = labor * self.fx
        capital_ntd = capital * self.fx
        total_ntd = fuel_ntd + other_ntd + labor_ntd + capital_ntd
        return TCOResult(fuel=fuel_ntd, labor=labor_ntd, capital=capital_ntd, other_opex=other_ntd, total=total_ntd)

    # ---------------------------- uncertainty ------------------------------
    def lce_mc(self, *args, n: Optional[int] = None, seed_offset: int = 0, **kwargs) -> LCEResult:
        n = self.n_mc if n is None else n
        rng = np.random.default_rng(self.seed + seed_offset)
        u = self._triangular(rng, n)
        return self.lce(*args, u=u, **kwargs)

    def tco_mc(self, *args, n: Optional[int] = None, seed_offset: int = 0, **kwargs) -> TCOResult:
        n = self.n_mc if n is None else n
        rng = np.random.default_rng(self.seed + seed_offset)
        u_vehicle = self._triangular(rng, n)
        u_price = self._triangular(rng, n)
        return self.tco(*args, u_vehicle=u_vehicle, u_price=u_price, **kwargs)

    @staticmethod
    def q(x: np.ndarray) -> Tuple[float, float, float]:
        a = np.asarray(x, dtype=float)
        return tuple(float(v) for v in np.quantile(a, [0.05, 0.50, 0.95]))

    # --------------------------- output helpers ----------------------------
    def _savefig(self, fig: plt.Figure, stem: str):
        for d in self.fig_dirs:
            fig.savefig(d / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
            fig.savefig(d / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.05)

    def _save_table(self, df: pd.DataFrame, filename: str):
        out = self.root / "results"
        out.mkdir(exist_ok=True)
        df.to_csv(out / filename, index=False)

    # ------------------------------ plots ----------------------------------
    def plot_lce(self):
        """Plot the Reference-scenario vehicle-level LCE figure.

        Every vehicle/powertrain case is shown for both 2025 and 2050. Stacked
        bars show deterministic component values at the midpoint assumptions;
        black diamonds and whiskers show the Monte Carlo median and P5--P95.
        """
        plt.rcParams.update({
            'font.weight':'bold','axes.labelweight':'bold',
            'axes.titleweight':'bold','axes.linewidth':2,
            'xtick.major.width':1.8,'ytick.major.width':1.8,
        })
        panels = {
            "Trucks": [("Diesel","diesel",None),("Electric","electric",None),("H$_2$ blue","hydrogen","blue"),("H$_2$ green","hydrogen","green")],
            "Cars": [("Gasoline","gasoline",None),("Hybrid","hybrid",None),("PHEV","phev",None),("Electric","electric",None)],
            "Buses": [("Diesel","diesel",None),("Electric","electric",None)],
            "Light Trucks": [("Gasoline","gasoline",None),("Electric","electric",None)],
            "Scooters": [("Gasoline","gasoline",None),("Electric","electric",None)],
        }
        vehicle_key = {"Trucks":"trucks","Cars":"cars","Buses":"bus","Light Trucks":"trucks_light","Scooters":"scooters"}
        years = (2025, 2050)
        stack_colors = ['#1B5E20','#66BB6A','#C8E6C9']

        fig = plt.figure(figsize=(16.8,11.4), dpi=300)
        gs = gridspec.GridSpec(2,1,height_ratios=[1,1])
        top = gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=gs[0],width_ratios=[5,5],wspace=.18)
        bottom = gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=gs[1],wspace=.26)
        axes = {"Trucks":fig.add_subplot(top[0]),"Cars":fig.add_subplot(top[1]),"Buses":fig.add_subplot(bottom[0]),"Light Trucks":fig.add_subplot(bottom[1]),"Scooters":fig.add_subplot(bottom[2])}
        rows=[]

        for pi,(title,specs) in enumerate(panels.items()):
            ax=axes[title]; v=vehicle_key[title]
            width=.18 if len(specs)>=4 else .25
            x=np.arange(len(specs),dtype=float)*1.1
            ymax=0.0
            for i,(label,pt,pathway) in enumerate(specs):
                for j,(off,yr) in enumerate(zip((-width*.55,width*.55),years)):
                    det=self.lce(v,pt,yr,"Reference",pathway)
                    mc=self.lce_mc(v,pt,yr,"Reference",pathway,seed_offset=1000+pi*100+i*10+j)
                    q05,q50,q95=self.q(mc.total)
                    comps=[float(np.asarray(det.vehicle)),float(np.asarray(det.fuel_production)),float(np.asarray(det.fuel_use))]
                    xx=x[i]+off; btm=0.0
                    for comp,c in zip(comps,stack_colors):
                        ax.bar(xx,comp,width,bottom=btm,color=c,linewidth=0); btm+=comp
                    ax.errorbar(xx,q50,yerr=[[q50-q05],[q95-q50]],fmt='D',color='black',mfc='black',mec='black',markersize=4.2,ecolor='black',elinewidth=1.2,capsize=3.5,zorder=20)
                    ax.annotate(str(yr),(xx,q95),xytext=(0,10),textcoords='offset points',ha='center',va='bottom',rotation=90,fontsize=15.5,fontweight='bold')
                    ymax=max(ymax,q95)
                    rows.append({'panel':title,'vehicle':v,'powertrain':pt,'pathway':pathway,'year':yr,'vehicle_production':comps[0],'fuel_production':comps[1],'fuel_use':comps[2],'deterministic_total':sum(comps),'p05':q05,'median':q50,'p95':q95})
            ax.set_title(title,fontsize=24,pad=7,fontweight='bold')
            ax.set_xticks(x); ax.set_xticklabels([s[0] for s in specs],fontsize=16,fontweight='bold')
            ax.tick_params(axis='y',labelsize=15.5,width=1.8,length=5); ax.tick_params(axis='x',labelsize=15.5,width=1.8,length=4)
            ax.set_ylim(0,ymax*1.28)
            if title in ("Trucks","Buses"): ax.set_ylabel("Emissions\n(gCO$_2$-eq/km)",fontsize=19,fontweight='bold')
            for sp in ax.spines.values(): sp.set_linewidth(1.8)

        legend=[Patch(facecolor=stack_colors[0],label='Vehicle production'),Patch(facecolor=stack_colors[1],label='Fuel production'),Patch(facecolor=stack_colors[2],label='Fuel usage'),Line2D([0],[0],color='black',marker='D',markersize=5,linewidth=1.0,label='MC median (P5–P95)')]
        fig.legend(handles=legend,loc='upper center',ncol=4,frameon=False,bbox_to_anchor=(.5,.998),fontsize=17)
        fig.tight_layout(rect=[0,0,1,.91])
        self._savefig(fig,'LCE'); plt.close(fig)
        lce_df=pd.DataFrame(rows); self._save_table(lce_df,'LCE_reference_MC.csv'); self._save_table(lce_df,'LCE_summary.csv')

    def plot_tco(self):
        """Plot the Reference-scenario vehicle-level TCO figure.

        Every vehicle/powertrain case is shown for both 2025 and 2050. Stacked
        bars show deterministic cost components; black diamonds and whiskers
        show the Monte Carlo median and P5--P95.
        """
        plt.rcParams.update({
            'font.weight':'bold','axes.labelweight':'bold',
            'axes.titleweight':'bold','axes.linewidth':2,
            'xtick.major.width':1.8,'ytick.major.width':1.8,
        })
        panels = {
            "Trucks": [("Diesel","diesel",None),("Electric","electric",None),("H$_2$ blue","hydrogen","blue"),("H$_2$ green","hydrogen","green")],
            "Cars": [("Gasoline","gasoline",None),("Hybrid","hybrid",None),("PHEV","phev",None),("Electric","electric",None)],
            "Buses": [("Diesel","diesel",None),("Electric","electric",None)],
            "Light Trucks": [("Gasoline","gasoline",None),("Electric","electric",None)],
            "Scooters": [("Gasoline","gasoline",None),("Electric","electric",None)],
        }
        vehicle_key={"Trucks":"trucks","Cars":"cars","Buses":"bus","Light Trucks":"trucks_light","Scooters":"scooters"}
        years=(2025,2050)
        colors=['#4CAF50','#FF5722','#FFC107','#1E88E5']

        fig=plt.figure(figsize=(16.8,11.4),dpi=300)
        gs=gridspec.GridSpec(2,1,height_ratios=[1,1])
        top=gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=gs[0],width_ratios=[5,5],wspace=.18)
        bottom=gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=gs[1],wspace=.26)
        axes={"Trucks":fig.add_subplot(top[0]),"Cars":fig.add_subplot(top[1]),"Buses":fig.add_subplot(bottom[0]),"Light Trucks":fig.add_subplot(bottom[1]),"Scooters":fig.add_subplot(bottom[2])}
        rows=[]

        for pi,(title,specs) in enumerate(panels.items()):
            ax=axes[title]; v=vehicle_key[title]
            width=.18 if len(specs)>=4 else .25
            x=np.arange(len(specs),dtype=float)*1.1
            ymax=0.0
            for i,(label,pt,pathway) in enumerate(specs):
                for j,(off,yr) in enumerate(zip((-width*.55,width*.55),years)):
                    det=self.tco(v,pt,yr,"Reference",pathway)
                    mc=self.tco_mc(v,pt,yr,"Reference",pathway,seed_offset=2000+pi*100+i*10+j)
                    q05,q50,q95=self.q(mc.total)
                    comps=[float(np.asarray(det.fuel)),float(np.asarray(det.capital)),float(np.asarray(det.other_opex)),float(np.asarray(det.labor))]
                    xx=x[i]+off; btm=0.0
                    for comp,c in zip(comps,colors):
                        ax.bar(xx,comp,width,bottom=btm,color=c,linewidth=0); btm+=comp
                    ax.errorbar(xx,q50,yerr=[[q50-q05],[q95-q50]],fmt='D',color='black',mfc='black',mec='black',markersize=4.2,ecolor='black',elinewidth=1.2,capsize=3.5,zorder=20)
                    ax.annotate(str(yr),(xx,q95),xytext=(0,10),textcoords='offset points',ha='center',va='bottom',rotation=90,fontsize=15.5,fontweight='bold')
                    ymax=max(ymax,q95)
                    rows.append({'panel':title,'vehicle':v,'powertrain':pt,'pathway':pathway,'year':yr,'fuel':comps[0],'capital':comps[1],'other_opex':comps[2],'labor':comps[3],'deterministic_total':sum(comps),'p05':q05,'median':q50,'p95':q95})
            ax.set_title(title,fontsize=24,pad=7,fontweight='bold')
            ax.set_xticks(x); ax.set_xticklabels([s[0] for s in specs],fontsize=16,fontweight='bold')
            ax.tick_params(axis='y',labelsize=15.5,width=1.8,length=5); ax.tick_params(axis='x',labelsize=15.5,width=1.8,length=4)
            ax.set_ylim(0,ymax*1.28)
            if title in ("Trucks","Buses"): ax.set_ylabel("Cost (NT\\$/km)",fontsize=18,fontweight='bold')
            for sp in ax.spines.values(): sp.set_linewidth(1.8)

        legend=[Patch(facecolor=colors[0],label='Fuel'),Patch(facecolor=colors[1],label='Capital'),Patch(facecolor=colors[2],label='Other OPEX'),Patch(facecolor=colors[3],label='Labor'),Line2D([0],[0],color='black',marker='D',markersize=5,linewidth=1.0,label='MC median (P5–P95)')]
        fig.legend(handles=legend,loc='upper center',ncol=5,frameon=False,bbox_to_anchor=(.5,.998),fontsize=17)
        fig.tight_layout(rect=[0,0,1,.91])
        self._savefig(fig,'TCO'); plt.close(fig)
        tco_df=pd.DataFrame(rows); self._save_table(tco_df,'TCO_reference_MC.csv'); self._save_table(tco_df,'TCO_summary.csv')

    def plot_lce_vs_tco(self):
        """Plot the manuscript EV model comparison: LCE vs TCO for Taiwan BEV models.

        This figure is intentionally different from the multi-panel vehicle-class
        comparison used elsewhere. It reproduces the manuscript figure built from
        the 61 BEV models in lisa_cars.csv, using the study's 2025 Reference
        gasoline-car result as the baseline for CO2-avoidance cost.
        """
        plt.rcParams.update({
            'font.weight': 'bold',
            'axes.labelweight': 'bold',
            'axes.titleweight': 'bold',
            'axes.linewidth': 2,
            'xtick.major.width': 1.8,
            'ytick.major.width': 1.8,
        })

        lisa_path = self.data_dir / 'lisa_cars.csv'
        if not lisa_path.exists():
            fallback = Path('/mnt/data/lisa_cars.csv')
            if fallback.exists():
                lisa_path = fallback
            else:
                raise FileNotFoundError('lisa_cars.csv not found in data/ or /mnt/data.')
        df = pd.read_csv(lisa_path)

        # Use the study's own 2025 Reference gasoline-car baseline.
        baseline_cost = float(np.asarray(self.tco('cars', 'gasoline', 2025, 'Reference').total))
        baseline_emission = float(np.asarray(self.lce('cars', 'gasoline', 2025, 'Reference').total)) / 1000.0

        df_bev = df[df['Type'] == 'BEV'].copy()
        df_bev['CO2_abatement_cost'] = 1000.0 * (
            (df_bev['Total Cost'] - baseline_cost)
            / (baseline_emission - df_bev['Total Emission'])
        ) / 10000.0

        # Split positive vs. non-positive abatement costs for filled/open markers.
        df_bev_pos = df_bev[df_bev['CO2_abatement_cost'] > 0].copy()
        df_bev_neg = df_bev[df_bev['CO2_abatement_cost'] <= 0].copy()

        x = np.linspace(df_bev['Total Cost'].min() - 3.0, df_bev['Total Cost'].max() + 6.0, 500)
        y = np.linspace(df_bev['Total Emission'].min() - 0.03, df_bev['Total Emission'].max() + 0.01, 500)
        X, Y = np.meshgrid(x, y)
        Z = (X - baseline_cost) / (baseline_emission - Y) * 1000.0 / 10000.0

        fig, ax = plt.subplots(figsize=(12.8, 7.4), dpi=300)

        # Shade reasonable and expensive avoided-cost regions.
        reasonable_threshold = 5000.0 / 10000.0
        ax.contourf(X, Y, Z, levels=[-1e6, reasonable_threshold], colors=['green'], alpha=0.10)
        ax.contourf(X, Y, Z, levels=[800.0 * self.fx / 10000.0, 1e8], colors=['red'], alpha=0.05)

        markers = {'Sedan': 'o', 'SUV': 's', 'Hatchback': 'D', 'Wagons': '^'}
        default_marker = 'o'

        # Positive abatement cost: filled markers colored by avoided cost.
        last_scatter = None
        for vehicle_class, group in df_bev_pos.groupby('Class'):
            last_scatter = ax.scatter(
                group['Total Cost'],
                group['Total Emission'],
                c=group['CO2_abatement_cost'],
                cmap='viridis',
                s=70,
                edgecolor='black',
                marker=markers.get(vehicle_class, default_marker),
                label=vehicle_class,
                zorder=3,
            )

        # Non-positive abatement cost: open markers.
        added_labels = set(df_bev_pos['Class'].dropna().unique())
        for vehicle_class, group in df_bev_neg.groupby('Class'):
            label = vehicle_class if vehicle_class not in added_labels else None
            ax.scatter(
                group['Total Cost'],
                group['Total Emission'],
                facecolors='none',
                edgecolors='black',
                s=70,
                linewidths=1.2,
                marker=markers.get(vehicle_class, default_marker),
                label=label,
                zorder=3,
            )

        # Annotate edge cases only.
        edge_cases = pd.concat([
            df_bev.loc[[df_bev['Total Cost'].idxmin()]],
            df_bev.loc[[df_bev['Total Cost'].idxmax()]],
            df_bev.loc[[df_bev['Total Emission'].idxmin()]],
            df_bev.loc[[df_bev['Total Emission'].idxmax()]],
        ]).drop_duplicates()

        label_positions = {
            'Luxgen n7': (3.6, 0.160, 'left', 'center'),
            'Porsche Taycan': (38.1, 0.184, 'right', 'center'),
            'Opel Mokka Electric': (8.5, 0.124, 'left', 'center'),
            'Audi Q8 e-tron': (12.4, 0.213, 'left', 'center'),
        }
        for _, row in edge_cases.iterrows():
            text = f"{row['Model']}\n{row['Trim']}"
            label_x, label_y, ha, va = label_positions.get(
                str(row['Model']),
                (row['Total Cost'] + 1.0, row['Total Emission'] + 0.006, 'left', 'center'),
            )
            ax.annotate(
                text,
                xy=(row['Total Cost'], row['Total Emission']),
                xytext=(label_x, label_y),
                textcoords='data',
                ha=ha,
                va=va,
                fontsize=10.2,
                fontweight='bold',
                linespacing=0.95,
                arrowprops=dict(arrowstyle='-', color='0.25', lw=0.8, shrinkA=2, shrinkB=5),
                bbox=dict(facecolor='white', edgecolor='0.75', linewidth=0.35, alpha=0.92, pad=1.5),
                zorder=4,
            )

        ax.text(0.005, 0.98, 'Reasonable', transform=ax.transAxes, fontsize=13, fontweight='bold', color='green', ha='left', va='top')
        ax.text(0.99, 0.98, 'Expensive', transform=ax.transAxes, fontsize=13, fontweight='bold', color='red', ha='right', va='top')

        sm = plt.cm.ScalarMappable(cmap='viridis')
        sm.set_array(df_bev_pos['CO2_abatement_cost'].to_numpy() if not df_bev_pos.empty else np.array([0.0]))
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label(r'Cost of CO$_2$ avoided (10KNT\$/tCO$_2$)', fontsize=20, fontweight='bold')
        cbar.ax.tick_params(labelsize=14)

        ax.set_xlabel('Total Cost of Ownership (NT$/km)', fontsize=20, fontweight='bold')
        ax.set_ylabel('Total Emissions (kgCO$_2$eq/km)', fontsize=20, fontweight='bold')
        ax.set_xlim(3, 42)
        ax.set_ylim(0.12, 0.23)
        ax.tick_params(labelsize=15.5, width=1.8, length=5)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=True, loc='lower right', fontsize=15)
        fig.tight_layout()

        self._savefig(fig, 'LCEvsTCO')
        plt.close(fig)

        out = df_bev[['Model', 'Trim', 'Class', 'Origin', 'Price', 'Total Cost', 'Total Emission', 'CO2_abatement_cost']].copy()
        out['baseline_gasoline_cost_ntd_per_km'] = baseline_cost
        out['baseline_gasoline_emission_kg_per_km'] = baseline_emission
        self._save_table(out, 'LCEvsTCO_vehicle_models.csv')

    def avoided_cost(self, vehicle: str, powertrain: str, year: int, scenario: str, pathway: Optional[str]=None) -> float:
        ref_pt='diesel' if vehicle in ('bus','trucks') else 'gasoline'
        t_alt=float(np.asarray(self.tco(vehicle,powertrain,year,scenario,pathway).total))
        t_ref=float(np.asarray(self.tco(vehicle,ref_pt,year,scenario).total))
        l_alt=float(np.asarray(self.lce(vehicle,powertrain,year,scenario,pathway).total))
        l_ref=float(np.asarray(self.lce(vehicle,ref_pt,year,scenario).total))
        de=l_ref-l_alt
        return np.nan if de<=0 else (t_alt-t_ref)*1e6/de

    def plot_avoided_cost(self):
        years=np.arange(2025,2051); scenarios=['Conservative','Reference','Optimistic']; dac=800*self.fx
        cases_a=[('bus','electric',None,'Buses','lightseagreen'),('trucks_light','electric',None,'Light Trucks','navajowhite'),('cars','electric',None,'Cars','pink'),('scooters','electric',None,'Scooters','lightgreen')]
        cases_b=[('trucks','electric',None,'Battery Electric Truck','silver'),('trucks','hydrogen','blue','Blue Hydrogen Truck','lightblue'),('trucks','hydrogen','green','Green Hydrogen Truck','lightgreen')]
        rows=[]
        def one(which,cases,stem):
            fig,axes=plt.subplots(1,3,figsize=(19.5,5.0),dpi=300,sharey=True); fig.subplots_adjust(top=.78,bottom=.16,wspace=.08)
            for ax,sc in zip(axes,scenarios):
                vis=list(cases)
                n=len(vis); bw=.18 if len(cases)==4 else .24; offs=np.linspace(-(n-1)*bw/2,(n-1)*bw/2,n)
                for off,(v,pt,path,label,col) in zip(offs,vis):
                    vals=[]
                    for yr in years:
                        val=self.avoided_cost(v,pt,int(yr),sc,path); vals.append(val); rows.append({'figure':which,'scenario':sc,'year':yr,'label':label,'cost_ntd_per_tco2':val})
                    ax.bar(np.arange(len(years))+off,np.nan_to_num(vals,nan=0),width=bw,color=col,zorder=2)
                ax.axhline(dac,linestyle='-.',color='black',lw=2,zorder=3); ax.set_ylim(0,50000); ax.set_xlim(-.6,len(years)-.4); ax.set_title(sc,fontsize=22,fontweight='bold'); ax.set_xlabel('Year',fontsize=17,fontweight='bold')
                ax.set_xticks(np.arange(0,len(years),5)); ax.set_xticklabels(np.arange(2025,2051,5),fontsize=13.5,fontweight='bold'); ax.grid(axis='y',ls='--',alpha=.7); ax.tick_params(axis='y',labelsize=14,width=2)
                for sp in ax.spines.values(): sp.set_linewidth(2)
            axes[0].set_ylabel('NT\\$/tCO$_2$eq',fontsize=17,fontweight='bold')
            handles=[Patch(facecolor=c[4],label=c[3]) for c in cases]+[Line2D([0],[0],color='black',ls='-.',lw=2,label='DAC')]
            fig.legend(handles=handles,loc='upper center',ncol=len(handles),frameon=False,fontsize=14)
            self._savefig(fig,stem); plt.close(fig)
        one('A',cases_a,'Cost_co2_avoided_a'); one('B',cases_b,'Cost_co2_avoided_b'); self._save_table(pd.DataFrame(rows),'Cost_of_CO2_avoided_scenarios.csv')

    def _contour_colors(self):
        # Positive avoided-cost bands use the green-to-red scale. Values below
        # zero are assigned explicitly to the colormap's under-range so the
        # gray/green transition is exactly the TCO-parity boundary (Z = 0),
        # rather than a regular contour interval that can shift when extend is used.
        levels=np.array([0,.25,.5,.75,1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,3.25,3.5,3.75,4,4.25,4.5,4.75,5,6,8,10,15,20,30,50,100,200],dtype=float)
        colors=plt.get_cmap('RdYlGn_r')(np.linspace(0,1,len(levels)-1))
        cmap=ListedColormap(colors)
        cmap.set_under((.55,.55,.55,1.0))
        cmap.set_over(colors[-1])
        norm=BoundaryNorm(levels,cmap.N)
        return levels,cmap,norm

    def plot_contours(self):
        scenarios=['Conservative','Reference','Optimistic']; five=5000/10000; dac=800*self.fx/10000; levels,cmap,norm=self._contour_colors()
        # BEV heavy truck with year-specific broken-y contour panels.
        ep=np.linspace(0,20.0,141); gc=np.linspace(0,.45,121); X,Y=np.meshgrid(ep,gc)
        fig=plt.figure(figsize=(17.2,6.4), dpi=300)
        outer=gridspec.GridSpec(1,4, figure=fig, left=.075, right=.90, top=.83, bottom=.13, wspace=.18, width_ratios=[1,1,1,.06])
        last=None
        elec_lo, elec_hi = map(float, self.fuel_prices['electricity_per_kwh'])
        xmid = 0.5*(elec_lo + elec_hi)

        def _draw_break(ax_top, ax_bottom):
            ax_top.spines['bottom'].set_visible(False)
            ax_bottom.spines['top'].set_visible(False)
            ax_top.tick_params(labelbottom=False, bottom=False)
            kwargs_top=dict(transform=ax_top.transAxes, color='k', clip_on=False, linewidth=1.2)
            kwargs_bottom=dict(transform=ax_bottom.transAxes, color='k', clip_on=False, linewidth=1.2)
            ax_top.plot([0.05,0.95],[-0.012,-0.012],**kwargs_top)
            ax_top.plot([0.05,0.95],[-0.035,-0.035],**kwargs_top)
            ax_bottom.plot([0.05,0.95],[1.012,1.012],**kwargs_bottom)
            ax_bottom.plot([0.05,0.95],[1.035,1.035],**kwargs_bottom)

        for c,sc in enumerate(scenarios):
            inner=gridspec.GridSpecFromSubplotSpec(2,1, subplot_spec=outer[c], hspace=.05)
            ax_top=fig.add_subplot(inner[0])
            ax_bottom=fig.add_subplot(inner[1], sharex=ax_top)
            pair_center=(ax_top.get_position().x0 + ax_top.get_position().x1)/2
            fig.text(pair_center,0.85,sc,ha='center',va='bottom',fontsize=20,fontweight='bold')
            for year,ax in [(2025,ax_top),(2050,ax_bottom)]:
                tref=float(np.asarray(self.tco('trucks','diesel',year,sc).total)); lref=float(np.asarray(self.lce('trucks','diesel',year,sc).total))
                b=self._bundle('trucks','electric',.5); eff=float(b['fuel_efficiency'])*self.factor(sc,'trucks','electric',year,'energy_use_factor'); vc=float(b['vehicle_cost'])*self.factor(sc,'trucks','electric',year,'purchase_cost_factor'); mr=float(b['maintenance_and_repair_cost']); ins=float(b['insurance_cost']); labor=float(b['labor_cost']); vkt=FLEET_VKT['trucks']; pv=sum(1/(1+self.discount_rate)**np.arange(1,self.lifetime+1)); dist=vkt*self.lifetime
                resid=vc*(1-self.depreciation_rate)**self.lifetime; cap=(vc+self._replacement_cost_usd('trucks','electric',year,sc)/(1+self.discount_rate)**self.replacement_year-resid/(1+self.discount_rate)**self.lifetime)/dist*self.fx; other=(mr+ins)*pv/self.lifetime*self.fx; lab=labor*pv/self.lifetime*self.fx
                talt=cap+other+lab+eff*X/self.fx*pv/self.lifetime*self.fx; vp=self._vehicle_cycle_g_per_km('trucks','electric'); lalt=vp+eff*Y*1000; Z=(talt-tref)*1e6/(lref-lalt)/10000; Z=np.where(lref-lalt>0,Z,np.nan)
                last=ax.contourf(X,Y,Z,levels=levels,cmap=cmap,norm=norm,extend='both'); ax.contour(X,Y,Z,levels=[five],colors='black',linestyles=':',linewidths=1.2); ax.contour(X,Y,Z,levels=[dac],colors='black',linestyles='--',linewidths=1.35)
                y=self.grid_ci(sc,year)
                marker='o' if year==2025 else 's'; face='white' if year==2025 else 'royalblue'
                ax.plot(xmid,y,marker=marker,mfc=face,mec='royalblue' if year==2050 else 'red',mew=1.5,ms=6.5,zorder=6)
                ax.text(0.03,0.93,str(year),transform=ax.transAxes,ha='left',va='top',fontsize=13.5,fontweight='bold',bbox=dict(facecolor='white', edgecolor='none', alpha=.75, pad=1.3))
                ax.set_xlim(0,20.0); ax.tick_params(labelsize=12.5,width=1.4)
                for sp in ax.spines.values(): sp.set_linewidth(1.6)
            # reasonable split between 2025 and 2050 grid CI values
            ax_top.set_ylim(0.27,0.45)
            ax_bottom.set_ylim(0.0,0.25)
            _draw_break(ax_top, ax_bottom)
            con=ConnectionPatch(xyA=(xmid, self.grid_ci(sc,2025)), coordsA=ax_top.transData,
                                xyB=(xmid, self.grid_ci(sc,2050)), coordsB=ax_bottom.transData,
                                arrowstyle='->', lw=1.5, color='royalblue', mutation_scale=11)
            fig.add_artist(con)
            ax_bottom.set_xlabel(r'Electricity price (NT\$/kWh)',fontsize=14,fontweight='bold')

        fig.text(.018,.50,'Grid carbon intensity\n(kgCO$_2$eq/kWh)',rotation=90,va='center',ha='center',fontsize=15.5,fontweight='bold')
        cax=fig.add_subplot(outer[3]); cb=fig.colorbar(last,cax=cax); cb.set_label(r'Cost of CO$_2$ avoided (10K NT\$/tCO$_2$eq)',fontsize=16.5,fontweight='bold'); cb.ax.tick_params(labelsize=13)
        handles=[Line2D([0],[0],color='black',ls=':',label=r'5K NT\$'),Line2D([0],[0],color='black',ls='--',label='DAC'),Line2D([0],[0],marker='o',mfc='white',mec='red',ls='None',label='2025'),Line2D([0],[0],marker='s',mfc='royalblue',mec='royalblue',ls='None',label='2050'),Line2D([0],[0],color='royalblue',lw=1.7,label='BET'),Line2D([0],[0],color='black',lw=1.5,label='2025 → 2050')]
        fig.legend(handles=handles,loc='upper center',ncol=6,frameon=False,fontsize=14)
        self._savefig(fig,'contour_electricity'); plt.close(fig)

        # Hydrogen contours: split blue and green into separate figures with a horizontal break.
        hp=np.linspace(0,529.6,151); hc=np.linspace(0,11.4,151); X,Y=np.meshgrid(hp,hc)
        pathcols={'blue':'royalblue','green':'forestgreen'}

        def _draw_break(ax_top, ax_bottom):
            ax_top.spines['bottom'].set_visible(False)
            ax_bottom.spines['top'].set_visible(False)
            ax_top.tick_params(labelbottom=False, bottom=False)
            kwargs_top=dict(transform=ax_top.transAxes, color='k', clip_on=False, linewidth=1.2)
            kwargs_bottom=dict(transform=ax_bottom.transAxes, color='k', clip_on=False, linewidth=1.2)
            ax_top.plot([0.05,0.95],[-0.012,-0.012],**kwargs_top)
            ax_top.plot([0.05,0.95],[-0.035,-0.035],**kwargs_top)
            ax_bottom.plot([0.05,0.95],[1.012,1.012],**kwargs_bottom)
            ax_bottom.plot([0.05,0.95],[1.035,1.035],**kwargs_bottom)

        def _plot_h2_path(path, fig_key, title_text, top_ylim, bottom_ylim):
            fig=plt.figure(figsize=(17.2,6.4), dpi=300)
            outer=gridspec.GridSpec(1,4, figure=fig, left=.075, right=.90, top=.83, bottom=.13, wspace=.18, width_ratios=[1,1,1,.06])
            last=None
            col=pathcols[path]
            for c,sc in enumerate(scenarios):
                inner=gridspec.GridSpecFromSubplotSpec(2,1, subplot_spec=outer[c], hspace=.05)
                ax_top=fig.add_subplot(inner[0])
                ax_bottom=fig.add_subplot(inner[1], sharex=ax_top)
                pair_center=(ax_top.get_position().x0 + ax_top.get_position().x1)/2
                fig.text(pair_center,0.85,sc,ha='center',va='bottom',fontsize=20,fontweight='bold')
                for year,ax in [(2025,ax_top),(2050,ax_bottom)]:
                    tref=float(np.asarray(self.tco('trucks','diesel',year,sc).total)); lref=float(np.asarray(self.lce('trucks','diesel',year,sc).total))
                    b=self._bundle('trucks','hydrogen',.5); eff=float(b['fuel_efficiency'])*self.factor(sc,'trucks','hydrogen',year,'energy_use_factor'); vc=float(b['vehicle_cost'])*self.factor(sc,'trucks','hydrogen',year,'purchase_cost_factor'); mr=float(b['maintenance_and_repair_cost']); ins=float(b['insurance_cost']); labor=float(b['labor_cost']); vkt=FLEET_VKT['trucks']; pv=sum(1/(1+self.discount_rate)**np.arange(1,self.lifetime+1)); dist=vkt*self.lifetime
                    resid=vc*(1-self.depreciation_rate)**self.lifetime; cap=(vc+self._replacement_cost_usd('trucks','hydrogen',year,sc)/(1+self.discount_rate)**self.replacement_year-resid/(1+self.discount_rate)**self.lifetime)/dist*self.fx; other=(mr+ins)*pv/self.lifetime*self.fx; lab=labor*pv/self.lifetime*self.fx
                    talt=cap+other+lab+(eff/float(self.ec['hydrogen_kwh_per_kg']))*X/self.fx*pv/self.lifetime*self.fx; vp=self._vehicle_cycle_g_per_km('trucks','hydrogen'); lalt=vp+eff/float(self.ec['hydrogen_kwh_per_kg'])*Y*1000; Z=(talt-tref)*1e6/(lref-lalt)/10000; Z=np.where(lref-lalt>0,Z,np.nan)
                    last=ax.contourf(X,Y,Z,levels=levels,cmap=cmap,norm=norm,extend='both'); ax.contour(X,Y,Z,levels=[five],colors='black',linestyles=':',linewidths=1.2); ax.contour(X,Y,Z,levels=[dac],colors='black',linestyles='--',linewidths=1.35)
                    x=self.h2_production_mid(path)+self.h2_dispensing(sc,year); y=self.h2_ci(path,year)
                    marker='o' if year==2025 else 's'; face='white' if year==2025 else col
                    ax.plot(x,y,marker=marker,mfc=face,mec=col,mew=1.5,ms=6.5,zorder=6)
                    ax.text(0.03,0.93,str(year),transform=ax.transAxes,ha='left',va='top',fontsize=13.5,fontweight='bold',bbox=dict(facecolor='white', edgecolor='none', alpha=.75, pad=1.3))
                    ax.set_xlim(0,529.6); ax.tick_params(labelsize=12.5,width=1.4)
                    for sp in ax.spines.values(): sp.set_linewidth(1.6)
                ax_top.set_ylim(*top_ylim); ax_bottom.set_ylim(*bottom_ylim)
                _draw_break(ax_top, ax_bottom)
                con=ConnectionPatch(xyA=(self.h2_production_mid(path)+self.h2_dispensing(sc,2025), self.h2_ci(path,2025)), coordsA=ax_top.transData,
                                    xyB=(self.h2_production_mid(path)+self.h2_dispensing(sc,2050), self.h2_ci(path,2050)), coordsB=ax_bottom.transData,
                                    arrowstyle='->', lw=1.5, color=col, mutation_scale=11)
                fig.add_artist(con)
                ax_bottom.set_xlabel(r'Hydrogen price at pump (NT\$/kg)',fontsize=14,fontweight='bold')
            fig.text(.018,.50,'Hydrogen production emissions\n(kgCO$_2$eq/kg H$_2$)',rotation=90,va='center',ha='center',fontsize=15.5,fontweight='bold')
            cax=fig.add_subplot(outer[3]); cb=fig.colorbar(last,cax=cax); cb.set_label(r'Cost of CO$_2$ avoided (10K NT\$/tCO$_2$eq)',fontsize=16.5,fontweight='bold'); cb.ax.tick_params(labelsize=13)
            handles=[Line2D([0],[0],color='black',ls=':',label=r'5K NT\$'),Line2D([0],[0],color='black',ls='--',label='DAC'),Line2D([0],[0],marker='o',mfc='white',mec='black',ls='None',label='2025'),Line2D([0],[0],marker='s',mfc='black',mec='black',ls='None',label='2050'),Line2D([0],[0],color=col,lw=1.7,label=title_text),Line2D([0],[0],color='black',lw=1.5,label='2025 → 2050')]
            fig.legend(handles=handles,loc='upper center',ncol=6,frameon=False,fontsize=14)
            self._savefig(fig, fig_key); plt.close(fig)

        _plot_h2_path('blue','contour_hydrogen_blue','Blue H$_2$', top_ylim=(6.0,11.4), bottom_ylim=(0.0,6.0))
        _plot_h2_path('green','contour_hydrogen_green','Green H$_2$', top_ylim=(0.7,1.5), bottom_ylim=(0.0,0.7))

    # ---------------------- deterministic sensitivity ----------------------
    def _sensitivity_tco(
        self,
        vehicle: str,
        powertrain: str,
        year: int = 2050,
        scenario: str = "Reference",
        pathway: Optional[str] = None,
        *,
        discount_rate: Optional[float] = None,
        vkt: Optional[float] = None,
        lifetime_years: Optional[int] = None,
        electricity_price_ntd_per_kwh: Optional[float] = None,
        gasoline_price_ntd_per_l: Optional[float] = None,
        diesel_price_ntd_per_l: Optional[float] = None,
        h2_delivered_price_ntd_per_kg: Optional[float] = None,
        scooter_energy_cost_usd_per_km: Optional[float] = None,
        utility_factor: Optional[float] = None,
        replacement_count: int = 1,
        battery_salvage_ntd_per_kwh: Optional[float] = None,
    ) -> float:
        """Deterministic TCO evaluator used by one-at-a-time sensitivity plots.

        Defaults reproduce the midpoint/reference ``tco`` calculation. Optional
        overrides change one physical/economic assumption without altering the
        underlying low/high vehicle bundle or future technology scenario.
        """
        r = self.discount_rate if discount_rate is None else float(discount_rate)
        vkt = float(FLEET_VKT[vehicle] if vkt is None else vkt)
        lifetime = self.lifetime if lifetime_years is None else int(lifetime_years)
        if lifetime < 1:
            raise ValueError("Vehicle lifetime must be at least 1 year.")
        uf = self.uf if utility_factor is None else float(utility_factor)

        b = self._bundle(vehicle, powertrain, 0.5)
        cost_factor = self.factor(scenario, vehicle, powertrain, year, "purchase_cost_factor")
        vehicle_cost = float(b["vehicle_cost"]) * cost_factor
        fee = float(b["fuel_fee_rate"])

        if vehicle == "scooters" and powertrain == "electric":
            # Taiwan scooter operating costs are represented by the battery-
            # swapping per-km cost already contained in the vehicle input case.
            energy_usd_per_km = (
                float(self.scooter_swap_cost_usd_per_km(b["fuel_cost"], scenario, year))
                if scooter_energy_cost_usd_per_km is None
                else float(scooter_energy_cost_usd_per_km)
            )
            fee = 0.0

        elif vehicle == "cars" and powertrain == "phev":
            # PHEV charge-sustaining and charge-depleting modes follow the
            # selected PHEV-specific ATB CS/CD trajectories.
            fuel_eff, electric_eff = self.phev_mode_efficiencies(scenario, year, 0.5)
            fuel_eff = float(fuel_eff)
            electric_eff = float(electric_eff)
            if gasoline_price_ntd_per_l is None:
                gl, gh = self.fuel_prices["gasoline_per_l"]
                gasoline_price_ntd_per_l = 0.5 * (float(gl) + float(gh))
            if electricity_price_ntd_per_kwh is None:
                el, eh = self.fuel_prices["electricity_per_kwh"]
                electricity_price_ntd_per_kwh = 0.5 * (float(el) + float(eh))
            energy_usd_per_km = (
                (1.0 - uf)
                * fuel_eff
                / float(self.ec["gasoline_kwh_per_l"])
                * float(gasoline_price_ntd_per_l)
                / self.fx
                + uf * electric_eff * float(electricity_price_ntd_per_kwh) / self.fx
            )
            fee = (
                fee
                * ((1.0 - uf) / (1.0 - self.uf))
                * self.phev_mode_energy_factor(scenario, year, "fuel")
            )

        else:
            eff_factor = self.factor(scenario, vehicle, powertrain, year, "energy_use_factor")
            eff = float(b["fuel_efficiency"]) * eff_factor
            if powertrain in ("gasoline", "hybrid"):
                if gasoline_price_ntd_per_l is None:
                    lo, hi = self.fuel_prices["gasoline_per_l"]
                    gasoline_price_ntd_per_l = 0.5 * (float(lo) + float(hi))
                energy_usd_per_km = (
                    eff / float(self.ec["gasoline_kwh_per_l"])
                    * float(gasoline_price_ntd_per_l) / self.fx
                )
                fee *= eff_factor
            elif powertrain == "diesel":
                if diesel_price_ntd_per_l is None:
                    lo, hi = self.fuel_prices["diesel_per_l"]
                    diesel_price_ntd_per_l = 0.5 * (float(lo) + float(hi))
                energy_usd_per_km = (
                    eff / float(self.ec["diesel_kwh_per_l"])
                    * float(diesel_price_ntd_per_l) / self.fx
                )
                fee *= eff_factor
            elif powertrain == "electric":
                if electricity_price_ntd_per_kwh is None:
                    lo, hi = self.fuel_prices["electricity_per_kwh"]
                    electricity_price_ntd_per_kwh = 0.5 * (float(lo) + float(hi))
                energy_usd_per_km = eff * float(electricity_price_ntd_per_kwh) / self.fx
                fee = 0.0
            elif powertrain == "hydrogen":
                if pathway not in ("blue", "green"):
                    raise ValueError("Hydrogen pathway must be 'blue' or 'green'.")
                if h2_delivered_price_ntd_per_kg is None:
                    h2_delivered_price_ntd_per_kg = (
                        self.h2_production_mid(pathway) + self.h2_dispensing(scenario, year)
                    )
                energy_usd_per_km = (
                    eff / float(self.ec["hydrogen_kwh_per_kg"])
                    * float(h2_delivered_price_ntd_per_kg) / self.fx
                )
                fee = 0.0
            else:
                raise ValueError((vehicle, powertrain))

        mr = float(b["maintenance_and_repair_cost"])
        insurance = float(b["insurance_cost"])
        labor_per_km = float(b["labor_cost"])
        years = np.arange(1, lifetime + 1, dtype=float)
        pv_sum = float(np.sum(1.0 / (1.0 + r) ** years))

        fuel_ntd = (energy_usd_per_km + fee) * pv_sum / lifetime * self.fx
        other_ntd = (mr + insurance) * pv_sum / lifetime * self.fx
        labor_ntd = labor_per_km * pv_sum / lifetime * self.fx

        residual = vehicle_cost * (1.0 - self.depreciation_rate) ** lifetime
        replacement = self._replacement_cost_usd(vehicle, powertrain, year, scenario) * int(replacement_count)
        salvage_ntd = 0.0
        if battery_salvage_ntd_per_kwh is not None and int(replacement_count) > 0:
            salvage_ntd = BATTERY_KWH.get((vehicle, powertrain), 0.0) * float(battery_salvage_ntd_per_kwh)
        capital_ntd = (
            vehicle_cost
            + replacement / (1.0 + r) ** self.replacement_year
            - (salvage_ntd / self.fx) / (1.0 + r) ** self.replacement_year
            - residual / (1.0 + r) ** lifetime
        ) / (vkt * lifetime) * self.fx

        return float(fuel_ntd + other_ntd + labor_ntd + capital_ntd)

    def _sensitivity_lce(
        self,
        vehicle: str,
        powertrain: str,
        year: int = 2050,
        scenario: str = "Reference",
        pathway: Optional[str] = None,
        *,
        vkt: Optional[float] = None,
        lifetime_years: Optional[int] = None,
        grid_ci_kgco2_per_kwh: Optional[float] = None,
        h2_ci_kgco2_per_kg: Optional[float] = None,
        utility_factor: Optional[float] = None,
        replacement_count: int = 1,
    ) -> float:
        """Deterministic LCE evaluator used by one-at-a-time sensitivity plots."""
        vkt = float(FLEET_VKT[vehicle] if vkt is None else vkt)
        lifetime = self.lifetime if lifetime_years is None else int(lifetime_years)
        if lifetime < 1:
            raise ValueError("Vehicle lifetime must be at least 1 year.")
        uf = self.uf if utility_factor is None else float(utility_factor)
        b = self._bundle(vehicle, powertrain, 0.5)

        production_kg = VEHICLE_PRODUCTION_KG[(vehicle, powertrain)]
        replacement_kg = (
            BATTERY_REPLACEMENT_EMISSIONS_KG[(vehicle, powertrain)]
            * int(replacement_count)
        )
        vehicle_cycle = (production_kg + replacement_kg) * 1000.0 / (lifetime * vkt)

        if vehicle == "cars" and powertrain == "phev":
            fuel_eff, electric_eff = self.phev_mode_efficiencies(scenario, year, 0.5)
            fuel_eff = float(fuel_eff)
            electric_eff = float(electric_eff)
            grid_ci = (
                self.grid_ci(scenario, year)
                if grid_ci_kgco2_per_kwh is None
                else float(grid_ci_kgco2_per_kwh)
            )
            liters = (1.0 - uf) * fuel_eff / float(self.ec["gasoline_kwh_per_l"])
            operating = (
                liters * float(self.ec["gasoline_ci_kgco2_per_l"]) * 1000.0
                + uf * electric_eff * grid_ci * 1000.0
            )
        else:
            eff = float(b["fuel_efficiency"]) * self.factor(
                scenario, vehicle, powertrain, year, "energy_use_factor"
            )
            if powertrain in ("gasoline", "hybrid"):
                operating = (
                    eff / float(self.ec["gasoline_kwh_per_l"])
                    * float(self.ec["gasoline_ci_kgco2_per_l"]) * 1000.0
                )
            elif powertrain == "diesel":
                operating = (
                    eff / float(self.ec["diesel_kwh_per_l"])
                    * float(self.ec["diesel_ci_kgco2_per_l"]) * 1000.0
                )
            elif powertrain == "electric":
                grid_ci = (
                    self.grid_ci(scenario, year)
                    if grid_ci_kgco2_per_kwh is None
                    else float(grid_ci_kgco2_per_kwh)
                )
                operating = eff * grid_ci * 1000.0
            elif powertrain == "hydrogen":
                if pathway not in ("blue", "green"):
                    raise ValueError("Hydrogen pathway must be 'blue' or 'green'.")
                h2_ci = (
                    self.h2_ci(pathway, year)
                    if h2_ci_kgco2_per_kg is None
                    else float(h2_ci_kgco2_per_kg)
                )
                operating = (
                    eff / float(self.ec["hydrogen_kwh_per_kg"])
                    * h2_ci * 1000.0
                )
            else:
                raise ValueError((vehicle, powertrain))

        return float(vehicle_cycle + operating)

    def _sensitivity_avoided_cost(
        self,
        vehicle: str,
        powertrain: str,
        year: int = 2050,
        scenario: str = "Reference",
        pathway: Optional[str] = None,
        **overrides,
    ) -> float:
        """Cost of CO2 avoided under deterministic one-at-a-time overrides.

        Class-wide assumptions such as VKT, discount rate, and gasoline/diesel
        prices are applied consistently to both the alternative and its
        conventional comparator.
        """
        ref_pt = "diesel" if vehicle in ("bus", "trucks") else "gasoline"

        tco_keys = {
            "discount_rate", "vkt", "lifetime_years", "electricity_price_ntd_per_kwh",
            "gasoline_price_ntd_per_l", "diesel_price_ntd_per_l",
            "h2_delivered_price_ntd_per_kg", "scooter_energy_cost_usd_per_km",
            "utility_factor", "replacement_count", "battery_salvage_ntd_per_kwh",
        }
        lce_keys = {
            "vkt", "lifetime_years", "grid_ci_kgco2_per_kwh", "h2_ci_kgco2_per_kg",
            "utility_factor", "replacement_count",
        }
        alt_tco_kwargs = {k: v for k, v in overrides.items() if k in tco_keys}
        alt_lce_kwargs = {k: v for k, v in overrides.items() if k in lce_keys}

        # Only global assumptions relevant to the reference technology propagate
        # to the comparator. Alternative-specific electricity/H2 inputs do not.
        ref_tco_kwargs = {}
        for key in ("discount_rate", "vkt", "lifetime_years"):
            if key in overrides:
                ref_tco_kwargs[key] = overrides[key]
        if ref_pt == "gasoline" and "gasoline_price_ntd_per_l" in overrides:
            ref_tco_kwargs["gasoline_price_ntd_per_l"] = overrides["gasoline_price_ntd_per_l"]
        if ref_pt == "diesel" and "diesel_price_ntd_per_l" in overrides:
            ref_tco_kwargs["diesel_price_ntd_per_l"] = overrides["diesel_price_ntd_per_l"]

        ref_lce_kwargs = {}
        if "vkt" in overrides:
            ref_lce_kwargs["vkt"] = overrides["vkt"]
        if "lifetime_years" in overrides:
            ref_lce_kwargs["lifetime_years"] = overrides["lifetime_years"]

        alt_tco = self._sensitivity_tco(
            vehicle, powertrain, year, scenario, pathway, **alt_tco_kwargs
        )
        ref_tco = self._sensitivity_tco(
            vehicle, ref_pt, year, scenario, None, **ref_tco_kwargs
        )
        alt_lce = self._sensitivity_lce(
            vehicle, powertrain, year, scenario, pathway, **alt_lce_kwargs
        )
        ref_lce = self._sensitivity_lce(
            vehicle, ref_pt, year, scenario, None, **ref_lce_kwargs
        )

        delta_emissions = ref_lce - alt_lce  # g CO2e/km
        if delta_emissions <= 0:
            return float("nan")
        return float((alt_tco - ref_tco) * 1e6 / delta_emissions)

    def tornado_sensitivity_data(
        self, year: int = 2050, scenario: str = "Reference"
    ) -> pd.DataFrame:
        """Return the deterministic sensitivity ranges used in the tornado plot."""
        sens = self.cfg["deterministic_sensitivity"]
        discount_low, discount_high = map(float, sens["discount_rate"])
        vkt_low_mult, vkt_high_mult = map(float, sens["vkt_multiplier"])
        lifetime_low, lifetime_high = map(int, sens.get("lifetime_years", [13, 19]))
        uf_low, uf_high = map(float, sens["phev_utility_factor"])
        salvage_low, salvage_high = map(float, sens.get("battery_salvage_ntd_per_kwh", [0.0, 841.0]))
        threshold = float(sens["abatement_threshold_ntd_per_tco2"])
        _ = threshold  # retained in the same config used by the plot

        elec_low, elec_high = map(float, self.fuel_prices["electricity_per_kwh"])
        gas_low, gas_high = map(float, self.fuel_prices["gasoline_per_l"])
        grid_low = self.grid_ci("Optimistic", year)
        grid_high = self.grid_ci("Conservative", year)

        scooter_swap_low = float(self.scooter_swap_cost_usd_per_km(
            self._bundle("scooters", "electric", 0.0)["fuel_cost"], scenario, year))
        scooter_swap_high = float(self.scooter_swap_cost_usd_per_km(
            self._bundle("scooters", "electric", 1.0)["fuel_cost"], scenario, year))

        panels = [
            ("Electric scooter", "scooters", "electric", None),
            ("PHEV car", "cars", "phev", None),
            ("Electric car", "cars", "electric", None),
            ("Electric light truck", "trucks_light", "electric", None),
            ("Electric bus", "bus", "electric", None),
            ("Electric heavy truck", "trucks", "electric", None),
            ("Blue H$_2$ truck", "trucks", "hydrogen", "blue"),
            ("Green H$_2$ truck", "trucks", "hydrogen", "green"),
        ]

        rows = []
        for title, vehicle, powertrain, pathway in panels:
            vkt_mid = float(FLEET_VKT[vehicle])
            vkt_low = vkt_low_mult * vkt_mid
            vkt_high = vkt_high_mult * vkt_mid
            baseline = self._sensitivity_avoided_cost(
                vehicle, powertrain, year, scenario, pathway
            )

            if vehicle == "scooters" and powertrain == "electric":
                variables = [
                    ("Battery swapping cost", f"{scooter_swap_low:.4f} USD/km", f"{scooter_swap_high:.4f} USD/km",
                     {"scooter_energy_cost_usd_per_km": scooter_swap_low}, {"scooter_energy_cost_usd_per_km": scooter_swap_high}),
                    ("Grid carbon intensity", f"{grid_low:.2f} kgCO$_2$eq/kWh", f"{grid_high:.2f} kgCO$_2$eq/kWh",
                     {"grid_ci_kgco2_per_kwh": grid_low}, {"grid_ci_kgco2_per_kwh": grid_high}),
                    ("Discount rate", f"{discount_low:.0%}", f"{discount_high:.0%}",
                     {"discount_rate": discount_low}, {"discount_rate": discount_high}),
                    ("Battery replacement", "No replacement", "1 replacement",
                     {"replacement_count": 0}, {"replacement_count": 1}),
                    ("Annual VKT", f"{vkt_low:,.0f} km/yr", f"{vkt_high:,.0f} km/yr",
                     {"vkt": vkt_low}, {"vkt": vkt_high}),
                    ("Vehicle ownership/lifetime", f"{lifetime_low} years", f"{lifetime_high} years",
                     {"lifetime_years": lifetime_low}, {"lifetime_years": lifetime_high}),
                ]
            elif vehicle == "cars" and powertrain == "phev":
                variables = [
                    ("Electricity price", f"{elec_low:.1f} NT$/kWh", f"{elec_high:.1f} NT$/kWh",
                     {"electricity_price_ntd_per_kwh": elec_low}, {"electricity_price_ntd_per_kwh": elec_high}),
                    ("Gasoline price", f"{gas_low:.1f} NT$/L", f"{gas_high:.1f} NT$/L",
                     {"gasoline_price_ntd_per_l": gas_low}, {"gasoline_price_ntd_per_l": gas_high}),
                    ("Grid carbon intensity", f"{grid_low:.2f} kgCO$_2$eq/kWh", f"{grid_high:.2f} kgCO$_2$eq/kWh",
                     {"grid_ci_kgco2_per_kwh": grid_low}, {"grid_ci_kgco2_per_kwh": grid_high}),
                    ("PHEV utility factor", f"{uf_low:.1f}", f"{uf_high:.1f}",
                     {"utility_factor": uf_low}, {"utility_factor": uf_high}),
                    ("Battery replacement", "No replacement", "1 replacement",
                     {"replacement_count": 0}, {"replacement_count": 1}),
                    ("Battery salvage", f"{salvage_low:.0f} NT$/kWh", f"{salvage_high:.0f} NT$/kWh",
                     {"battery_salvage_ntd_per_kwh": salvage_low}, {"battery_salvage_ntd_per_kwh": salvage_high}),
                    ("Vehicle ownership/lifetime", f"{lifetime_low} years", f"{lifetime_high} years",
                     {"lifetime_years": lifetime_low}, {"lifetime_years": lifetime_high}),
                ]
            elif powertrain == "electric":
                price_low, price_high = elec_low, elec_high
                variables = [
                    ("Electricity price", f"{price_low:.1f} NT$/kWh", f"{price_high:.1f} NT$/kWh",
                     {"electricity_price_ntd_per_kwh": price_low}, {"electricity_price_ntd_per_kwh": price_high}),
                    ("Grid carbon intensity", f"{grid_low:.2f} kgCO$_2$eq/kWh", f"{grid_high:.2f} kgCO$_2$eq/kWh",
                     {"grid_ci_kgco2_per_kwh": grid_low}, {"grid_ci_kgco2_per_kwh": grid_high}),
                    ("Discount rate", f"{discount_low:.0%}", f"{discount_high:.0%}",
                     {"discount_rate": discount_low}, {"discount_rate": discount_high}),
                    ("Battery replacement", "No replacement", "1 replacement",
                     {"replacement_count": 0}, {"replacement_count": 1}),
                    ("Annual VKT", f"{vkt_low:,.0f} km/yr", f"{vkt_high:,.0f} km/yr",
                     {"vkt": vkt_low}, {"vkt": vkt_high}),
                    ("Vehicle ownership/lifetime", f"{lifetime_low} years", f"{lifetime_high} years",
                     {"lifetime_years": lifetime_low}, {"lifetime_years": lifetime_high}),
                    ("Battery salvage", f"{salvage_low:.0f} NT$/kWh", f"{salvage_high:.0f} NT$/kWh",
                     {"battery_salvage_ntd_per_kwh": salvage_low}, {"battery_salvage_ntd_per_kwh": salvage_high}),
                ]
            else:
                prod_low, prod_high = map(
                    float, self.cfg["hydrogen_production_ntd_per_kg"][pathway]
                )
                h2_price_low = prod_low + self.h2_dispensing("Optimistic", year)
                h2_price_high = prod_high + self.h2_dispensing("Conservative", year)
                h2_ci_low, h2_ci_high = map(
                    float, sens["h2_ci_kgco2_per_kg_2050"][pathway]
                )
                variables = [
                    ("Delivered H$_2$ price", f"{h2_price_low:.1f} NT$/kg", f"{h2_price_high:.1f} NT$/kg",
                     {"h2_delivered_price_ntd_per_kg": h2_price_low}, {"h2_delivered_price_ntd_per_kg": h2_price_high}),
                    ("H$_2$ carbon intensity", f"{h2_ci_low:.1f} kgCO$_2$eq/kg", f"{h2_ci_high:.1f} kgCO$_2$eq/kg",
                     {"h2_ci_kgco2_per_kg": h2_ci_low}, {"h2_ci_kgco2_per_kg": h2_ci_high}),
                    ("Discount rate", f"{discount_low:.0%}", f"{discount_high:.0%}",
                     {"discount_rate": discount_low}, {"discount_rate": discount_high}),
                    ("Battery replacement", "No replacement", "1 replacement",
                     {"replacement_count": 0}, {"replacement_count": 1}),
                    ("Annual VKT", f"{vkt_low:,.0f} km/yr", f"{vkt_high:,.0f} km/yr",
                     {"vkt": vkt_low}, {"vkt": vkt_high}),
                    ("Vehicle ownership/lifetime", f"{lifetime_low} years", f"{lifetime_high} years",
                     {"lifetime_years": lifetime_low}, {"lifetime_years": lifetime_high}),
                ]

            for variable, low_label, high_label, low_kwargs, high_kwargs in variables:
                low_result = self._sensitivity_avoided_cost(
                    vehicle, powertrain, year, scenario, pathway, **low_kwargs
                )
                high_result = self._sensitivity_avoided_cost(
                    vehicle, powertrain, year, scenario, pathway, **high_kwargs
                )
                finite = [x for x in (low_result, high_result) if np.isfinite(x)]
                left = min(finite) if finite else float("nan")
                right = max(finite) if finite else float("nan")
                rows.append({
                    "technology": title,
                    "vehicle": vehicle,
                    "powertrain": powertrain,
                    "pathway": pathway,
                    "year": year,
                    "scenario": scenario,
                    "baseline_ntd_per_tco2": baseline,
                    "variable": variable,
                    "low_input": low_label,
                    "high_input": high_label,
                    "low_result_ntd_per_tco2": low_result,
                    "high_result_ntd_per_tco2": high_result,
                    "left_edge_ntd_per_tco2": left,
                    "right_edge_ntd_per_tco2": right,
                    "range_width_ntd_per_tco2": right - left if finite else float("nan"),
                })

        return pd.DataFrame(rows)

    def plot_tornado(self, year: int = 2050, scenario: str = "Reference"):
        """Plot deterministic one-at-a-time cost-effectiveness sensitivities.

        The figure contains eight panels: electric scooter, PHEV car, electric
        car, electric light truck, electric bus, electric heavy truck, blue-H2
        truck, and green-H2 truck. Green/orange floating bars show the lower and
        higher result sides relative to the black reference line. Endpoint text
        reports the actual sensitivity values, so separate low/high-input legend
        entries are intentionally omitted.
        """
        sens = self.cfg["deterministic_sensitivity"]
        threshold = float(sens["abatement_threshold_ntd_per_tco2"])
        dac = float(sens["dac_usd_per_tco2"]) * self.fx
        df = self.tornado_sensitivity_data(year=year, scenario=scenario)

        variable_labels = {
            "Battery swapping cost": "Battery swapping\ncost",
            "Electricity price": "Electricity\nprice",
            "Gasoline price": "Gasoline\nprice",
            "Grid carbon intensity": "Grid carbon\nintensity",
            "PHEV utility factor": "PHEV utility\nfactor",
            "Discount rate": "Discount\nrate",
            "Battery replacement": "Battery\nreplacement",
            "Battery salvage": "Battery\nsalvage",
            "Annual VKT": "Annual VKT",
            "Vehicle ownership/lifetime": "Vehicle ownership/\nlifetime",
            "Delivered H$_2$ price": "Delivered H$_2$\nprice",
            "H$_2$ carbon intensity": "H$_2$ carbon\nintensity",
        }
        panel_order = [
            "Electric scooter", "PHEV car", "Electric car", "Electric light truck",
            "Electric bus", "Electric heavy truck", "Blue H$_2$ truck", "Green H$_2$ truck",
        ]
        green = "#2ca02c"
        orange = "#ff7f0e"

        def format_endpoint(label: str) -> str:
            label = str(label)
            if label == "No replacement":
                return "No\nreplacement"
            if label == "1 replacement":
                return "1\nreplacement"
            for unit in (
                "kgCO$_2$eq/kWh", "kgCO$_2$eq/kg", "NT$/kWh",
                "NT$/L", "NT$/kg", "USD/km", "km/yr",
            ):
                token = " " + unit
                if token in label:
                    label = label.replace(token, "\n" + unit)
                    break
            return label.replace("NT$/", "NT\\$/")

        fig, axes = plt.subplots(4, 2, figsize=(21.6, 25.8), dpi=300)
        fig.subplots_adjust(
            left=0.125, right=0.985, top=0.956, bottom=0.052,
            wspace=0.34, hspace=0.62,
        )

        for ax, tech in zip(axes.flatten(), panel_order):
            sub = df[df["technology"] == tech].copy()
            sub = sub.sort_values("range_width_ntd_per_tco2", ascending=False).reset_index(drop=True)
            baseline = float(sub["baseline_ntd_per_tco2"].iloc[0])
            y = np.arange(len(sub), dtype=float) * 1.85
            left = sub["left_edge_ntd_per_tco2"].to_numpy(dtype=float)
            right = sub["right_edge_ntd_per_tco2"].to_numpy(dtype=float)
            low_result = sub["low_result_ntd_per_tco2"].to_numpy(dtype=float)
            high_result = sub["high_result_ntd_per_tco2"].to_numpy(dtype=float)

            for yi, lo, hi in zip(y, left, right):
                if not (np.isfinite(lo) and np.isfinite(hi)):
                    continue
                if baseline > lo:
                    ax.barh(yi, baseline - lo, left=lo, height=0.48, color=green, edgecolor="none")
                if hi > baseline:
                    ax.barh(yi, hi - baseline, left=baseline, height=0.48, color=orange, edgecolor="none")

            ax.axvline(baseline, color="black", linewidth=2.0, zorder=2)
            ax.axvline(threshold, color="black", linewidth=1.1, linestyle=":", zorder=1)
            ax.axvline(dac, color="black", linewidth=1.1, linestyle="--", zorder=1)

            ax.set_yticks(y)
            ax.set_yticklabels(
                [variable_labels.get(v, v) for v in sub["variable"]],
                fontweight="bold", fontsize=16.0,
            )
            ax.invert_yaxis()
            ax.grid(axis="x", linestyle="--", alpha=0.28)
            ax.tick_params(axis="x", width=1.4, labelsize=15.0)
            ax.tick_params(axis="y", width=1.4, labelsize=16.0)
            for spine in ax.spines.values():
                spine.set_linewidth(1.4)

            finite_vals = np.r_[
                left[np.isfinite(left)], right[np.isfinite(right)],
                [0.0, threshold, dac, baseline],
            ]
            xmin = float(np.min(finite_vals))
            xmax = float(np.max(finite_vals))
            data_span = max(xmax - xmin, 1.0)
            data_pad = 0.11 * data_span
            label_pad = 0.32 * data_span
            xleft = xmin - data_pad - label_pad
            xright = xmax + data_pad + label_pad
            ax.set_xlim(xleft, xright)

            panel_span = xright - xleft
            low_label_x = xleft + 0.035 * panel_span
            high_label_x = xright - 0.035 * panel_span
            leader_style = dict(
                color="0.35", linewidth=0.8, alpha=0.55,
                linestyle="-", zorder=3, clip_on=False,
            )

            for i, row in sub.iterrows():
                lo_res = float(row["low_result_ntd_per_tco2"])
                hi_res = float(row["high_result_ntd_per_tco2"])
                if not (np.isfinite(lo_res) and np.isfinite(hi_res)):
                    continue

                low_label = format_endpoint(row["low_input"])
                high_label = format_endpoint(row["high_input"])
                yi = y[i]

                low_x = low_label_x
                high_x = high_label_x
                label_y = yi
                leader_gap = 0.018 * panel_span
                if abs(lo_res - low_x) > leader_gap:
                    ax.plot([lo_res, low_x + leader_gap], [yi, label_y], **leader_style)
                if abs(hi_res - high_x) > leader_gap:
                    ax.plot([hi_res, high_x - leader_gap], [yi, label_y], **leader_style)

                ax.text(
                    low_x, label_y, low_label,
                    va="center", ha="left", fontsize=14.0, fontweight="bold",
                    linespacing=0.98,
                    bbox=dict(facecolor="white", edgecolor="0.85", linewidth=0.25, alpha=0.92, pad=0.9),
                    zorder=5,
                )
                ax.text(
                    high_x, label_y, high_label,
                    va="center", ha="right", fontsize=14.0, fontweight="bold",
                    linespacing=0.98,
                    bbox=dict(facecolor="white", edgecolor="0.85", linewidth=0.25, alpha=0.92, pad=0.9),
                    zorder=5,
                )

            ax.set_title(tech, fontsize=21.5, fontweight="bold", pad=9)
            ax.set_xlabel(
                r"Cost of CO$_2$ avoided" + "\n" + r"(NT\$/tCO$_2$eq)",
                fontweight="bold", fontsize=17.0,
            )

        legend = [
            Patch(facecolor=green, label="Lower result side"),
            Patch(facecolor=orange, label="Higher result side"),
            Line2D([0], [0], color="black", lw=2.0, label="Reference"),
            Line2D([0], [0], color="black", lw=1.1, linestyle=":", label=r"5,000 NT\$/tCO$_2$eq"),
            Line2D([0], [0], color="black", lw=1.1, linestyle="--", label="DAC"),
        ]
        fig.legend(
            handles=legend, loc="upper center", ncol=5, frameon=False,
            fontsize=22.0, bbox_to_anchor=(0.5, 0.996),
        )
        self._savefig(fig, "tornado_sensitivity")
        plt.close(fig)
        self._save_table(df, "deterministic_tornado_sensitivity.csv")


    # ---------------------------- fleet model ------------------------------
    def sales_share(self, vehicle: str, year: int) -> float:
        anchors=SALES_SHARE_ANCHORS[vehicle]; x=np.array(sorted(anchors)); y=np.array([anchors[k] for k in x]); return float(np.interp(year,x,y))

    def fleet_stock(self, vehicle: str, year: int) -> Tuple[float,float]:
        # Historical years use observed electric registrations directly.
        if year <= 2024:
            ev=float(OBSERVED_EV[vehicle][year]); return float(FLEET_SIZE[vehicle]-ev),ev
        # 2024 stock is uniformly distributed across ages 0–15, separately by technology.
        ev24=float(OBSERVED_EV[vehicle][2024]); conv24=float(FLEET_SIZE[vehicle]-ev24)
        ev=0.0; conv=0.0
        for birth in range(2009,2025):
            if year-birth < self.lifetime:
                ev += ev24/self.lifetime; conv += conv24/self.lifetime
        for birth in range(2025,year+1):
            if year-birth < self.lifetime:
                n=float(ANNUAL_SALES[vehicle]); sh=self.sales_share(vehicle,birth); ev += n*sh; conv += n*(1-sh)
        return conv,ev

    def plot_fleet(self):
        years_em=np.arange(2025,2051); years_en=np.arange(2025,2051); scenarios=['Conservative','Reference','Optimistic']; rng=np.random.default_rng(self.seed)
        # shared MC efficiency quantiles for consistent cross-year traces
        u={}
        for v in FLEET_SIZE:
            for pt in set([('diesel' if v in ('bus','trucks') else 'gasoline'),'electric']): u[(v,pt)]=self._triangular(rng,self.n_mc)
        em_rows=[]; en_rows=[]
        for sc in scenarios:
            for yr in years_em:
                comps={}; draws=np.zeros(self.n_mc)
                for v in ['trucks','cars','scooters','trucks_light','bus']:
                    ref='diesel' if v in ('bus','trucks') else 'gasoline'; conv,ev=self.fleet_stock(v,int(yr)); vkt=FLEET_VKT[v]
                    yr_factor=int(yr)
                    bmid=self._bundle(v,ref,.5); emid=self._bundle(v,'electric',.5); fref=self.factor(sc,v,ref,yr_factor,'energy_use_factor'); fev=self.factor(sc,v,'electric',yr_factor,'energy_use_factor')
                    ref_eff=float(bmid['fuel_efficiency'])*fref; ev_eff=float(emid['fuel_efficiency'])*fev
                    if ref=='diesel': ref_ci=ref_eff/float(self.ec['diesel_kwh_per_l'])*float(self.ec['diesel_ci_kgco2_per_l'])
                    else: ref_ci=ref_eff/float(self.ec['gasoline_kwh_per_l'])*float(self.ec['gasoline_ci_kgco2_per_l'])
                    gci=self.grid_ci(sc,yr_factor); ev_ci=ev_eff*gci
                    ice_mt=conv*vkt*ref_ci/1e9; ev_mt=ev*vkt*ev_ci/1e9; comps[f'{v}_ICE']=ice_mt; comps[f'{v}_EV']=ev_mt
                    br=self._bundle(v,ref,u[(v,ref)]); be=self._bundle(v,'electric',u[(v,'electric')]); re=np.asarray(br['fuel_efficiency'])*fref; ee=np.asarray(be['fuel_efficiency'])*fev
                    if ref=='diesel': rci=re/float(self.ec['diesel_kwh_per_l'])*float(self.ec['diesel_ci_kgco2_per_l'])
                    else: rci=re/float(self.ec['gasoline_kwh_per_l'])*float(self.ec['gasoline_ci_kgco2_per_l'])
                    draws += conv*vkt*rci/1e9 + ev*vkt*ee*gci/1e9
                q05,q50,q95=self.q(draws); em_rows.append({'scenario':sc,'year':yr,**comps,'p05':q05,'median':q50,'p95':q95})
            for yr in years_en:
                comps={}; draws=np.zeros(self.n_mc)
                for v in ['cars','scooters','trucks_light','bus','trucks']:
                    _,ev=self.fleet_stock(v,int(yr)); vkt=FLEET_VKT[v]; f=self.factor(sc,v,'electric',int(yr),'energy_use_factor'); mid=float(self._bundle(v,'electric',.5)['fuel_efficiency'])*f; comps[v]=ev*vkt*mid/1e9; dr=np.asarray(self._bundle(v,'electric',u[(v,'electric')])['fuel_efficiency'])*f; draws += ev*vkt*dr/1e9
                q05,q50,q95=self.q(draws); en_rows.append({'scenario':sc,'year':yr,**comps,'p05':q05,'median':q50,'p95':q95})
        em=pd.DataFrame(em_rows); en=pd.DataFrame(en_rows)
        order=[('trucks','ICE','brown','ICE Heavy Trucks'),('trucks','EV','lightblue','EV Heavy Trucks'),('cars','ICE','red','ICE Cars'),('cars','EV','pink','EV Cars'),('scooters','ICE','purple','ICE Scooters'),('scooters','EV','lightgreen','EV Scooters'),('trucks_light','ICE','orange','ICE Light Trucks'),('trucks_light','EV','navajowhite','EV Light Trucks'),('bus','ICE','darkgreen','ICE Buses'),('bus','EV','lightseagreen','EV Buses')]
        em_ymax = max(3.3, float(em["p95"].max()))
        em_ymax = float(np.ceil(em_ymax * 1.08 / 5.0) * 5.0)
        fig,axes=plt.subplots(1,3,figsize=(21.0,5.8),dpi=300,sharey=True); fig.subplots_adjust(left=.06,right=.83,top=.80,bottom=.16,wspace=.08)
        for ax,sc in zip(axes,scenarios):
            sub=em[em.scenario==sc].sort_values('year'); x=sub.year.to_numpy(); bottom=np.zeros(len(sub))
            for v,kind,col,lab in order:
                vals=sub[f'{v}_{kind}'].to_numpy(); ax.bar(x,vals,bottom=bottom,width=.82,color=col,linewidth=0); bottom+=vals
            med=sub['median'].to_numpy(); lo=med-sub['p05'].to_numpy(); hi=sub['p95'].to_numpy()-med; ax.errorbar(x,med,yerr=[lo,hi],fmt='D',color='black',markersize=2,elinewidth=.7,capsize=1.5,zorder=20)
            ax.axhline(3.3,ls='-.',color='black',lw=1.8); ax.set_title(sc,fontsize=20.5,fontweight='bold'); ax.set_xlabel('Year',fontsize=16,fontweight='bold'); ax.grid(axis='y',ls='--',alpha=.7); ax.set_xlim(2024.5,2050.5); ax.set_ylim(0,em_ymax); ax.set_xticks(np.arange(2025,2051,5)); ax.tick_params(labelsize=13.5,width=2)
            for sp in ax.spines.values(): sp.set_linewidth(2)
        axes[0].set_ylabel('Total Operating Emissions\n(MMt CO$_2$eq)',fontsize=17,fontweight='bold'); handles=[Patch(facecolor=c,label=l) for _,_,c,l in order]+[Line2D([0],[0],color='black',ls='-.',label='2050 Goal'),Line2D([0],[0],marker='D',color='black',lw=0,markersize=4,label='MC median (P5–P95)')]; fig.legend(handles=handles,bbox_to_anchor=(.835,.80),loc='upper left',frameon=False,fontsize=12.4)
        self._savefig(fig,'total_emissions'); plt.close(fig)
        colors={'cars':'lightblue','scooters':'pink','trucks_light':'lightgreen','bus':'navajowhite','trucks':'lightseagreen'}; labels={'cars':'Cars','scooters':'Scooters','trucks_light':'Light Trucks','bus':'Buses','trucks':'Heavy Trucks'}; ymax=max(30,float(np.ceil(en.p95.max()/5)*5))
        fig,axes=plt.subplots(1,3,figsize=(21.0,5.8),dpi=300,sharey=True); fig.subplots_adjust(left=.06,right=.96,top=.81,bottom=.16,wspace=.08)
        for ax,sc in zip(axes,scenarios):
            sub=en[en.scenario==sc].sort_values('year'); x=sub.year.to_numpy(); bottom=np.zeros(len(sub))
            for v in ['cars','scooters','trucks_light','bus','trucks']:
                vals=sub[v].to_numpy(); ax.bar(x,vals,bottom=bottom,width=.82,color=colors[v],linewidth=0); bottom+=vals
            med=sub.median.to_numpy() if False else sub['median'].to_numpy(); lo=med-sub.p05.to_numpy(); hi=sub.p95.to_numpy()-med; ax.errorbar(x,med,yerr=[lo,hi],fmt='D',color='black',markersize=2,elinewidth=.7,capsize=1.5,zorder=20)
            ax.set_title(sc,fontsize=20.5,fontweight='bold'); ax.set_xlabel('Year',fontsize=16,fontweight='bold'); ax.grid(axis='y',ls='--',alpha=.7); ax.set_xlim(2024.5,2050.5); ax.set_ylim(0,ymax); ax.set_xticks(np.arange(2025,2051,5)); ax.tick_params(labelsize=13.5,width=2)
            for sp in ax.spines.values(): sp.set_linewidth(2)
        axes[0].set_ylabel('Electricity demand (TWh)',fontsize=17,fontweight='bold'); handles=[Patch(facecolor=colors[v],label=labels[v]) for v in colors]+[Line2D([0],[0],marker='D',color='black',lw=0,markersize=4,label='MC median (P5–P95)')]; fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.98),ncol=6,frameon=False,fontsize=12.8)
        self._savefig(fig,'energy_demand'); plt.close(fig); self._save_table(em,'fleet_emissions_scenarios_MC.csv'); self._save_table(en,'fleet_energy_scenarios_MC.csv')


    def plot_charging_infrastructure(self):
        """Charging infrastructure with independent technology and charging assumptions.

        Passenger cars use residential plus workplace/public charging. Their
        non-residential electricity is split between 7-kW L2 and 50-kW DCFC.
        Light trucks use 50-kW depot chargers. Buses and heavy trucks use
        150-kW depot chargers. Vehicle technology assumptions determine annual
        electricity demand; charging assumptions independently determine
        residential access, charger utilization, and depot charger cost.

        The main figure shows only the matched Reference technology + Reference
        charging case. A 3 x 3 factorial table reports all combinations in 2050.
        """
        plt.rcParams.update({
            'font.weight':'bold','axes.labelweight':'bold',
            'axes.titleweight':'bold','axes.linewidth':2,
            'xtick.major.width':1.8,'ytick.major.width':1.8,
        })

        dc = self.cfg['depot_charging']
        pc = self.cfg['public_charging']
        years = np.arange(2025,2051)
        vehicle_cases = ['Conservative','Reference','Optimistic']
        charging_cases = ['Conservative','Reference','Optimistic']

        public_life = int(pc['charger_lifetime_years'])
        depot_life = int(dc['charger_lifetime_years'])
        l2_share = float(pc['nonresidential_l2_energy_share'])
        dcfc_share = float(pc['dcfc_energy_share'])
        if not np.isclose(l2_share + dcfc_share, 1.0):
            raise ValueError('Public L2 and DCFC energy shares must sum to 1.')

        l2_power = float(pc['level2_power_kw'])
        dcfc_power = float(pc['dcfc_power_kw'])
        lt_depot_power = float(dc.get('light_truck_charger_power_kw', 50.0))
        hd_depot_power = float(dc['charger_power_kw'])
        depot_window_h = float(dc['overnight_window_hours'])

        l2_cost = float(pc['level2_installed_cost_ntd'])
        dcfc_cost = float(pc['dcfc_installed_cost_ntd'])
        lt_depot_cost = float(dc.get('light_truck_installed_cost_ntd', dcfc_cost))

        observed_public_2024 = float(pc['observed_public_ports_2024'])
        initial_l2_public = float(pc.get('observed_public_l2_ports_2024', observed_public_2024 * (1.0-dcfc_share)))
        initial_dcfc_public = float(pc.get('observed_public_dcfc_ports_2024', observed_public_2024 * dcfc_share))
        if not np.isclose(initial_l2_public + initial_dcfc_public, observed_public_2024):
            raise ValueError('Observed 2024 L2 + DCFC ports must equal total public ports.')

        def charger_turnover(years_, required_stock, initial_year, initial_stock, lifetime):
            """Exact charger-cohort turnover with an initial uniform age distribution."""
            years_ = np.asarray(years_, dtype=int)
            required_stock = np.asarray(required_stock, dtype=float)
            cohorts = {
                initial_year-age: float(initial_stock)/lifetime
                for age in range(lifetime)
            }
            in_service=[]; additions=[]; replacements=[]; expansions=[]; retirements=[]
            for y, target in zip(years_, required_stock):
                retired = sum(n for birth,n in cohorts.items() if y-birth >= lifetime)
                cohorts = {birth:n for birth,n in cohorts.items() if y-birth < lifetime}
                survivors = sum(cohorts.values())
                add = max(float(target)-survivors, 0.0)
                repl = min(retired, add)
                expand = max(add-repl, 0.0)
                if add > 0:
                    cohorts[int(y)] = cohorts.get(int(y), 0.0) + add
                in_service.append(survivors+add)
                additions.append(add); replacements.append(repl)
                expansions.append(expand); retirements.append(retired)
            return {
                'in_service':np.asarray(in_service),
                'additions':np.asarray(additions),
                'replacements':np.asarray(replacements),
                'expansions':np.asarray(expansions),
                'retirements':np.asarray(retirements),
            }

        def annual_ev_energy(vehicle, vehicle_case, year):
            _, ev = self.fleet_stock(vehicle, int(year))
            eff = float(self._bundle(vehicle,'electric',0.5)['fuel_efficiency']) * \
                self.factor(vehicle_case,vehicle,'electric',int(year),'energy_use_factor')
            return ev * FLEET_VKT[vehicle] * eff

        def initial_depot_stock(vehicle, vehicle_case, charging_case, power_kw):
            """Approximate 2024 depot stock from observed EVs and 2025 efficiency."""
            util = float(dc['scenarios'][charging_case]['utilization_of_12h_window'])
            annual_capacity = power_kw * depot_window_h * 365.0 * util
            ev24 = float(OBSERVED_EV[vehicle][2024])
            eff25 = float(self._bundle(vehicle,'electric',0.5)['fuel_efficiency']) * \
                self.factor(vehicle_case,vehicle,'electric',2025,'energy_use_factor')
            return ev24 * FLEET_VKT[vehicle] * eff25 / annual_capacity

        def calculate_combination(vehicle_case, charging_case):
            """Return time series for one technology x charging-assumption combination."""
            p_ass = pc['scenarios'][charging_case]
            d_ass = dc['scenarios'][charging_case]
            residential_share = float(p_ass['residential_energy_share'])
            l2_util = float(p_ass['level2_utilization'])
            dcfc_util = float(p_ass['dcfc_utilization'])
            depot_util = float(d_ass['utilization_of_12h_window'])

            cap_l2 = l2_power * 8760.0 * l2_util
            cap_dcfc = dcfc_power * 8760.0 * dcfc_util
            cap_lt = lt_depot_power * depot_window_h * 365.0 * depot_util
            cap_hd = hd_depot_power * depot_window_h * 365.0 * depot_util

            car_energy = np.asarray([annual_ev_energy('cars',vehicle_case,y) for y in years])
            lt_energy = np.asarray([annual_ev_energy('trucks_light',vehicle_case,y) for y in years])
            bus_energy = np.asarray([annual_ev_energy('bus',vehicle_case,y) for y in years])
            ht_energy = np.asarray([annual_ev_energy('trucks',vehicle_case,y) for y in years])

            car_nonres = (1.0-residential_share) * car_energy
            req_l2 = l2_share * car_nonres / cap_l2
            req_dcfc = dcfc_share * car_nonres / cap_dcfc
            req_lt = lt_energy / cap_lt
            req_bus = bus_energy / cap_hd
            req_ht = ht_energy / cap_hd

            turn_l2 = charger_turnover(years, req_l2, 2024, initial_l2_public, public_life)
            turn_dcfc = charger_turnover(years, req_dcfc, 2024, initial_dcfc_public, public_life)
            turn_lt = charger_turnover(
                years, req_lt, 2024,
                initial_depot_stock('trucks_light',vehicle_case,charging_case,lt_depot_power),
                depot_life,
            )
            turn_bus = charger_turnover(
                years, req_bus, 2024,
                initial_depot_stock('bus',vehicle_case,charging_case,hd_depot_power),
                depot_life,
            )
            turn_ht = charger_turnover(
                years, req_ht, 2024,
                initial_depot_stock('trucks',vehicle_case,charging_case,hd_depot_power),
                depot_life,
            )

            hd_installed_cost = (float(d_ass['hardware_usd']) + float(d_ass['installation_usd'])) * self.fx
            annual_inv_l2 = turn_l2['additions'] * l2_cost / 1e9
            annual_inv_dcfc = turn_dcfc['additions'] * dcfc_cost / 1e9
            annual_inv_lt = turn_lt['additions'] * lt_depot_cost / 1e9
            annual_inv_bus = turn_bus['additions'] * hd_installed_cost / 1e9
            annual_inv_ht = turn_ht['additions'] * hd_installed_cost / 1e9

            return {
                'vehicle_case':vehicle_case,
                'vehicle_trajectory':self.cfg['technology_trajectory_map'][vehicle_case],
                'charging_case':charging_case,
                'residential_share':residential_share,
                'l2_utilization':l2_util,
                'dcfc_utilization':dcfc_util,
                'depot_utilization':depot_util,
                'car_energy':car_energy,
                'lt_energy':lt_energy,
                'bus_energy':bus_energy,
                'ht_energy':ht_energy,
                'required_l2':req_l2,
                'required_dcfc':req_dcfc,
                'required_lt_depot':req_lt,
                'required_bus_depot':req_bus,
                'required_ht_depot':req_ht,
                'cumulative_inv_l2':np.cumsum(annual_inv_l2),
                'cumulative_inv_dcfc':np.cumsum(annual_inv_dcfc),
                'cumulative_inv_lt_depot':np.cumsum(annual_inv_lt),
                'cumulative_inv_bus_depot':np.cumsum(annual_inv_bus),
                'cumulative_inv_ht_depot':np.cumsum(annual_inv_ht),
                'installed_cost_hd_depot_ntd':hd_installed_cost,
            }

        # Delivered cost for 150-kW bus/heavy-truck depot charging.
        # Taipower's fixed customer charge is site-level and is not allocated per charger
        # because the number of chargers per site is not specified.
        depot_cost_rows=[]
        p_off=float(dc['taipower_offpeak_energy_ntd_per_kwh'])
        p_cap=float(dc['taipower_capacity_charge_ntd_per_kw_month'])
        eta=float(dc['charging_efficiency'])
        r_ch=float(dc['charger_discount_rate'])
        for charging_case in charging_cases:
            d_ass=dc['scenarios'][charging_case]
            util=float(d_ass['utilization_of_12h_window'])
            hw=float(d_ass['hardware_usd'])*self.fx
            inst=float(d_ass['installation_usd'])*self.fx
            cap=((hw+inst)*(1.0+r_ch)**depot_life /
                 (depot_life*hd_depot_power*depot_window_h*365.0*util))
            energy=p_off/eta
            capacity=(12.0*hd_depot_power*p_cap /
                      (hd_depot_power*depot_window_h*365.0*util))
            depot_cost_rows.append({
                'charging_assumption_case':charging_case,
                'hardware_cost_ntd':hw,
                'installation_cost_ntd':inst,
                'installed_cost_ntd':hw+inst,
                'utilization_of_12h_window':util,
                'capital_ntd_per_kwh':cap,
                'energy_ntd_per_kwh':energy,
                'capacity_ntd_per_kwh':capacity,
                'delivered_cost_ntd_per_kwh':cap+energy+capacity,
            })
        self._save_table(pd.DataFrame(depot_cost_rows),'depot_delivered_charging_costs.csv')

        # Full 3 x 3 factorial design separates technology from charging assumptions.
        combinations = {}
        detail_rows = []
        summary_rows = []
        for vehicle_case in vehicle_cases:
            for charging_case in charging_cases:
                d = calculate_combination(vehicle_case, charging_case)
                combinations[(vehicle_case,charging_case)] = d
                for i,y in enumerate(years):
                    total_ports = (d['required_l2'][i] + d['required_dcfc'][i] +
                                   d['required_lt_depot'][i] + d['required_bus_depot'][i] +
                                   d['required_ht_depot'][i])
                    total_inv = (d['cumulative_inv_l2'][i] + d['cumulative_inv_dcfc'][i] +
                                 d['cumulative_inv_lt_depot'][i] + d['cumulative_inv_bus_depot'][i] +
                                 d['cumulative_inv_ht_depot'][i])
                    detail_rows.append({
                        'vehicle_technology_case':vehicle_case,
                        'vehicle_technology_trajectory':d['vehicle_trajectory'],
                        'charging_assumption_case':charging_case,
                        'year':int(y),
                        'passenger_car_residential_energy_share':d['residential_share'],
                        'public_l2_utilization':d['l2_utilization'],
                        'public_dcfc_utilization':d['dcfc_utilization'],
                        'depot_utilization_of_12h_window':d['depot_utilization'],
                        'passenger_car_electricity_twh':d['car_energy'][i]/1e9,
                        'light_truck_electricity_twh':d['lt_energy'][i]/1e9,
                        'bus_electricity_twh':d['bus_energy'][i]/1e9,
                        'heavy_truck_electricity_twh':d['ht_energy'][i]/1e9,
                        'required_passenger_car_l2_ports':d['required_l2'][i],
                        'required_passenger_car_dcfc_ports':d['required_dcfc'][i],
                        'required_light_truck_50kw_depot_chargers':d['required_lt_depot'][i],
                        'required_bus_150kw_depot_chargers':d['required_bus_depot'][i],
                        'required_heavy_truck_150kw_depot_chargers':d['required_ht_depot'][i],
                        'required_total_chargers':total_ports,
                        'cumulative_passenger_car_l2_investment_billion_ntd':d['cumulative_inv_l2'][i],
                        'cumulative_passenger_car_dcfc_investment_billion_ntd':d['cumulative_inv_dcfc'][i],
                        'cumulative_light_truck_depot_investment_billion_ntd':d['cumulative_inv_lt_depot'][i],
                        'cumulative_bus_depot_investment_billion_ntd':d['cumulative_inv_bus_depot'][i],
                        'cumulative_heavy_truck_depot_investment_billion_ntd':d['cumulative_inv_ht_depot'][i],
                        'cumulative_total_investment_billion_ntd':total_inv,
                    })
                i=-1
                total_ports_2050 = (d['required_l2'][i] + d['required_dcfc'][i] +
                                    d['required_lt_depot'][i] + d['required_bus_depot'][i] +
                                    d['required_ht_depot'][i])
                total_inv_2050 = (d['cumulative_inv_l2'][i] + d['cumulative_inv_dcfc'][i] +
                                  d['cumulative_inv_lt_depot'][i] + d['cumulative_inv_bus_depot'][i] +
                                  d['cumulative_inv_ht_depot'][i])
                summary_rows.append({
                    'vehicle_technology_case':vehicle_case,
                    'vehicle_technology_trajectory':d['vehicle_trajectory'],
                    'charging_assumption_case':charging_case,
                    'passenger_car_residential_energy_share':d['residential_share'],
                    'public_l2_utilization':d['l2_utilization'],
                    'public_dcfc_utilization':d['dcfc_utilization'],
                    'depot_utilization_of_12h_window':d['depot_utilization'],
                    'passenger_car_l2_ports_2050':d['required_l2'][i],
                    'passenger_car_dcfc_ports_2050':d['required_dcfc'][i],
                    'light_truck_50kw_depot_chargers_2050':d['required_lt_depot'][i],
                    'bus_150kw_depot_chargers_2050':d['required_bus_depot'][i],
                    'heavy_truck_150kw_depot_chargers_2050':d['required_ht_depot'][i],
                    'total_chargers_2050':total_ports_2050,
                    'passenger_car_l2_investment_billion_ntd_2050':d['cumulative_inv_l2'][i],
                    'passenger_car_dcfc_investment_billion_ntd_2050':d['cumulative_inv_dcfc'][i],
                    'light_truck_depot_investment_billion_ntd_2050':d['cumulative_inv_lt_depot'][i],
                    'bus_depot_investment_billion_ntd_2050':d['cumulative_inv_bus_depot'][i],
                    'heavy_truck_depot_investment_billion_ntd_2050':d['cumulative_inv_ht_depot'][i],
                    'total_cumulative_investment_billion_ntd_2050':total_inv_2050,
                })

        # Main figure: matched Reference technology + Reference charging only.
        ref = combinations[('Reference','Reference')]
        components = [
            ('required_l2','cumulative_inv_l2','Passenger car L2'),
            ('required_dcfc','cumulative_inv_dcfc','Passenger car DCFC'),
            ('required_lt_depot','cumulative_inv_lt_depot','Light-truck depot (50 kW)'),
            ('required_bus_depot','cumulative_inv_bus_depot','Bus depot (150 kW)'),
            ('required_ht_depot','cumulative_inv_ht_depot','Heavy-truck depot (150 kW)'),
        ]

        fig,axes=plt.subplots(1,2,figsize=(19.4,6.4),dpi=300)
        fig.subplots_adjust(left=.07,right=.985,bottom=.16,top=.79,wspace=.34)

        ax=axes[0]
        bottom=np.zeros(len(years),dtype=float)
        handles=[]
        for req_key,_,label in components:
            vals=ref[req_key]/1000.0
            bars=ax.bar(years,vals,bottom=bottom,width=.82,label=label,linewidth=0)
            handles.append(Patch(facecolor=bars[0].get_facecolor(),label=label))
            bottom += vals
        ax.set_xlim(2024.5,2050.5); ax.set_xticks(np.arange(2025,2051,5))
        ax.set_ylim(0,np.ceil(bottom.max()*1.10/50.0)*50.0)
        ax.set_xlabel('Year',fontsize=16,fontweight='bold')
        ax.set_ylabel('Required charger stock\n(thousand ports)',fontsize=16,fontweight='bold',multialignment='center')
        ax.tick_params(axis='both',labelsize=13.5,width=1.8)
        ax.grid(axis='y',ls='--',alpha=.45)
        ax.text(0.015,0.975,'(a)',transform=ax.transAxes,ha='left',va='top',fontsize=18,fontweight='bold')
        for sp in ax.spines.values(): sp.set_linewidth(2)

        ax=axes[1]
        bottom=np.zeros(len(years),dtype=float)
        for _,inv_key,label in components:
            vals=ref[inv_key]
            ax.bar(years,vals,bottom=bottom,width=.82,label=label,linewidth=0)
            bottom += vals
        ax.set_xlim(2024.5,2050.5); ax.set_xticks(np.arange(2025,2051,5))
        ax.set_ylim(0,np.ceil(bottom.max()*1.10/50.0)*50.0)
        ax.set_xlabel('Year',fontsize=16,fontweight='bold')
        ax.set_ylabel('Cumulative investment since 2025\n(Billion NT$)',fontsize=16,fontweight='bold',multialignment='center')
        ax.tick_params(axis='both',labelsize=13.5,width=1.8)
        ax.grid(axis='y',ls='--',alpha=.45)
        ax.text(0.015,0.975,'(b)',transform=ax.transAxes,ha='left',va='top',fontsize=18,fontweight='bold')
        for sp in ax.spines.values(): sp.set_linewidth(2)

        legend_labels=[label for _,_,label in components]
        top_legend=fig.legend(handles[:3],legend_labels[:3],loc='upper center',
                              bbox_to_anchor=(0.53,0.985),ncol=3,frameon=False,fontsize=12.3)
        fig.add_artist(top_legend)
        fig.legend(handles[3:],legend_labels[3:],loc='upper center',
                   bbox_to_anchor=(0.53,0.900),ncol=2,frameon=False,fontsize=12.3)
        self._savefig(fig,'chargers')
        plt.close(fig)

        # Remove the legacy split investment figure to avoid stale duplicates.
        for d in self.fig_dirs:
            for ext in ('png','pdf'):
                stale=d/f'chargers_investment.{ext}'
                if stale.exists():
                    stale.unlink()

        self._save_table(pd.DataFrame(detail_rows),'charging_infrastructure_all_combinations_timeseries.csv')
        self._save_table(pd.DataFrame(summary_rows),'charging_infrastructure_2050_9_combinations.csv')
        ref_rows = [r for r in detail_rows if r['vehicle_technology_case']=='Reference' and r['charging_assumption_case']=='Reference']
        self._save_table(pd.DataFrame(ref_rows),'charging_infrastructure_reference_timeseries.csv')
        assumption_rows=[]
        for case in charging_cases:
            pa=pc['scenarios'][case]; da=dc['scenarios'][case]
            assumption_rows.append({
                'charging_assumption_case':case,
                'passenger_car_residential_energy_share':float(pa['residential_energy_share']),
                'passenger_car_l2_utilization':float(pa['level2_utilization']),
                'passenger_car_dcfc_utilization':float(pa['dcfc_utilization']),
                'light_truck_bus_heavy_truck_depot_utilization_of_12h_window':float(da['utilization_of_12h_window']),
                'passenger_car_nonresidential_l2_energy_share':l2_share,
                'passenger_car_nonresidential_dcfc_energy_share':dcfc_share,
                'passenger_car_l2_power_kw':l2_power,
                'passenger_car_dcfc_power_kw':dcfc_power,
                'light_truck_depot_power_kw':lt_depot_power,
                'bus_heavy_truck_depot_power_kw':hd_depot_power,
                'passenger_car_l2_installed_cost_ntd_per_port':l2_cost,
                'passenger_car_dcfc_installed_cost_ntd_per_port':dcfc_cost,
                'light_truck_depot_installed_cost_ntd_per_port':lt_depot_cost,
                'bus_heavy_truck_hardware_cost_ntd_per_port':float(da['hardware_usd'])*self.fx,
                'bus_heavy_truck_installation_cost_ntd_per_port':float(da['installation_usd'])*self.fx,
                'public_charger_lifetime_years':public_life,
                'depot_charger_lifetime_years':depot_life,
            })
        self._save_table(pd.DataFrame(assumption_rows),'charging_infrastructure_assumptions.csv')

    def generate_all(self):
        self.save_scooter_swapping_trajectory()
        self.plot_lce()
        self.plot_tco()
        self.plot_lce_vs_tco()
        self.plot_avoided_cost()
        self.plot_contours()
        self.plot_tornado()
        self.plot_fleet()
        self.plot_charging_infrastructure()


def generate_figures(repo_root: str | Path = ".") -> TigerModel:
    model = TigerModel(repo_root)
    model.generate_all()
    return model
