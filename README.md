# Integrated cost-effectiveness and life-cycle emissions analysis of road transport decarbonization in Taiwan

This repository contains the code and data supporting the study **“Integrated cost-effectiveness and life-cycle emissions analysis of road transport decarbonization in Taiwan”**, developed as part of Taiwan’s Innovative Green Economy Roadmap (TIGER).

The model evaluates life-cycle greenhouse-gas emissions (LCE), total cost of ownership (TCO), cost of CO₂ avoided, fleet turnover, electricity demand, and charging-infrastructure requirements for scooters, passenger cars, buses, light trucks, and heavy trucks.

<img width="845" height="330" alt="TOC" src="https://github.com/user-attachments/assets/f07e8a82-50fa-4087-a4ca-6ed784c90bb3" />

## Reproducing the analysis

Install the required Python packages:

```bash
pip install numpy pandas matplotlib
```

From the repository root, regenerate the main figures and numerical outputs with:

```bash
python scripts/regenerate_figures.py
```

Figures are written to `figs/` and `figures/`, and numerical outputs are written to `results/`.

## Key model assumptions

Future vehicle technology is represented using three scenarios:

| Study scenario | Whole-vehicle trajectory |
|---|---|
| Conservative | Static 2025 vehicle cost and energy use |
| Reference | ATB Baseline |
| Optimistic | ATB Advanced |

Scooters are not included in the ATB and were thus modeled seperately.

Passenger-car residential charging-energy shares are **30% / 40% / 50%** in the Conservative / Reference / Optimistic charging cases, with the remaining assigned to public L2 or DCFC. Light trucks use 50-kW depot charging; buses and heavy trucks use 150-kW depot charging.

## Repository structure

```text
data/       Model inputs and technology trajectorie
src/        Main model implementation
scripts/    Reproduction scripts
results/    Generated numerical outputs
figs/       Generated figures
```

The main model is implemented in:

```text
src/tiger_model.py
```

## Citation

If you use this repository, please cite:

> Emad Al Ibrahim, Kariana Moreno-Sader, Yu-Chi Kao, Deepjyoti Deka, Yi-Pei Li, and William H. Green. **Integrated cost-effectiveness and life-cycle emissions analysis of road transport decarbonization in Taiwan.**

Please refer to the manuscript and Supplementary Information for the complete methodology, data sources, assumptions, and limitations.
