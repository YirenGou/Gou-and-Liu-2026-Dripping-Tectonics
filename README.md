# Eclogite-pyroxenite drips and localized lithosphere thinning under the Central Andes

Yiren Gou and Mian Liu. Submitted to Nature Communications.

This repository contains the modified I2VIS function, the model initial conditions, and the figure source data for that manuscript. Display time in the figures is model time minus 50 Myr.

The experiments in `init/` and `plotting/` are the runs drawn in the 8 October 2026 submission. They were taken from the plotting scripts that produced those figures.

## Modified solver function

`Flow_Laws.c` is the flow-law block for dislocation creep and grain-boundary sliding, including grain size and water content. Insert it in `viscalc`, as marked in the file. The I2VIS method is Gerya and Yuen (2003, *Physics of the Earth and Planetary Interiors*). The solver download cited in the manuscript is Gerya, Bercovici, and Becker (2021, *Nature*).

## Initial conditions of the submitted figures

Each file in `init/` is the `init.t3c` of one experiment, named by the published figure. `MANIFEST.tsv` records the size, SHA-256 checksum, and the working folder each deposited file was copied from.

| File | Published figure | Experiment |
| --- | --- | --- |
| `Fig2a-c_Case1_dryML_eclogite80km.init.t3c` | Fig. 2a–c | Dry mantle lithosphere, 80 km eclogite block |
| `Fig2d-f_Case2_dryML_eclogite300km.init.t3c` | Fig. 2d–f | Dry mantle lithosphere, 300 km eclogite |
| `Fig2g-i_Case3_wetML_250ppm_eclogite80km.init.t3c` | Fig. 2g–i; Supplementary Fig. 2a–b | Wet mantle lithosphere (250 ppm H/Si), 80 km eclogite |
| `Fig2j-l_Case4_wetML_carbonatite.init.t3c` | Fig. 2j–l; Supplementary Fig. 2c–d | Case 3 with carbonatite-metasomatized mantle lithosphere |
| `Fig2m-o_Case5_wetML_pyroxenite.init.t3c` | Fig. 2m–o | Case 3 with additional pyroxenite |
| `Fig3a-c_sequential_pyroxenite_thickness.init.t3c` | Fig. 3a–c | Two eclogite bodies; thicker pyroxenite on the left |
| `Fig3d-f_sequential_hydration_400ppm_10ppm.init.t3c` | Fig. 3d–f | 400 ppm H/Si on the left, 10 ppm H/Si on the right |
| `Fig4_partial_melting.init.t3c` | Fig. 4 | Partial melting of the pyroxenite/eclogite drip |
| `FigS1a-c_250ppm_eclogite50km.init.t3c` | Supplementary Fig. 1a–c | 250 ppm H/Si, 50 km eclogite |
| `FigS1d-f_500ppm_eclogite50km.init.t3c` | Supplementary Fig. 1d–f | 500 ppm H/Si |
| `FigS1g-i_1000ppm_eclogite50km.init.t3c` | Supplementary Fig. 1g–i | 1000 ppm H/Si |
| `FigS1j-l_1500ppm_eclogite50km.init.t3c` | Supplementary Fig. 1j–l | 1500 ppm H/Si |
| `FigS3a-c_pyroxenite_dryML.init.t3c` | Supplementary Fig. 3a–c | Pyroxenite in dry mantle lithosphere |
| `FigS3d-f_pyroxenite_wetML_250ppm.init.t3c` | Supplementary Fig. 3d–f | Pyroxenite in wet mantle lithosphere (250 ppm H/Si) |
| `FigS4a-c_grainsize_100um.init.t3c` | Supplementary Fig. 4a–c | Initial grain size 100 μm |
| `FigS4d-f_grainsize_10um.init.t3c` | Supplementary Fig. 4d–f | Initial grain size 10 μm |
| `FigS4g-i_grainsize_1um.init.t3c` | Supplementary Fig. 4g–i | Initial grain size 1 μm |
| `FigS5a-c_plasticity_0.01.init.t3c` | Supplementary Fig. 5a–c | sin(φ)λ = 0.01, residual strength 1 MPa |
| `FigS5d-f_plasticity_0.005_cohesion1MPa.init.t3c` | Supplementary Fig. 5d–f | sin(φ)λ = 0.005, residual strength 1 MPa |

Supplementary Fig. 2 uses the same two experiments as Fig. 2g–i and Fig. 2j–l. The `init/` files are the initial conditions of the submitted Nature Communications figures.

## Plotting scripts and the files they read

Scripts are under `plotting/` with the same relative layout they use on disk. Only the snapshots opened by each script are included.

| Script | Figure | Inputs |
| --- | --- | --- |
| `plotting/Fig2/plot_fig2_strength.py` | Fig. 2 | `gouc` steps 53, 363, 499; 56, 548, 1076; 52, 170, 240; 52, 155, 330; 54, 100, 200 |
| `plotting/Fig3/plot_fig4_sequential_drips_Noisostasy.py` | Fig. 3 | `gouc` steps 23, 90, 240 and 19, 80, 170. Panel f elevation is `gouc221.txt` |
| `plotting/Fig4/plot_fig5_melting.py` and `fig5_data.py` | Fig. 4 | `gouc`, `gout`, and `goup` at steps 53, 115, and 205; `marker_pt.npz`; `marker_pt_asth.npz` |
| `plotting/FigS1/plot_figS1_water.py` | Supplementary Fig. 1 | `gouc` steps 85, 205, 285; 85, 178, 263; 85, 156, 245; 85, 147, 242 |
| `plotting/FigS2/plot_figS2_viscosity_80kmNoIsostasy.py` | Supplementary Fig. 2 | Case 3: `gouc120.txt`, `gouv120.txt`. Case 4: `gouc90.txt`, `gouv90.txt` |
| `plotting/FigS3/plot_figS3_existence.py` | Supplementary Fig. 3 | `gouc` steps 51, 151, 251 in each experiment |
| `plotting/FigS4/plot_figS4_grain.py` | Supplementary Fig. 4 | `gouc` steps 84, 199, 276; 84, 197, 270; 84, 192, 275 |
| `plotting/FigS5/plot_figS5_plasticity.py` | Supplementary Fig. 5 | `gouc` steps 80, 216, 286 and 80, 203, 279 |

Fig. 4d reads the two NumPy archives (pyroxenite marker 378201 and asthenosphere marker 204703).

Redraw a figure from its directory:

```bash
cd plotting/Fig2
python plot_fig2_strength.py
```

Keep `plotting/FigS2` next to `plotting/Fig2`. The viscosity script loads Case 3 and Case 4 from `../Fig2`.

The scripts import NumPy and Matplotlib. Fig. 4 also imports SciPy (`scipy.ndimage.gaussian_filter`).

## File formats and units

`gouc*.txt` is run-length-encoded composition on a uniform grid covering 0–800 km in x and 0–600 km in z. The z coordinate includes 10 km of sticky air; plotted depth is model z minus 10 km. The first value is time in years. A value of −2 starts a repeat count; −1 is an empty cell.

`gout*.txt` is temperature in kelvin on the 501 × 376 node grid. The melting script plots degrees Celsius as T(K) − 273.

`goup*.txt` is pressure in pascals. The melting script plots gigapascals.

`gouv*.txt` is effective viscosity in Pa·s. Supplementary Fig. 2 plots log10 of that field, between 16 and 26.

## Licence

This repository is released under the MIT License in `LICENSE`. Copyright (c) 2025 Yiren Gou.

## Not in this repository

Supplementary Fig. 6 (resolution tests) is not included. Fig. 1 is the topographic and seismic map drawn with the Generic Mapping Tools. Fig. 5 is the schematic of the model domain. Neither is produced by the scripts in `plotting/`.

## Citation

Gou, Y. & Liu, M. Model initial conditions and figure source data for “Eclogite-pyroxenite drips and localized lithosphere thinning under the Central Andes.” https://github.com/YirenGou/Gou-and-Liu-2026-Dripping-Tectonics
