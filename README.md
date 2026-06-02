# Isaac4Simulation

## Overview

`Isaac4Simulation` is an internal repository for Isaac Sim based robot simulation demos.

Main purpose:

* Isaac Sim scene setup
* Robot simulation demo scripts
* Palletizing scenario test
* Physical AI demo preparation

---

## Main Script

```text
scripts/run_palletizing.py
```

---

## How to Run

Run the script with Isaac Sim Python.

### Windows PowerShell

Set the Isaac Sim installation path first.

```powershell
$ISAAC_SIM_ROOT = "C:\path\to\isaacsim"
```

Run the main script from the repository root.

```powershell
& "$ISAAC_SIM_ROOT\python.bat" .\scripts\run_palletizing.py
```

Example:

```powershell
cd Isaac4Simulation
& "$ISAAC_SIM_ROOT\python.bat" .\scripts\run_palletizing.py
```

---

## Repository Structure

```text
Isaac4Simulation/
├── assets/      # Simulation assets
├── scripts/     # Isaac Sim execution scripts
├── src/         # Source modules
└── README.md
```

---

## Current Status

Initial setup in progress.

Current focus:

* Palletizing demo script
* Isaac Sim execution flow
* Physical AI simulation demo
