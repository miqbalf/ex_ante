# Ex-Ante tCO2 Carbon Estimation Documentation

## Overview

This documentation describes the **Ex-Ante Carbon Estimation** system, a comprehensive tool for estimating carbon sequestration (tCO2) in forestry projects using interactive Jupyter widgets. The system enables users to:

- Select tree species with allometric formulas
- Input plot-level seedling distribution data
- Configure forestry management scenarios (harvesting cycles, mortality, thinning)
- Calculate carbon sequestration over project duration
- Generate outputs for further analysis and reporting

## Table of Contents

1. [Introduction](#introduction)
2. [System Architecture](#system-architecture)
3. [Installation & Setup](#installation--setup)
4. [Workflow Overview](#workflow-overview)
5. [Step-by-Step Guide](#step-by-step-guide)
6. [UI Components](#ui-components)
7. [Output Files](#output-files)
8. [Configuration Options](#configuration-options)
9. [Troubleshooting](#troubleshooting)
10. [Technical Details](#technical-details)

---

## Introduction

### What is Ex-Ante Carbon Estimation?

Ex-ante carbon estimation calculates the **potential carbon sequestration** of a forestry project **before** planting begins. This forward-looking analysis helps:

- Estimate carbon credits for carbon markets
- Plan forestry operations
- Meet certification requirements (e.g., Tree C-Sink standards)
- Model different management scenarios

### Key Features

- **Interactive UI**: User-friendly Jupyter widgets for data entry and selection
- **Species Database**: Access to allometric formulas and growth models for 98+ tree species
- **Flexible Configuration**: Support for multiple zones (production/protected), replanting scenarios
- **Comprehensive Outputs**: Raw data, graphs, and formatted tables
- **Baseline Calculation**: Compare project carbon against baseline scenarios

---

## System Architecture

### Core Components

```
ex_ante/
├── exante.py                    # Main ExAnteCalc class
├── ex_ante/
│   ├── coredb_trees/           # Species database (allometry & growth)
│   │   ├── allometry_formulas.py
│   │   └── growth_models.py
│   ├── ui/                      # Interactive widgets
│   │   ├── main.py             # CSUEntryForm, SelectingScenario, Project_Setting_Species
│   │   └── utils.py            # Helper functions
│   ├── csi_tree/               # Carbon sequestration calculations
│   │   ├── main.py             # CSIExante class
│   │   ├── input_cooling_creation.py
│   │   └── utils.py
│   ├── plot/                   # Plot-level calculations
│   ├── population_tco2/        # Population-level carbon modeling
│   └── utils/                  # Utility functions
```

### Data Flow

```
1. Species Selection (UI) 
   ↓
2. Allometry & Growth Data Loading
   ↓
3. Plot Data Entry (UI)
   ↓
4. Scenario Configuration (UI)
   ↓
5. Carbon Calculation Engine
   ↓
6. Output Generation
```

---

## Installation & Setup

### Prerequisites

- Python 3.8+
- Jupyter Lab/Notebook
- Required packages (see `requirements.txt`)

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or with trusted hosts (if SSL certificate issues)
pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org -r requirements.txt
```

### Environment Configuration

Create a `.env` file in your project root:

```env
# Google Sheets links (optional, for downloading data)
CSV_GROWTH_MODEL=https://docs.google.com/spreadsheets/d/...
CSV_ALLOMETRY_FORMULA=https://docs.google.com/spreadsheets/d/...
```

### First-Time Setup

1. Create output directories:
   ```python
   import os
   os.makedirs('01_output', exist_ok=True)
   os.makedirs('01_output/ex_ante', exist_ok=True)
   ```

2. Set paths for data files:
   ```python
   current_path = os.getcwd()
   growth_csv_abs_path = os.path.join(current_path, '01_output/ex_ante/growth_model.csv')
   allometry_csv_abs_path = os.path.join(current_path, '01_output/ex_ante/allometry_model.csv')
   json_config_abspath = os.path.join(current_path, '01_output/00_ex_ante_result/example_project.json')
   ```

---

## Workflow Overview

The ex-ante carbon estimation process has **two main workflows** depending on whether this is a new project or using existing configuration:

### Workflow A: New Project (First Run)

When `first_run=True`, the system guides you through interactive data entry:

```
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: Initialize Project & Load Species Database         │
│  └─> Create project config, download/load growth & allometry│
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 2: Select Species & Allometric Formulas (UI Widget)   │
│  └─> Choose countries, allometry types, zones, species       │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 3: Enter Plot Data (CSUEntryForm Widget)              │
│  └─> Input plot info, zones, species counts per plot        │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 4: Configure Forestry Scenarios (UI Widgets)          │
│  └─> Set harvesting cycles, mortality, thinning per species │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 5: Calculate & Generate Outputs                       │
│  └─> Run model, generate graphs, calculate baseline         │
└─────────────────────────────────────────────────────────────┘
```

### Workflow B: Existing Project (Re-run with Existing Data)

When `first_run=False`, the system loads all existing configuration and data files, **skipping the UI steps**:

```
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: Load Existing Configuration                        │
│  └─> Load JSON config, CSV files (species, plots, scenarios) │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 2: Calculate & Generate Outputs                       │
│  └─> Run model with existing data, generate outputs         │
└─────────────────────────────────────────────────────────────┘
```

**Key Differences:**
- **No UI widgets** are displayed (data is loaded from saved files)
- **Faster workflow** - directly proceeds to calculation
- **Useful for**: Re-running calculations, parameter sensitivity analysis, updating outputs

---

## Step-by-Step Guide

### Step 1: Initialize the ExAnteCalc Instance

#### Option A: New Project (First Run)

For a **new project**, use `first_run=True`:

```python
from exante import ExAnteCalc
from dotenv import load_dotenv
import os

load_dotenv()

# Configuration paths
json_config_abspath = '01_output/00_ex_ante_result/example_project.json'
growth_csv_abs_path = '01_output/ex_ante/growth_model.csv'
allometry_csv_abs_path = '01_output/ex_ante/allometry_model.csv'

# Initialize ExAnteCalc for NEW project
ex_ante = ExAnteCalc(
    json_path_config=json_config_abspath,
    first_run=True,                    # NEW PROJECT: Set True
    download_csv=True,                  # Download from Google Sheets
    growth_csv=growth_csv_abs_path,
    allometry_csv=allometry_csv_abs_path,
    link_allometry=os.getenv('CSV_ALLOMETRY_FORMULA'),
    link_growth=os.getenv('CSV_GROWTH_MODEL'),
    name_column_species_allo='Lat. Name',
    name_column_species_growth='Tree Species(+origin of allom. formula)'
)
```

**First Run Configuration Prompts:**

When `first_run=True`, you'll be prompted interactively to enter:

- **Project Name**: Identifier for your project
- **Project Duration**: Years (default: 30)
- **Gap Harvest**: Enable/disable gap between harvest cycles (yes/no)
- **Thinning Stop**: Apply density-based thinning algorithm (yes/no)
- **Max Harvest Percentage**: Maximum harvest % per cycle (default: 59.9%)
- **Planting Year**: Year when planting starts (default: current year)

This creates a JSON configuration file at `json_config_abspath`.

#### Option B: Existing Project (Re-run with Existing Configuration)

For **existing projects**, use `first_run=False` to load saved data:

```python
from exante import ExAnteCalc
from dotenv import load_dotenv
import os

load_dotenv()

# Configuration paths (point to EXISTING project)
json_config_abspath = '01_output/00_ex_ante_result/existing_project_main.json'
growth_csv_abs_path = '01_output/ex_ante/growth_model.csv'
allometry_csv_abs_path = '01_output/ex_ante/allometry_model.csv'

# Initialize ExAnteCalc for EXISTING project
ex_ante = ExAnteCalc(
    json_path_config=json_config_abspath,
    first_run=False,                   # EXISTING PROJECT: Set False
    re_select_growth_data=False,       # Use saved growth data (True to re-select)
    download_csv=False,                 # Use local CSV files
    growth_csv=growth_csv_abs_path,
    allometry_csv=allometry_csv_abs_path,
    link_allometry=os.getenv('CSV_ALLOMETRY_FORMULA'),  # Optional
    link_growth=os.getenv('CSV_GROWTH_MODEL'),           # Optional
    name_column_species_allo='Lat. Name',
    name_column_species_growth='Tree Species(+origin of allom. formula)'
)
```

**What Happens with `first_run=False`:**

1. **Loads existing configuration** from JSON file:
   - Project name, duration, harvest settings
   - Planting year, gap harvest, thinning stop settings

2. **Loads existing data files** automatically:
   - `{project_name}_formulas_allometry.csv` → `ex_ante.df_tree_selected`
   - `{project_name}_selected_growth_model.csv` → `ex_ante.growth_selected`
   - `{project_name}_distribution_trees_seedling.csv` → `ex_ante.csu_seedling`
   - `{project_name}_forestry_scenario.json` → `ex_ante.input_scenario_species`

3. **Displays loaded data** in notebook for verification

4. **No UI widgets appear** - data is already configured

5. **Ready for calculation** - proceed directly to `csi_plot_model()`

**Important Notes:**
- All required files must exist in the project folder
- If `re_select_growth_data=True`, growth data will be re-acquired from species selection
- The system validates that all required files exist before proceeding

---

### Step 2: Species Selection (Interactive UI)

After initialization, the **SelectingScenario** widget appears automatically.

#### Using the Species Selection Widget

1. **Select Countries**: Choose countries where allometric formulas are validated
   - Example: "Indonesia", "Singapore"

2. **Select Allometry Type**: Choose formula types
   - Examples: "Above-ground biomass", "Total tree biomass (TTB)"

3. **Select Zones**: Check boxes for zones to include
   - `production_zone`: Managed plantation areas
   - `protected_zone`: Conservation areas

4. **Select Species**: For each zone, choose species from the dropdown list

5. **Submit**: Click "Submit species!" button

#### Outputs from Step 2

- `ex_ante.df_tree_selected`: DataFrame with selected species and allometry formulas
- `ex_ante.unique_species_selected`: List of unique species names
- `ex_ante.growth_selected`: Growth model data for selected species
- Saved to: `{project_name}_formulas_allometry.csv`

---

### Step 3A: Plot Data Entry (CSUEntryForm) - New Projects Only

> **Note**: This step is **only for new projects** (`first_run=True`).  
> For existing projects, plot data is loaded from `{project_name}_distribution_trees_seedling.csv`.

After submitting species, the **CSUEntryForm** widget appears.

#### Form Fields

| Field | Description | Example |
|-------|-------------|---------|
| `Plot_ID` | Unique identifier | 1 |
| `Plot_Name` | Plot name/label | "Plot_A" |
| `zone` | Production or protected zone | "production_zone" |
| `area_ha` | Plot area in hectares | 50.0 |
| `is_replanting` | Replanting scenario (checkbox) | True/False |
| `year_start` | Starting year for this plot | 1 |
| `mu` | Management unit identifier | "MU_1_1" |
| `{species}_num_trees` | Number of trees per species | 1000 |

#### Workflow

1. Fill in all fields for a plot
2. Click **"Add Row"** to add the plot data
3. Review data in the preview table below
4. Repeat for additional plots (Plot_ID auto-increments)
5. Click **"✅ SUBMIT CSU DATA"** when finished

#### Outputs from Step 3

- `ex_ante.csu_seedling`: DataFrame with plot-level seedling distribution
- Saved to: `{project_name}_distribution_trees_seedling.csv`

---

### Step 4A: Forestry Scenario Configuration - New Projects Only

> **Note**: This step is **only for new projects** (`first_run=True`).  
> For existing projects, scenario data is loaded from `{project_name}_forestry_scenario.json`.

After submitting CSU data, **Project_Setting_Species** widgets appear for each zone/replanting combination.

#### Scenario Parameters

For each species in each zone, configure:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `harvest_cycle_year` | Years until harvest | Based on growth data |
| `mortality_rate_percent` | Annual mortality % | 20.0 |
| `natural_thinning_percent` | Annual natural thinning % | 5.0 |
| `how many times manual thinning` | Number of manual thinnings | 0 |
| `thinning_cycle{N}_year` | Year for manual thinning N | - |
| `thinning_cycle{N}_percent` | % removed in thinning N | - |

#### Workflow

1. Review species widgets for each zone/replanting combination
2. Adjust parameters as needed
   - Harvest cycle must be ≤ max years in growth data
   - System warns if values seem unreasonable
3. Click **"✅ Submit Scenario Data"** when finished

#### Outputs from Step 4

- `ex_ante.input_scenario_species`: Dictionary with all scenario parameters
- Saved to: `{project_name}_forestry_scenario.json`

---

### Step 2B/5: Carbon Calculation & Outputs (Both Workflows)

This step is **common to both workflows** - whether using new or existing configuration, you'll run the carbon calculation model.

#### 5.1 Run the Main Calculation

```python
all_df_merged = ex_ante.csi_plot_model()
```

This executes:
- Plot-level carbon calculations
- Harvest cycle modeling
- Thinning calculations
- Replanting logic (if enabled)
- Time-series generation

**Output**: `all_df_merged` DataFrame with carbon estimates per plot/species/year

#### 5.2 Generate Input for Cooling Service

```python
from ex_ante.csi_tree.input_cooling_creation import input_cooling

ex_ante.input_gcs = input_cooling(
    all_df_merged,
    ex_ante.growth_melt,
    ex_ante.name_column_species_growth,
    ex_ante.conf_general["planting_year"],
)
ex_ante.input_gcs.to_csv(ex_ante.gdrive_input_cs, index=False)
```

#### 5.3 Generate Species-Level Graphs

```python
graph_saved_species = f"graph_species_tCO2stored_{ex_ante.config['project_name']}.png"
ex_ante.graph_species_project(ex_ante.input_gcs, location_save=graph_saved_species)
```

**Outputs**:
- Graph showing tCO2 stored per species over time
- Long-Term Average (LTA) capture value
- LTA with 20% buffer

#### 5.4 Calculate Baseline Comparison

```python
# Baseline as comma-separated values (negative for emissions)
baseline_str = '''
-832.4, -832.4, -832.4, ...  # 30 values
'''

joined_baseline_seq = ex_ante.baseline_calc(input_str=baseline_str)

# Save results
joined_baseline_seq.to_csv(f"table_nettCO2withbaseline_{ex_ante.config['project_name']}.csv")

# Generate graph
ex_ante.graph_seq_baseline(joined_baseline_seq, location_save=graph_path)
```

**Outputs**:
- Net tCO2 sequestration (project - baseline)
- LTA of net sequestration
- Graph comparing project vs baseline

---

## UI Components

### 1. SelectingScenario Widget

**Location**: `ex_ante/ex_ante/ui/main.py`

**Purpose**: Interactive species and allometry formula selection

**Features**:
- Multi-select for countries
- Multi-select for allometry types
- Zone-based species selection (checkboxes)
- Real-time filtering

**Usage**:
```python
self.wm = SelectingScenario(
    allometric_column_filter=self.allometry_df,
    name_column_species_allo=self.name_column_species_allo
)
display(self.wm)
```

### 2. CSUEntryForm Widget

**Location**: `ex_ante/ex_ante/ui/main.py`

**Purpose**: Plot-level data entry form

**Features**:
- Dynamic form fields based on selected species
- Data validation
- Row-by-row addition
- Reset functionality
- Data preview table

**Usage**:
```python
self.csu_form = CSUEntryForm(plot_csu_template)
self.csu_form.display_form()
```

### 3. Project_Setting_Species Widget

**Location**: `ex_ante/ex_ante/ui/main.py`

**Purpose**: Configure forestry management parameters per species

**Features**:
- Harvest cycle input with validation
- Mortality rate configuration
- Natural thinning settings
- Dynamic manual thinning fields
- Real-time parameter validation

**Usage**:
```python
widget = Project_Setting_Species(species_name, grouping_max_year)
```

---

## Output Files

All outputs are saved in the project folder (`00_ex_ante_result/{project_name}/`).

### Configuration Files

- **`{project_name}_main.json`**: Project configuration (duration, harvest settings, etc.)

### Input Files (Generated from UI)

- **`{project_name}_formulas_allometry.csv`**: Selected species with allometry formulas
- **`{project_name}_distribution_trees_seedling.csv`**: Plot-level seedling distribution
- **`{project_name}_forestry_scenario.json`**: Forestry management parameters

### Output Files (Generated from Calculation)

- **`{project_name}_OUTPUT_RAW_CALCULATION.csv`**: Raw calculation results (plot/species/year)
  - Columns: Plot_ID, zone, species, year, rotation_year, total_csu_tCO2e_species, num_trees_adjusted, etc.
  
- **`{project_name}_input_gcs_generated.csv`**: Input file for cooling service
  
- **`{project_name}_selected_growth_model.csv`**: Growth data for selected species

### Analysis Files

- **`table_nettCO2withbaseline_{project_name}.csv`**: Net sequestration after baseline subtraction

### Graphs

- **`graph_species_tCO2stored_{project_name}.png`**: Species-level carbon storage over time
- **`graph_nettCO2withbaseline_{project_name}.png`**: Net sequestration with baseline comparison

---

## Configuration Options

### ExAnteCalc Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `first_run` | bool | - | True for new projects, False to load existing |
| `json_path_config` | str | "" | Path to project JSON config file |
| `download_csv` | bool | False | Download data from Google Sheets |
| `growth_csv` | str | - | Path to growth model CSV file |
| `allometry_csv` | str | - | Path to allometry formulas CSV file |
| `name_column_species_allo` | str | "Lat. Name" | Column name for species in allometry data |
| `name_column_species_growth` | str | "Tree Species..." | Column name for species in growth data |
| `re_select_growth_data` | bool | False | Re-select growth data on load |
| `override_num_trees_0` | bool | False | Override zero tree counts |
| `override_avg_tree_perha` | str | '' | Override average trees per hectare |

### Project Configuration (JSON)

```json
{
  "project_name": "example_project",
  "csv_plot_summary_zone": true,
  "duration_project": 30,
  "gap_harvest": false,
  "thinning_stop": true,
  "harvesting_max_percent": 59.9,
  "planting_year": 2025,
  "root_folder": "/path/to/project"
}
```

---

## Troubleshooting

### Common Issues

#### 1. Widgets Not Displaying

**Symptom**: UI widgets don't appear in Jupyter

**Solutions**:
- Ensure `ipywidgets` is installed: `pip install ipywidgets`
- Enable widgets extension: `jupyter nbextension enable --py widgetsnbextension`
- Restart Jupyter kernel after installation

#### 2. Import Errors

**Symptom**: `ModuleNotFoundError: No module named 'exante'`

**Solution**:
```python
import sys
sys.path.append('/path/to/ex_ante/parent/directory')
```

#### 3. Google Colab Compatibility

**Symptom**: Widgets work in Colab but not in local Jupyter

**Solution**: The library automatically detects environment. Widgets should work in both.

#### 4. Data Download Issues

**Symptom**: CSV download fails from Google Sheets

**Solutions**:
- Verify Google Sheets links in `.env` file
- Ensure sheets are publicly accessible
- Check internet connection
- Use local CSV files with `download_csv=False`

#### 5. Missing Species Data

**Symptom**: Species appears in selection but has no growth/allometry data

**Solution**: 
- Check console output for warnings about missing data
- Verify species names match exactly between databases
- Use species that appear in both growth and allometry databases

---

## Technical Details

### Carbon Calculation Methodology

#### 1. Biomass Calculation

For each tree, biomass is calculated using **allometric formulas**:

```
Biomass = Allometric_Formula(DBH, Height, ...)
```

Where:
- DBH (Diameter at Breast Height) comes from growth models
- Formulas vary by species and region

#### 2. Carbon Conversion

```
Carbon (tC) = Biomass (tonnes) × Carbon_Fraction (default: 0.47)
tCO2e = Carbon (tC) × Conversion_Factor (44/12)
```

#### 3. Plot-Level Aggregation

```
Plot_Carbon = Σ(Tree_Biomass × Num_Trees × Carbon_Fraction × 44/12)
```

#### 4. Harvest Cycle Modeling

The system models:
- **Harvesting**: Removes percentage of trees at harvest year
- **Retention**: Keeps 40%+ carbon in forest (CSI standard)
- **Replanting**: Re-establishes trees after harvest (if enabled)
- **Thinning**: Removes trees at specific years/percentages

#### 5. Time-Series Generation

For each year in project duration:
1. Calculate tree growth (DBH from growth model)
2. Apply mortality and thinning
3. Calculate carbon stored
4. Apply harvesting (if harvest year)
5. Replant (if enabled and after gap period)

### Key Algorithms

#### Thinning Stop Algorithm

When `thinning_stop=True`, the system:
1. Calculates maximum stand density based on average DBH
2. Stops natural thinning when density threshold is reached
3. Uses `max_density_trees_ha.csv` for density thresholds

#### Gap Harvest Logic

When `gap_harvest=True`:
- After harvest, replanting is delayed by 1 year
- Allows for soil recovery and operational planning

---

## Working with Existing Projects

### Complete Workflow for Existing Projects

When you have an existing project configuration and want to re-run calculations:

```python
from exante import ExAnteCalc
from dotenv import load_dotenv
import os

load_dotenv()

# Point to your existing project configuration
json_config_abspath = '01_output/00_ex_ante_result/your_project_main.json'
growth_csv_abs_path = '01_output/ex_ante/growth_model.csv'
allometry_csv_abs_path = '01_output/ex_ante/allometry_model.csv'

# Initialize with existing configuration
ex_ante = ExAnteCalc(
    json_path_config=json_config_abspath,
    first_run=False,                   # Load existing config
    re_select_growth_data=False,       # Use saved growth data
    download_csv=False,                # Use local CSV files
    growth_csv=growth_csv_abs_path,
    allometry_csv=allometry_csv_abs_path,
    name_column_species_allo='Lat. Name',
    name_column_species_growth='Tree Species(+origin of allom. formula)'
)

# Data is automatically loaded:
# - ex_ante.df_tree_selected (species & allometry)
# - ex_ante.growth_selected (growth models)
# - ex_ante.csu_seedling (plot distribution)
# - ex_ante.input_scenario_species (forestry scenarios)

# Verify loaded data
display(ex_ante.df_tree_selected)
display(ex_ante.growth_selected)
display(ex_ante.csu_seedling)

# Proceed directly to calculation
all_df_merged = ex_ante.csi_plot_model()

# Continue with outputs (same as new project workflow)
# ... (see Step 5 for details)
```

### Required Files for Existing Projects

When using `first_run=False`, ensure these files exist in your project folder:

| File | Description | Location |
|------|-------------|----------|
| `{project_name}_main.json` | Project configuration | `00_ex_ante_result/{project_name}/` |
| `{project_name}_formulas_allometry.csv` | Selected species/allometry | Same folder |
| `{project_name}_distribution_trees_seedling.csv` | Plot data | Same folder |
| `{project_name}_forestry_scenario.json` | Scenario parameters | Same folder |
| `{project_name}_selected_growth_model.csv` | Growth models (optional if `re_select_growth_data=True`) | Same folder |

**File Structure Example:**
```
01_output/
└── 00_ex_ante_result/
    └── my_project/
        ├── my_project_main.json
        ├── my_project_formulas_allometry.csv
        ├── my_project_distribution_trees_seedling.csv
        ├── my_project_forestry_scenario.json
        └── my_project_selected_growth_model.csv
```

### Re-selecting Growth Data

If you want to update growth data while keeping other configurations:

```python
ex_ante = ExAnteCalc(
    json_path_config=json_config_abspath,
    first_run=False,
    re_select_growth_data=True,        # Re-select growth data
    download_csv=False,
    # ... other parameters
)

# Growth data will be re-acquired based on existing species selection
# Other data (plots, scenarios) remain unchanged
```

## Advanced Usage

### Modifying Scenarios After Submission

Edit JSON file directly:
```python
import json

with open(ex_ante.gdrive_location_scenario_rate, 'r') as f:
    scenario = json.load(f)

# Modify scenario
scenario['replanting']['production_zone']['Species Name']['harvesting_year'] = 15

# Save back
with open(ex_ante.gdrive_location_scenario_rate, 'w') as f:
    json.dump(scenario, f, indent=4)
```

### Re-running Calculations with Updated Parameters

#### Option 1: Re-initialize with existing data

```python
# Re-initialize to reload all data
ex_ante = ExAnteCalc(
    json_path_config='path/to/existing_project_main.json',
    first_run=False,
    download_csv=False
)

# Recalculate
all_df_merged = ex_ante.csi_plot_model()
```

#### Option 2: Modify files and reload

```python
# 1. Edit CSV/JSON files manually
# 2. Re-initialize to reload
ex_ante = ExAnteCalc(
    json_path_config='path/to/existing_project_main.json',
    first_run=False
)

# 3. Recalculate with updated data
all_df_merged = ex_ante.csi_plot_model()
```

#### Option 3: Modify scenario parameters programmatically

```python
# Load existing project
ex_ante = ExAnteCalc(
    json_path_config='path/to/existing_project_main.json',
    first_run=False
)

# Modify scenario parameters in memory
ex_ante.input_scenario_species['replanting']['production_zone']['Species Name']['harvesting_year'] = 15

# Save modified scenario
import json
with open(ex_ante.gdrive_location_scenario_rate, 'w') as f:
    json.dump(ex_ante.input_scenario_species, f, indent=4)

# Recalculate with modified scenario
all_df_merged = ex_ante.csi_plot_model()
```

---

## References

### Key Standards

- **Tree C-Sink**: Retain 40%+ carbon in forest after harvest
- **Allometric Formulas**: Regional and species-specific
- **Growth Models**: Based on CoreDB tree growth database

### Data Sources

- **Growth Models**: CoreDB 30/35-year growth database
- **Allometry Formulas**: Validated formulas from scientific literature
- **Max Density**: Stand density thresholds from forestry research

---

## Support & Contribution

For issues, questions, or contributions, please refer to the project repository or contact muh.firdausiqbal@gmail.com

---

## Workflow Comparison Summary

| Feature | New Project (`first_run=True`) | Existing Project (`first_run=False`) |
|---------|--------------------------------|--------------------------------------|
| **UI Widgets** | ✅ All widgets displayed | ❌ No widgets (data loaded from files) |
| **Data Entry** | Interactive forms | Pre-loaded from CSV/JSON |
| **Configuration** | Interactive prompts | Loaded from JSON |
| **Workflow Steps** | 5 steps (including UI) | 2 steps (load + calculate) |
| **Use Case** | Setting up new project | Re-running, updating, analysis |
| **Time to Results** | Longer (data entry required) | Faster (direct to calculation) |

### When to Use Each Workflow

**Use `first_run=True` when:**
- Creating a new project from scratch
- Need to select species and configure scenarios interactively
- Want to explore different species/zone combinations
- First-time setup of a project

**Use `first_run=False` when:**
- Re-running calculations with same configuration
- Updating outputs after modifying input files
- Running sensitivity analysis
- Batch processing multiple scenarios
- Quick recalculation after parameter changes

---

**Last Updated**: 2025-01-31  
**Version**: Based on ex_ante library with UI widgets  
**Compatibility**: Jupyter Lab/Notebook, Google Colab

