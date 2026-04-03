import json

# Read the notebook
with open("SLP/SLP-MNIST/fitting_function_IPA.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Build the new Cell 2 source
new_cell2_lines = [
    "import os\n",
    "import glob\n",
    "import re\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "from lmfit import Parameters, minimize\n",
    "import warnings\n",
    "warnings.filterwarnings(\"ignore\")\n",
    "\n",
    "# ── Paths ────────────────────────────────────────────────────────────────────\n",
    "BASE_DIR  = r\"C:\\Users\\Student\\Desktop\\Neural_research\\physlab\\SLP\\SLP-MNIST\\prune_layers_ALL\"\n",
    "OUT_DIR   = r\"C:\\Users\\Student\\Desktop\\Neural_research\\physlab\\SLP\\SLP-MNIST\\Fitting_IPA_curves_data\"\n",
    "\n",
    "BATCH_SIZES = [64, 1024, 60000]\n",
    "LN10        = np.log(10)\n",
    "\n",
    "# ── Auto-detect pruning percentages from directory names ─────────────────────\n",
    "p_dirs = glob.glob(os.path.join(BASE_DIR, \"p-percentage_*\"))\n",
    "PRUNING_LEVELS = sorted([\n",
    "    float(re.search(r\"p-percentage_([\\d.]+)\", d).group(1))\n",
    "    for d in p_dirs\n",
    "])\n",
    "print(f\"Found {len(PRUNING_LEVELS)} pruning percentages: {PRUNING_LEVELS}\")\n",
    "\n",
    "# ── Color palette ─────────────────────────────────────────────────────────────\n",
    "COLOR_LIST_BASE = [\n",
    "    \"#1f77b4\", \"#ff7f0e\", \"#2ca02c\", \"#d62728\", \"#800080\",\n",
    "    \"#8c564b\", \"#e377c2\", \"#7f7f7f\", \"#bcbd22\", \"#B9D9EB\", \"#17becf\",\n",
    "    \"#C5B0D5\", \"#FDB462\", \"#80B1D3\", \"#FB8072\", \"#BEBADA\",\n",
    "    \"#8DD3C7\", \"#FFFFB3\", \"#A6D854\", \"#FCCDE5\"\n",
    "]\n",
    "n_colors   = len(PRUNING_LEVELS)\n",
    "COLOR_LIST = (COLOR_LIST_BASE * ((n_colors // len(COLOR_LIST_BASE)) + 1))[:n_colors]\n",
    "COLOR_LIST = COLOR_LIST[::-1]\n",
    "P_COLOR    = {p: COLOR_LIST[i] for i, p in enumerate(PRUNING_LEVELS)}\n",
    "\n",
    "\n",
    "# ── FIT PARAMETERS ────────────────────────────────────────────────────────────\n",
    "A_MIN, A_MAX = 0.1, 2.3\n",
    "B_MIN, B_MAX = 0, 1000\n",
    "N_MIN, N_MAX = 0.5, 3.0\n",
    "\n",
    "\n",
    "# ── FITTING FUNCTIONS (lmfit approach) ────────────────────────────────────────\n",
    "def initialize_guesses(x, y):\n",
    "    \"\"\"Smart initial parameter estimation for CE power-law fit.\"\"\"\n",
    "    x = np.asarray(x, dtype=float)\n",
    "    y = np.asarray(y, dtype=float)\n",
    "    y = y[np.isfinite(y)]\n",
    "\n",
    "    A0 = np.percentile(y, 5)\n",
    "    B0 = np.percentile(y, 95) - A0\n",
    "    n0 = 0.5\n",
    "\n",
    "    if len(x) > 10:\n",
    "        denom = y[0] - A0\n",
    "        if abs(denom) > 1e-10:\n",
    "            frac = max(1e-6, (y[0] - y[-1]) / denom)\n",
    "            if frac > 0:\n",
    "                n0 = max(0.3, min(1.5, -np.log(frac)))\n",
    "\n",
    "    return A0, n0, B0\n",
    "\n",
    "\n",
    "def model(params, x):\n",
    "    \"\"\"Model function: A + B / (x+1)^n\"\"\"\n",
    "    vals = params.valuesdict()\n",
    "    A, B, n = vals['A'], vals['B'], vals['n']\n",
    "    return A + B / ((x + 1)**n)\n",
    "\n",
    "\n",
    "def residual(params, x, data):\n",
    "    \"\"\"Weighted residual: weight = x\"\"\"\n",
    "    weight = x\n",
    "    return weight * (model(params, x) - data)\n",
    "\n",
    "\n",
    "def fit_curve(x, y):\n",
    "    \"\"\"Fit CE curve using lmfit. Returns lmfit result or None.\"\"\"\n",
    "    mask  = ~np.isnan(y)\n",
    "    x_fit = x[mask]\n",
    "    y_fit = y[mask]\n",
    "\n",
    "    if len(x_fit) < 10:\n",
    "        return None\n",
    "\n",
    "    A0, n0, B0 = initialize_guesses(x_fit, y_fit)\n",
    "\n",
    "    params = Parameters()\n",
    "    params.add('A', value=A0, min=A_MIN, max=A_MAX)\n",
    "    params.add('B', value=B0, min=B_MIN, max=B_MAX)\n",
    "    params.add('n', value=n0, min=N_MIN, max=N_MAX)\n",
    "\n",
    "    try:\n",
    "        return minimize(residual, params, args=(x_fit, y_fit))\n",
    "    except Exception:\n",
    "        return None\n",
    "\n",
    "print(\"Cell 1 ready.\")\n",
]

# Update Cell 2 (index 2)
nb['cells'][2]['source'] = new_cell2_lines

# Now update Cell 4 (index 4) - Per-run fitting
# Extract current cell and replace fit_run calls
cell4_src = ''.join(nb['cells'][4]['source'])

# Replace fit_run calls with fit_curve and update result extraction
# Old pattern: popt = fit_run(x, y); if popt is None:; A, B, n = popt
# New pattern: result = fit_curve(x, y); if result is None:; vals = result.params.valuesdict(); A, B, n = vals['A'], vals['B'], vals['n']

cell4_new = cell4_src.replace(
    "            popt = fit_run(x, y)\n            if popt is None:",
    "            result = fit_curve(x, y)\n            if result is None:"
).replace(
    "            A, B, n = popt",
    "            vals = result.params.valuesdict()\n            A, B, n = vals['A'], vals['B'], vals['n']"
).replace(
    "            fitted_y = power_decay(x, A, B, n)",
    "            fitted_y = model(Parameters(A=A, B=B, n=n), x)"
).replace(
    "            fitted_matrix = np.array([power_decay(all_x, A, B, n) for A, B, n in run_params])",
    "            fitted_matrix = np.array([model(Parameters(A=A, B=B, n=n), all_x) for A, B, n in run_params])"
)

nb['cells'][4]['source'] = [line if line.endswith('\n') else line + '\n' for line in cell4_new.split('\n')[:-1]] + [cell4_new.split('\n')[-1]]

# Now update Cell 6 (index 6) - v4 single fit
cell6_src = ''.join(nb['cells'][6]['source'])

cell6_new = cell6_src.replace(
    "        popt = fit_run(x, y)\n        if popt is None:",
    "        result = fit_curve(x, y)\n        if result is None:"
).replace(
    "        A, B, n = popt",
    "        vals = result.params.valuesdict()\n        A, B, n = vals['A'], vals['B'], vals['n']"
).replace(
    "        fit_y = power_decay(x, A, B, n)",
    "        fit_y = model(Parameters(A=A, B=B, n=n), x)"
)

nb['cells'][6]['source'] = [line if line.endswith('\n') else line + '\n' for line in cell6_new.split('\n')[:-1]] + [cell6_new.split('\n')[-1]]

# Write back the updated notebook
with open("SLP/SLP-MNIST/fitting_function_IPA.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("[OK] All cells updated successfully")
print("  - Cell 2: Replaced scipy with lmfit, added fitting functions")
print("  - Cell 4: Updated fit_run -> fit_curve, power_decay -> model")
print("  - Cell 6: Updated fit_run -> fit_curve, power_decay -> model")
