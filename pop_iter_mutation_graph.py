import random
import numpy as np
import matplotlib.pyplot as plt

from GulAbdulHalim_Elif_R00278544_MH1 import BasicTSP, genDists, myStudentNum


import sys, os, random, math, traceback
import numpy as np
import matplotlib.pyplot as plt

# ---- CONFIG (edit here as you like) ----
MODULE_NAME = "MODULE_NAME"
INST        = "inst-c.tsp"   # your TSP instance filename/path
XOVERH      = 0              # crossover operator index (per your code)
PC          = 0.90           # crossover probability
EL          = 0.10           # elitism rate
TR          = 0.50           # truncation rate

# Grids for Figure 1
POPS  = [50, 100, 200]
ITERS = [500, 1000, 1500]
PMS   = [0.00, 0.05, 0.10, 0.20]

# Settings for Figure 2 (operator comparison)
OP_COMPARE_POP   = 200
OP_COMPARE_ITERS = 1000
OP_COMPARE_PM    = 0.20
OPERATOR_LABELS  = {0: "Reciprocal mutation", 1: "Inversion mutation"}

# Optional fixed seed for reproducibility (will use your myStudentNum if it exists)
#FALLBACK_SEED = 12345
# ----------------------------------------

def main():
    # Make sure we can import your module
    sys.path.append(os.getcwd())
    try:
        user_mod = __import__(MODULE_NAME)
    except Exception as e:
        print("Import error while loading your module:", e)
        print("Tip: Run this script from the same directory as your project files.")
        raise

    # Pull required API from your module
    try:
        BasicTSP = getattr(user_mod, "BasicTSP")
        genDists = getattr(user_mod, "genDists")
    except AttributeError as e:
        raise RuntimeError(
            f"Your module '{MODULE_NAME}' does not expose expected symbols. "
            f"Make sure BasicTSP and genDists are available."
        ) from e

    # Seed: use your myStudentNum if present; otherwise fall back
    my_seed = getattr(user_mod, "myStudentNum")

    # Build distances (as per your module’s API)
    dists = genDists(INST)

    # ===========================================================
    # Figure 1 — pM curves vs computed generations (pop*iters)
    # ===========================================================
    print("Running grid for Figure 1 (pM vs computed generations)...")

    # pm_to_points maps each pM to list of (computed_generations, final_best_fitness)
    pm_to_points = {pm: [] for pm in PMS}

    for pm in PMS:
        for pop in POPS:
            for nit in ITERS:
                random.seed(my_seed)
                # Fix mutH=1 (e.g., inversion) for this scan; we only vary pM, pop, iters here
                ga = BasicTSP(INST, nit, pop, XOVERH, PC, 1, pm, EL, TR, dists)
                final_best, init_best, best_sol, gen_data = ga.search()

                computed_generations = pop * nit  # <<< requested: popSize × nIters
                pm_to_points[pm].append((computed_generations, final_best))

    # Plot: x = computed generations, y = final best fitness, one curve per pM
    plt.figure(figsize=(8,5))
    for pm, pts in pm_to_points.items():
        pts_sorted = sorted(pts, key=lambda t: t[0])
        xs = [t[0] for t in pts_sorted]
        ys = [t[1] for t in pts_sorted]
        plt.plot(xs, ys, marker='o', label=f"pM = {pm}")

    plt.xlabel("Computed generations (popSize × nIters)")
    plt.ylabel("Final best fitness (distance)")
    plt.title("Final best fitness vs computed generations (grouped by pM)")
    plt.legend(title="Mutation probability")
    plt.tight_layout()
    plt.savefig("figure_1_pm_vs_computed_generations.png", dpi=150)
    plt.show()

    # ==========================================
    # Figure 2 — MUTATION OPERATORS COMPARISON
    # ==========================================
    print("Running operator comparison for Figure 2...")
    fig2 = plt.figure(figsize=(8,5))

    # We try to draw best_fitness vs generation (if your search() returns generation logs).
    # If not available, we will fall back to a simple two-point line (start->final).
    for mutH in [0, 1]:
        random.seed(my_seed)
        ga = BasicTSP(INST, OP_COMPARE_ITERS, OP_COMPARE_POP, XOVERH, PC, mutH, OP_COMPARE_PM, EL, TR, dists)
        final_best, init_best, best_sol, gen_data = ga.search()

        label = OPERATOR_LABELS.get(mutH, f"Operator {mutH}")

        # Try to plot the full evolution curve
        gens = None
        fits = None
        if isinstance(gen_data, (list, tuple)) and len(gen_data) > 0 and isinstance(gen_data[0], dict):
            gens = [row.get("generation", i) for i, row in enumerate(gen_data)]
            fits = [row.get("best_fitness", np.nan) for row in gen_data]

        if gens is not None and fits is not None and len(gens) == len(fits) and len(gens) > 1:
            plt.plot(gens, fits, label=label)
        else:
            # Fallback: simple line from iteration 0 to OP_COMPARE_ITERS using initial/final
            plt.plot([0, OP_COMPARE_ITERS], [init_best, final_best], label=label)

    plt.xlabel("Generation")
    plt.ylabel("Best fitness (distance)")
    plt.title(
        f"Effect of mutation operators (pop={OP_COMPARE_POP}, "
        f"iters={OP_COMPARE_ITERS}, pM={OP_COMPARE_PM})"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig("figure_2_operator_comparison.png", dpi=150)
    plt.show()

    print("\nSaved files:")
    print(" - figure_1_pm_vs_computed_generations.png")
    print(" - figure_2_operator_comparison.png")

if __name__ == "__main__":
    main()
