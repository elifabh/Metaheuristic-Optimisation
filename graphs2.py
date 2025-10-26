import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


sns.set_style("whitegrid")
sns.set_palette("husl")

# Matplotlib RC Params
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': '#f8f9fa',
    'axes.edgecolor': '#cccccc',
    'axes.linewidth': 1.2,
    'grid.color': '#e0e0e0',
    'grid.linewidth': 0.8,
    'grid.alpha': 0.4,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'legend.frameon': True,
    'legend.fancybox': False,
    'legend.edgecolor': '#cccccc',
    'legend.framealpha': 0.95,
    'lines.linewidth': 2.5,
    'lines.markersize': 7,
})


COLORS = {
    'run10': '#2E86AB',    # Deep Blue
    'run20': '#A23B72',    # Deep Purple
    'run30': '#F18F01',    # Deep Orange
    'comparison1': '#C73E1D',  # Deep Red
    'comparison2': '#6A994E',  # Deep Green
}

# CSV 
run_data = {}
for run in [10, 20, 30]:
    try:
        df = pd.read_csv(f"experiment_results_run{run}.csv")
        run_data[run] = df
        print(f"✓ experiment_results_run{run}.csv yüklendi ({len(df)} satır)")
    except FileNotFoundError:
        print(f"✗ experiment_results_run{run}.csv bulunamadı")

if not run_data:
    print("Hiç CSV dosyası bulunamadı!")
    exit()

# ===== ELITISM  =====
print("\n" + "="*60)
print("1. ELITISM KARŞILAŞTIRMASI (0.0 vs 0.1)")
print("="*60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, metric in enumerate(['Best Distance Found', 'Average Distance']):
    ax = axes[idx]
    
    x_pos = np.arange(3)
    bar_width = 0.35
    
    elite_0 = []
    elite_01 = []
    
    for i, run in enumerate([10, 20, 30]):
        if run in run_data:
            df = run_data[run]
            # Exp5_Elitism=0.0 ve Exp5_Elitism=0.1 bulunuz
            row_0 = df[df['Experiment Name'] == 'Exp5_Elitism=0.0']
            row_01 = df[df['Experiment Name'] == 'Exp5_Elitism=0.1']
            
            if not row_0.empty:
                elite_0.append(float(row_0[metric].values[0]))
            if not row_01.empty:
                elite_01.append(float(row_01[metric].values[0]))
    
    ax.bar(x_pos[:len(elite_0)] - bar_width/2, elite_0, bar_width, label='Elitism=0.0', 
           color='#C73E1D', alpha=0.85, edgecolor='#8B2E17', linewidth=1.2)
    ax.bar(x_pos[:len(elite_01)] + bar_width/2, elite_01, bar_width, label='Elitism=0.1', 
           color='#6A994E', alpha=0.85, edgecolor='#4A6A2E', linewidth=1.2)
    
    ax.set_xlabel('Number of Runs', fontweight='normal')
    ax.set_ylabel(metric, fontweight='normal')
    ax.set_title(f'Impact of Elitism on {metric}', fontweight='bold', pad=15)
    ax.set_xticks(x_pos[:max(len(elite_0), len(elite_01))])
    ax.set_xticklabels([10, 20, 30][:max(len(elite_0), len(elite_01))])
    ax.legend(loc='best', framealpha=0.95)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('elitism_comparison.png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print("✓ elitism_comparison.png kaydedildi")
plt.show()

# ===== MUTATION PROBABILITY  =====
print("\n" + "="*60)
print("2. MUTATION PROBABILITY KARŞILAŞTIRMASI (Pm)")
print("="*60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, metric in enumerate(['Best Distance Found', 'Average Distance']):
    ax = axes[idx]
    
    pm_values = {}  # {Pm : [run10, run20, run30]}
    
    for run in [10, 20, 30]:
        if run in run_data:
            df = run_data[run]
            exp_rows = df[df['Experiment Name'].str.contains('Exp4_Pm')]
            
            for _, row in exp_rows.iterrows():
                exp_name = row['Experiment Name']
                pm_val = row['Pm']
                metric_val = float(row[metric])
                
                if pm_val not in pm_values:
                    pm_values[pm_val] = {}
                pm_values[pm_val][run] = metric_val
    

    colors_palette = ['#2E86AB', '#A23B72', '#F18F01']
    
    for idx_color, pm_val in enumerate(sorted(pm_values.keys())):
        runs = []
        values = []
        for run in [10, 20, 30]:
            if run in pm_values[pm_val]:
                runs.append(run)
                values.append(pm_values[pm_val][run])
        
        ax.plot(runs, values, marker='o', label=f'Pm = {pm_val}', linewidth=2.5, 
                markersize=8, color=colors_palette[idx_color % len(colors_palette)],
                markerfacecolor=colors_palette[idx_color % len(colors_palette)],
                markeredgewidth=1.5, markeredgecolor='white')
    
    ax.set_xlabel('Number of Runs', fontweight='normal')
    ax.set_ylabel(metric, fontweight='normal')
    ax.set_title(f'Impact of Mutation Probability on {metric}', fontweight='bold', pad=15)
    ax.legend(loc='best', framealpha=0.95)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('mutation_probability_comparison.png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print("✓ mutation_probability_comparison.png kaydedildi")
plt.show()

# ===== CROSSOVER PROBABILITY  =====
print("\n" + "="*60)
print("3. CROSSOVER PROBABILITY KARŞILAŞTIRMASI (Pc)")
print("="*60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, metric in enumerate(['Best Distance Found', 'Average Distance']):
    ax = axes[idx]
    
    pc_values = {}
    
    for run in [10, 20, 30]:
        if run in run_data:
            df = run_data[run]
            exp_rows = df[df['Experiment Name'].str.contains('Exp3_Pc')]
            
            for _, row in exp_rows.iterrows():
                exp_name = row['Experiment Name']
                pc_val = row['Pc']
                metric_val = float(row[metric])
                
                if pc_val not in pc_values:
                    pc_values[pc_val] = {}
                pc_values[pc_val][run] = metric_val
    

    colors_palette = ['#2E86AB', '#A23B72', '#F18F01']
    
    for idx_color, pc_val in enumerate(sorted(pc_values.keys())):
        runs = []
        values = []
        for run in [10, 20, 30]:
            if run in pc_values[pc_val]:
                runs.append(run)
                values.append(pc_values[pc_val][run])
        
        ax.plot(runs, values, marker='s', label=f'Pc = {pc_val}', linewidth=2.5, 
                markersize=8, color=colors_palette[idx_color % len(colors_palette)],
                markerfacecolor=colors_palette[idx_color % len(colors_palette)],
                markeredgewidth=1.5, markeredgecolor='white')
    
    ax.set_xlabel('Number of Runs', fontweight='normal')
    ax.set_ylabel(metric, fontweight='normal')
    ax.set_title(f'Impact of Crossover Probability on {metric}', fontweight='bold', pad=15)
    ax.legend(loc='best', framealpha=0.95)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('crossover_probability_comparison.png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print("✓ crossover_probability_comparison.png kaydedildi")
plt.show()

# ===== CROSSOVER OPERATOR  =====
print("\n" + "="*60)
print("4. CROSSOVER OPERATOR  (OX vs Uniform)")
print("="*60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, metric in enumerate(['Best Distance Found', 'Average Distance']):
    ax = axes[idx]
    
    ox_vals = []
    uniform_vals = []
    
    for run in [10, 20, 30]:
        if run in run_data:
            df = run_data[run]
            
            ox_row = df[df['Experiment Name'] == 'Exp1_OX Crossover']
            uniform_row = df[df['Experiment Name'] == 'Exp1_Uniform Crossover']
            
            if not ox_row.empty:
                ox_vals.append((run, float(ox_row[metric].values[0])))
            if not uniform_row.empty:
                uniform_vals.append((run, float(uniform_row[metric].values[0])))
    
    if ox_vals:
        ox_runs, ox_metrics = zip(*ox_vals)
        ax.plot(ox_runs, ox_metrics, marker='o', label='OX Crossover', linewidth=2.5, 
                markersize=8, color='#C73E1D',
                markerfacecolor='#C73E1D', markeredgewidth=1.5, markeredgecolor='white')
    
    if uniform_vals:
        uniform_runs, uniform_metrics = zip(*uniform_vals)
        ax.plot(uniform_runs, uniform_metrics, marker='^', label='Uniform Crossover', 
                linewidth=2.5, markersize=8, color='#6A994E',
                markerfacecolor='#6A994E', markeredgewidth=1.5, markeredgecolor='white')
    
    ax.set_xlabel('Number of Runs', fontweight='normal')
    ax.set_ylabel(metric, fontweight='normal')
    ax.set_title(f'Impact of Crossover Operator on {metric}', fontweight='bold', pad=15)
    ax.legend(loc='best', framealpha=0.95)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('crossover_operator_comparison.png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print("✓ crossover_operator_comparison.png kaydedildi")
plt.show()

# ===== MUTATION OPERATOR  =====
print("\n" + "="*60)
print("5. MUTATION OPERATOR  (Reciprocal vs Inversion)")
print("="*60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, metric in enumerate(['Best Distance Found', 'Average Distance']):
    ax = axes[idx]
    
    recip_vals = []
    inver_vals = []
    
    for run in [10, 20, 30]:
        if run in run_data:
            df = run_data[run]
            
            recip_row = df[df['Experiment Name'] == 'Exp2_Reciprocal Mutation']
            inver_row = df[df['Experiment Name'] == 'Exp2_Inversion Mutation']
            
            if not recip_row.empty:
                recip_vals.append((run, float(recip_row[metric].values[0])))
            if not inver_row.empty:
                inver_vals.append((run, float(inver_row[metric].values[0])))
    
    if recip_vals:
        recip_runs, recip_metrics = zip(*recip_vals)
        ax.plot(recip_runs, recip_metrics, marker='o', label='Reciprocal Mutation', 
                linewidth=2.5, markersize=8, color='#C73E1D',
                markerfacecolor='#C73E1D', markeredgewidth=1.5, markeredgecolor='white')
    
    if inver_vals:
        inver_runs, inver_metrics = zip(*inver_vals)
        ax.plot(inver_runs, inver_metrics, marker='^', label='Inversion Mutation', 
                linewidth=2.5, markersize=8, color='#6A994E',
                markerfacecolor='#6A994E', markeredgewidth=1.5, markeredgecolor='white')
    
    ax.set_xlabel('Number of Runs', fontweight='normal')
    ax.set_ylabel(metric, fontweight='normal')
    ax.set_title(f'Impact of Mutation Operator on {metric}', fontweight='bold', pad=15)
    ax.legend(loc='best', framealpha=0.95)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('mutation_operator_comparison.png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print("✓ mutation_operator_comparison.png kaydedildi")
plt.show()

# ===== POPULATION SIZE VE ITERATIONS  =====
print("\n" + "="*60)
print("6. POPULATION SIZE & ITERATIONS (Constant Total Chromosomes = 100K)")
print("="*60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, metric in enumerate(['Best Distance Found', 'Average Distance']):
    ax = axes[idx]
    
    pop_iter_configs = {}  # {(pop, iter): {run: value}}
    
    for run in [10, 20, 30]:
        if run in run_data:
            df = run_data[run]
            exp_rows = df[df['Experiment Name'].str.contains('Exp6_')]
            
            for _, row in exp_rows.iterrows():
                pop = int(row['popSize'])
                iterations = int(row['nIters (Generations)'])
                metric_val = float(row[metric])
                
                config_key = f"Pop={pop}\nGen={iterations}"
                
                if config_key not in pop_iter_configs:
                    pop_iter_configs[config_key] = {}
                pop_iter_configs[config_key][run] = metric_val
    

    x_pos = np.arange(len(pop_iter_configs))
    bar_width = 0.25
    
    configs = sorted(pop_iter_configs.keys())
    
    run_10_vals = []
    run_20_vals = []
    run_30_vals = []
    
    for config in configs:
        run_10_vals.append(pop_iter_configs[config].get(10, 0))
        run_20_vals.append(pop_iter_configs[config].get(20, 0))
        run_30_vals.append(pop_iter_configs[config].get(30, 0))
    
    ax.bar(x_pos - bar_width, run_10_vals, bar_width, label='Runs = 10', 
           color='#2E86AB', alpha=0.85, edgecolor='#1F5A7E', linewidth=1.2)
    ax.bar(x_pos, run_20_vals, bar_width, label='Runs = 20', 
           color='#A23B72', alpha=0.85, edgecolor='#712850', linewidth=1.2)
    ax.bar(x_pos + bar_width, run_30_vals, bar_width, label='Runs = 30', 
           color='#F18F01', alpha=0.85, edgecolor='#B86A00', linewidth=1.2)
    
    ax.set_xlabel('Population Size & Generations (100K Total Chromosomes)', fontweight='normal')
    ax.set_ylabel(metric, fontweight='normal')
    ax.set_title(f'Impact of Population Size & Iterations on {metric}\n(Constant: Pop × Gen = 100K)', 
                 fontweight='bold', pad=15)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(configs, fontsize=9)
    ax.legend(loc='best', framealpha=0.95)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('population_iterations_comparison.png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print("✓ population_iterations_comparison.png kaydedildi")
plt.show()

print("\n" + "="*60)
print("✓ completeds!")
print("="*60)
print("\nfiles:")
print("  1. elitism_comparison.png")
print("  2. mutation_probability_comparison.png")
print("  3. crossover_probability_comparison.png")
print("  4. crossover_operator_comparison.png")
print("  5. mutation_operator_comparison.png")
print("  6. population_iterations_comparison.png")