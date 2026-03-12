"""
GENERADOR DE FIGURAS DE PUBLICACIÓN
====================================
Crea visualizaciones en alta resolución para el manuscrito.
"""

import sys
sys.path.insert(0, 'src')
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Configuración para publicación
PUBLICATION_CONFIG = {
    'dpi': 600,
    'font_size': 10,
    'linewidth': 1.5,
    'figsize_single': (3.5, 2.8),
    'figsize_double': (7.2, 3.5),
}

def setup_publication_style():
    """Aplica configuración de estilo para publicación."""
    plt.rcParams.update({
        'font.size': PUBLICATION_CONFIG['font_size'],
        'axes.linewidth': PUBLICATION_CONFIG['linewidth'],
        'lines.linewidth': PUBLICATION_CONFIG['linewidth'],
        'figure.dpi': PUBLICATION_CONFIG['dpi'],
        'savefig.dpi': PUBLICATION_CONFIG['dpi'],
        'savefig.bbox': 'tight',
        'savefig.format': 'png',
    })

def create_publication_figure_1(model, output_path=None):
    """
    Crea Figura 1 para el manuscrito: Paisaje energético + trayectorias.
    """
    setup_publication_style()
    
    fig, axes = plt.subplots(1, 2, figsize=PUBLICATION_CONFIG['figsize_double'])
    
    # Panel A: Potencial de doble pozo
    ax = axes[0]
    x = np.linspace(-2, 2, 300)
    V = model.potential(x)
    
    ax.plot(x, V, 'k-', linewidth=1.5, label='V(x)')
    
    well1, well2 = model.find_well_positions()
    ax.plot(well1, model.potential(well1), 'o', color='#2ecc71', markersize=8, 
            markeredgecolor='black', markeredgewidth=0.5, label='Homeostasis')
    ax.plot(well2, model.potential(well2), 'o', color='#e74c3c', markersize=8, 
            markeredgecolor='black', markeredgewidth=0.5, label='Agregación')
    
    ax.set_xlabel('Estado proteostático, x')
    ax.set_ylabel('Energía potencial, V(x)')
    ax.set_title('A) Paisaje de estabilidad', fontweight='bold', fontsize=10)
    ax.legend(fontsize=8, frameon=True, framealpha=0.9)
    ax.grid(True, alpha=0.2, linestyle='--')
    
    # Panel B: Trayectorias estocásticas
    ax = axes[1]
    params_plot = {'eta': 0.5, 'sigma': 0.3, 'T': 10, 'dt': 0.001, 'n_trials': 20}
    model_plot = type(model)(params_plot)
    well1_plot, _ = model_plot.find_well_positions()
    trajectories = model_plot.simulate_euler_maruyama(x0=well1_plot, n_trials=20)
    
    for i, traj in enumerate(trajectories):
        alpha = 0.3 + 0.4 * (i / len(trajectories))
        ax.plot(model_plot.t, traj, color='steelblue', linewidth=0.5, alpha=alpha)
    
    ax.axhline(y=well1_plot, color='#2ecc71', linestyle='--', linewidth=1, alpha=0.7)
    ax.axhline(y=well2, color='#e74c3c', linestyle='--', linewidth=1, alpha=0.7)
    ax.axhline(y=0, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
    
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Estado, x(t)')
    ax.set_title('B) Dinámica estocástica', fontweight='bold', fontsize=10)
    ax.set_ylim(-2.5, 2.5)
    ax.grid(True, alpha=0.2, linestyle='--')
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path + '.png', dpi=600, bbox_inches='tight')
        plt.savefig(output_path + '.pdf', bbox_inches='tight')
        plt.savefig(output_path + '.svg', bbox_inches='tight')
        print(f"✓ Figura 1 guardada: {output_path}.[png|pdf|svg]")
    
    return fig

def create_publication_figure_2(results_df, output_path=None):
    """
    Crea Figura 2 para el manuscrito: Heatmap de estabilidad + curva SR.
    """
    setup_publication_style()
    
    fig, axes = plt.subplots(1, 2, figsize=PUBLICATION_CONFIG['figsize_double'])
    
    # Panel A: Heatmap de MFPT
    ax = axes[0]
    pivot = results_df.pivot(index='eta', columns='sigma', values='mfpt')
    pivot_masked = pivot.mask(pivot > 50)
    
    im = ax.pcolormesh(pivot.columns, pivot.index, pivot_masked, 
                      cmap='viridis', shading='auto')
    
    # Marcar óptimo
    idx_opt = pivot_masked.stack().idxmax()
    ax.plot(idx_opt[1], idx_opt[0], 'r*', markersize=15, markeredgecolor='white', 
            markeredgewidth=1.5, label='Óptimo')
    
    ax.set_xlabel('Ruido, σ')
    ax.set_ylabel('Acoplamiento, η')
    ax.set_title('A) Estabilidad en espacio de parámetros', fontweight='bold', fontsize=10)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('MFPT (s)', fontsize=8)
    ax.legend(fontsize=7, frameon=True, framealpha=0.9)
    
    # Panel B: Curva de resonancia estocástica
    ax = axes[1]
    subset = results_df[np.abs(results_df['eta'] - 0.5) < 0.05].sort_values('sigma')
    
    ax.plot(subset['sigma'], subset['mfpt'], 'bo-', linewidth=1.5, 
            markersize=4, markerfacecolor='white', markeredgecolor='blue', label='MFPT')
    
    # Marcar óptimo
    idx_max = subset['mfpt'].idxmax()
    ax.plot(subset.loc[idx_max, 'sigma'], subset.loc[idx_max, 'mfpt'], 
            'r*', markersize=15, markeredgecolor='white', markeredgewidth=1.5)
    
    ax.set_xlabel('Intensidad de ruido, σ')
    ax.set_ylabel('Mean First Passage Time (s)')
    ax.set_title('B) Resonancia estocástica (η=0.5)', fontweight='bold', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=8, frameon=True, framealpha=0.9)
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path + '.png', dpi=600, bbox_inches='tight')
        plt.savefig(output_path + '.pdf', bbox_inches='tight')
        plt.savefig(output_path + '.svg', bbox_inches='tight')
        print(f"✓ Figura 2 guardada: {output_path}.[png|pdf|svg]")
    
    return fig

def plot_potential_3d(model, output_path=None):
    """
    Genera visualización 3D del paisaje energético.
    """
    setup_publication_style()
    
    resolution = 100
    x = np.linspace(-2, 2, resolution)
    y = np.linspace(-2, 2, resolution)
    X, Y = np.meshgrid(x, y)
    
    Z = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            Z[i, j] = model.potential(X[i, j])
    
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.9, 
                          linewidth=0, antialiased=True)
    
    well1, well2 = model.find_well_positions()
    ax.scatter([well1, well2], [0, 0], 
              [model.potential(well1), model.potential(well2)], 
              c=['green', 'red'], s=100, edgecolors='black', linewidth=1.5)
    
    ax.set_xlabel('Estado x', fontsize=8)
    ax.set_ylabel('Dimensión auxiliar', fontsize=8)
    ax.set_zlabel('Energía V(x)', fontsize=8)
    ax.set_title('Paisaje 3D', fontsize=10, fontweight='bold', pad=10)
    
    ax.view_init(elev=25, azim=45)
    
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label('Energía (kT)', fontsize=7)
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path + '.png', dpi=600, bbox_inches='tight')
        plt.savefig(output_path + '.pdf', bbox_inches='tight')
        print(f"✓ Figura S1 guardada: {output_path}.[png|pdf]")
    
    return fig

# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    from src.model import SchumannProteostasisModel
    
    print("\n" + "="*70)
    print("GENERADOR DE FIGURAS DE PUBLICACIÓN")
    print("="*70 + "\n")
    
    # Crear directorio de salida
    os.makedirs('paper/figures', exist_ok=True)
    
    # Modelo base
    model = SchumannProteostasisModel({'eta': 0.5, 'sigma': 0.3, 'T': 10, 'dt': 0.001})
    
    # Figura 1: Paisaje + Trayectorias
    print("1. Generando Figura 1: Paisaje energético y dinámica...")
    fig1_path = 'paper/figures/fig01_potential_trajectories'
    fig1 = create_publication_figure_1(model, output_path=fig1_path)
    plt.close(fig1)
    
    # Figura 2: Heatmap + Resonancia estocástica
    print("2. Generando Figura 2: Mapa de parámetros y resonancia...")
    results_path = 'results/processed/sweep_results.csv'
    if os.path.exists(results_path):
        results_df = pd.read_csv(results_path)
        fig2_path = 'paper/figures/fig02_parameter_space'
        fig2 = create_publication_figure_2(results_df, output_path=fig2_path)
        plt.close(fig2)
    else:
        print("   ⚠️  Archivo de barrido no encontrado.")
    
    # Figura S1: Paisaje 3D (suplementario)
    print("3. Generando Figura S1: Paisaje energético 3D...")
    fig3_path = 'paper/figures/figS1_potential_3d'
    fig3 = plot_potential_3d(model, output_path=fig3_path)
    plt.close(fig3)
    
    # Resumen
    print("\n" + "="*70)
    print("✅ FIGURAS DE PUBLICACIÓN GENERADAS")
    print("="*70)
    print(f"\n📁 Archivos en paper/figures/:")
    print(f"   • fig01_potential_trajectories.[png|pdf|svg]")
    print(f"   • fig02_parameter_space.[png|pdf|svg]")
    print(f"   • figS1_potential_3d.[png|pdf]")
    
    print(f"\n📋 Especificaciones:")
    print(f"   • Resolución: 600 DPI")
    print(f"   • Formatos: PNG, PDF (vectorial), SVG (vectorial)")
    print(f"   • Listas para inserción en LaTeX")
    
    print("\n" + "="*70 + "\n")
