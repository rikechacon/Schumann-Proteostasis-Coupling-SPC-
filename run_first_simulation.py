"""
PRIMERA SIMULACIÓN - Schumann-Proteostasis Coupling
====================================================
Ejecuta una simulación básica y genera visualizaciones iniciales.
"""

import sys
sys.path.insert(0, 'src')
import numpy as np
import matplotlib.pyplot as plt
from model import SchumannProteostasisModel

print("\n" + "="*70)
print("PRIMERA SIMULACIÓN: Schumann-Proteostasis Coupling")
print("="*70 + "\n")

# Configuración de parámetros para la primera ejecución
params = {
    'eta': 0.5,           # Acoplamiento medio
    'sigma': 0.3,         # Ruido moderado
    'A_schumann': 0.5,    # Amplitud del campo
    'T': 10.0,            # 10 segundos de simulación
    'dt': 0.001,          # 1 ms de resolución
    'n_trials': 50,       # 50 trayectorias
    'seed': 42            # Reproducibilidad
}

print("1. Inicializando modelo...")
model = SchumannProteostasisModel(params)

print("2. Calculando atractores del potencial...")
well_healthy, well_pathological = model.find_well_positions()
print(f"   ✓ Pozo saludable: x = {well_healthy:.3f}")
print(f"   ✓ Pozo patológico: x = {well_pathological:.3f}")

print("3. Simulando trayectorias (Euler-Maruyama)...")
trajectories = model.simulate_euler_maruyama(x0=well_healthy, n_trials=params['n_trials'])
print(f"   ✓ {params['n_trials']} trayectorias generadas")

print("4. Calculando métricas de estabilidad...")
mfpt, success_rate = model.calculate_mfpt(trajectories, threshold=0.0)
print(f"   ✓ MFPT: {mfpt:.3f} segundos")
print(f"   ✓ Tasa de transición: {success_rate:.1%}")

# ============================================================================
# GENERACIÓN DE GRÁFICOS
# ============================================================================

print("\n5. Generando visualizaciones...")

# Guardar en carpeta results/figures
import os
os.makedirs('results/figures', exist_ok=True)

# --- Gráfico 1: Paisaje Energético ---
print("   • Potencial de doble pozo...")
x = np.linspace(-2, 2, 500)
V = model.potential(x)

plt.figure(figsize=(8, 5))
plt.plot(x, V, 'b-', linewidth=2, label='V(x)')
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.axvline(x=0, color='k', linestyle=':', alpha=0.3)
plt.plot(well_healthy, model.potential(well_healthy), 'go', markersize=8, label='Saludable')
plt.plot(well_pathological, model.potential(well_pathological), 'ro', markersize=8, label='Patológico')
plt.xlabel('Estado proteostático x')
plt.ylabel('Energía potencial V(x)')
plt.title('Paisaje Energético de Proteostasis')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/figures/01_potential.png', dpi=300)
plt.close()

# --- Gráfico 2: Trayectorias ---
print("   • Trayectorias estocásticas...")
plt.figure(figsize=(10, 6))
for i in range(min(15, trajectories.shape[0])):
    plt.plot(model.t, trajectories[i, :], alpha=0.5, linewidth=0.8)
plt.axhline(y=well_healthy, color='g', linestyle='--', alpha=0.5, label='Pozo saludable')
plt.axhline(y=well_pathological, color='r', linestyle='--', alpha=0.5, label='Pozo patológico')
plt.axhline(y=0, color='k', linestyle=':', alpha=0.3)
plt.xlabel('Tiempo (s)')
plt.ylabel('Estado x(t)')
plt.title(f'Trayectorias (η={params["eta"]}, σ={params["sigma"]})')
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/figures/02_trajectories.png', dpi=300)
plt.close()

# --- Gráfico 3: Campo de Schumann ---
print("   • Campo de Schumann (7.83 Hz)...")
t_short = np.linspace(0, 1, 500)
field = model.schumann_field(t_short)

plt.figure(figsize=(8, 4))
plt.plot(t_short, field, 'b-', linewidth=2)
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.title(f'Resonancia de Schumann: f = {model.f_SR} Hz')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/figures/03_schumann_field.png', dpi=300)
plt.close()

print("\n" + "="*70)
print("✅ SIMULACIÓN COMPLETADA")
print("="*70)
print(f"\n📊 Resultados guardados en: results/figures/")
print(f"📈 Métricas clave:")
print(f"   • MFPT: {mfpt:.3f} s (mayor = más estable)")
print(f"   • Transiciones: {success_rate:.1%}")
print(f"\n💡 Interpretación:")
if mfpt > 5.0:
    print("   → Sistema ALTAMENTE ESTABLE bajo estas condiciones")
elif mfpt > 2.0:
    print("   → Sistema MODERADAMENTE ESTABLE")
else:
    print("   → Sistema PROPENSO a transiciones (investigar parámetros)")
print("\n" + "="*70 + "\n")
