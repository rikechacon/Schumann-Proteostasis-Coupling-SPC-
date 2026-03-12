"""
Módulo de barrido paramétrico para análisis de resonancia estocástica.
Explora el espacio (η, σ) para identificar regímenes de estabilidad óptima.
"""

import numpy as np
import pandas as pd
import json
import os
from tqdm import tqdm
from src.model import SchumannProteostasisModel


def sweep_eta_sigma(
    eta_range=None,
    sigma_range=None,
    base_params=None,
    n_trials=30,
    output_file=None
):
    """
    Ejecuta barrido bidimensional sobre η (acoplamiento) y σ (ruido).
    
    Args:
        eta_range: Lista o array de valores para η [0, 1]
        sigma_range: Lista o array de valores para σ
        base_params: Parámetros base del modelo
        n_trials: Trayectorias por combinación de parámetros
        output_file: Ruta para guardar resultados CSV/JSON
    
    Returns:
        results_df: DataFrame con métricas por combinación de parámetros
    """
    # Rangos por defecto
    if eta_range is None:
        eta_range = np.linspace(0.0, 1.0, 11)
    if sigma_range is None:
        sigma_range = np.linspace(0.05, 0.8, 11)
    if base_params is None:
        base_params = {'T': 5.0, 'dt': 0.001, 'seed': 42}
    
    results = []
    total_combinations = len(eta_range) * len(sigma_range)
    
    print(f"🔍 Iniciando barrido paramétrico: {total_combinations} combinaciones")
    print(f"   η: {len(eta_range)} valores, σ: {len(sigma_range)} valores\n")
    
    for i, eta in enumerate(eta_range):
        for j, sigma in enumerate(sigma_range):
            # Configurar parámetros específicos
            params = base_params.copy()
            params.update({
                'eta': float(eta),
                'sigma': float(sigma),
                'n_trials': n_trials,
                'seed': 42 + i*100 + j  # Semilla única por combinación
            })
            
            # Inicializar modelo y simular
            model = SchumannProteostasisModel(params)
            well_healthy, _ = model.find_well_positions()
            
            trajectories = model.simulate_euler_maruyama(
                x0=well_healthy, 
                n_trials=n_trials
            )
            
            # Calcular métricas
            mfpt, success_rate = model.calculate_mfpt(trajectories, threshold=0.0)
            
            # Almacenar resultado
            results.append({
                'eta': eta,
                'sigma': sigma,
                'mfpt': float(mfpt) if np.isfinite(mfpt) else 999.0,  # Manejar infinito
                'success_rate': float(success_rate),
                'n_transitions': int(success_rate * n_trials)
            })
            
            # Progreso
            if (i * len(sigma_range) + j + 1) % 10 == 0:
                print(f"   Progreso: {(i * len(sigma_range) + j + 1)}/{total_combinations} combinaciones")
    
    # Convertir a DataFrame
    results_df = pd.DataFrame(results)
    
    # Guardar si se especifica archivo
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        if output_file.endswith('.csv'):
            results_df.to_csv(output_file, index=False)
        elif output_file.endswith('.json'):
            results_df.to_json(output_file, orient='records', indent=2)
        print(f"💾 Resultados guardados en: {output_file}")
    
    return results_df


def find_optimal_region(results_df, mfpt_threshold=None):
    """
    Identifica la región óptima de parámetros que maximiza la estabilidad.
    
    Args:
        results_df: DataFrame con resultados del barrido
        mfpt_threshold: Umbral para definir "alta estabilidad" (por defecto: percentil 75)
    
    Returns:
        optimal_params: Diccionario con rango óptimo de η y σ
    """
    if mfpt_threshold is None:
        mfpt_threshold = results_df['mfpt'].quantile(0.75)
    
    # Filtrar combinaciones de alta estabilidad
    stable_region = results_df[results_df['mfpt'] >= mfpt_threshold]
    
    if len(stable_region) == 0:
        return None
    
    optimal_params = {
        'eta_min': float(stable_region['eta'].min()),
        'eta_max': float(stable_region['eta'].max()),
        'sigma_min': float(stable_region['sigma'].min()),
        'sigma_max': float(stable_region['sigma'].max()),
        'eta_optimal': float(stable_region.loc[stable_region['mfpt'].idxmax(), 'eta']),
        'sigma_optimal': float(stable_region.loc[stable_region['mfpt'].idxmax(), 'sigma']),
        'mfpt_max': float(stable_region['mfpt'].max()),
        'n_combinations': len(stable_region)
    }
    
    return optimal_params


def analyze_stochastic_resonance(results_df, fixed_eta=0.5):
    """
    Analiza la curva de resonancia estocástica para un η fijo.
    
    Args:
        results_df: DataFrame con resultados
        fixed_eta: Valor de η para el análisis (por defecto: 0.5)
    
    Returns:
        resonance_data: Diccionario con datos de la curva SR
    """
    # Filtrar por η cercano al valor fijo
    tolerance = 0.05
    subset = results_df[np.abs(results_df['eta'] - fixed_eta) < tolerance]
    
    if len(subset) == 0:
        return None
    
    # Ordenar por σ
    subset = subset.sort_values('sigma')
    
    # Encontrar óptimo de resonancia (máximo MFPT)
    idx_optimal = subset['mfpt'].idxmax()
    
    resonance_data = {
        'sigma_values': subset['sigma'].tolist(),
        'mfpt_values': subset['mfpt'].tolist(),
        'success_rates': subset['success_rate'].tolist(),
        'sigma_optimal': float(subset.loc[idx_optimal, 'sigma']),
        'mfpt_optimal': float(subset.loc[idx_optimal, 'mfpt']),
        'fixed_eta': fixed_eta
    }
    
    return resonance_data
