"""
Módulo principal del modelo Schumann-Proteostasis.
"""

import numpy as np
from scipy.optimize import minimize_scalar


class SchumannProteostasisModel:
    """Modelo de acoplamiento Schumann-Proteostasis."""
    
    def __init__(self, params=None):
        self.params = {
            'a': -1.0, 'b': 1.0, 'mu': 0.0,
            'gamma': 1.0, 'dt': 0.001, 'T': 10.0,
            'f_schumann': 7.83, 'A_schumann': 0.5, 'phi': 0.0,
            'eta': 0.5, 'tau_water': 0.1,
            'sigma': 0.3, 'kT': 0.1, 'seed': 42
        }
        if params:
            self.params.update(params)
        self.setup_simulation()
    
    def setup_simulation(self):
        self.dt = self.params['dt']
        self.T = self.params['T']
        self.n_steps = int(self.T / self.dt)
        self.t = np.linspace(0, self.T, self.n_steps)
        self.f_SR = self.params['f_schumann']
        self.A_SR = self.params['A_schumann']
        self.phi = self.params['phi']
        self.eta = self.params['eta']
        np.random.seed(self.params['seed'])
        self.sigma = self.params['sigma']
        self.kT = self.params['kT']
        self.gamma = self.params['gamma']
    
    def potential(self, x):
        a, b, mu = self.params['a'], self.params['b'], self.params['mu']
        return 0.5 * a * x**2 + 0.25 * b * x**4 - mu * x
    
    def potential_derivative(self, x):
        a, b, mu = self.params['a'], self.params['b'], self.params['mu']
        return a * x + b * x**3 - mu
    
    def schumann_field(self, t):
        return self.A_SR * np.sin(2 * np.pi * self.f_SR * t + self.phi)
    
    def drift_term(self, x, t):
        dV_dx = self.potential_derivative(x)
        F_SR = self.schumann_field(t)
        coupling_term = self.eta * F_SR
        return (-dV_dx + coupling_term) / self.gamma
    
    def simulate_euler_maruyama(self, x0=0.0, n_trials=1):
        x = np.zeros((n_trials, self.n_steps))
        x[:, 0] = x0
        D = np.sqrt(2 * self.kT * self.gamma / self.dt)
        
        for i in range(1, self.n_steps):
            t = self.t[i-1]
            drift = self.drift_term(x[:, i-1], t)
            dW = np.random.normal(0, 1, n_trials)
            x[:, i] = x[:, i-1] + drift * self.dt + D * dW * self.dt
        
        return x
    
    def calculate_mfpt(self, trajectories, threshold=0.0):
        n_trials = trajectories.shape[0]
        first_passage_times = []
        
        for trial in range(n_trials):
            crossings = np.where(trajectories[trial, :] > threshold)[0]
            if len(crossings) > 0:
                first_passage_times.append(self.t[crossings[0]])
        
        if len(first_passage_times) == 0:
            return np.inf, 0.0
        
        mfpt = np.mean(first_passage_times)
        success_rate = len(first_passage_times) / n_trials
        return mfpt, success_rate
    
    def find_well_positions(self):
        res1 = minimize_scalar(self.potential, bounds=(-2, 0), method='bounded')
        res2 = minimize_scalar(self.potential, bounds=(0, 2), method='bounded')
        return res1.x, res2.x


    def schumann_field_filtered(self, t, tau_filter=0.02):
        """
        Campo de Schumann con filtro biológico de primer orden.
        
        Modela la respuesta frecuencial del agua vicinal/membrana.
        tau_filter: tiempo de relajación típico (~20 ms para membranas)
        """
        omega = 2 * np.pi * self.f_SR
        # Ganancia del filtro paso-bajo: máxima en bajas frecuencias
        gain = 1 / np.sqrt(1 + (omega * tau_filter)**2)
        # Desfase asociado
        phase_shift = -np.arctan(omega * tau_filter)
        
        return self.A_SR * gain * np.sin(2 * np.pi * self.f_SR * t + self.phi + phase_shift)


    def schumann_field_filtered(self, t, tau_filter=0.02):
        """
        Campo de Schumann con filtro biológico de primer orden.
        
        Modela la respuesta frecuencial del agua vicinal/membrana.
        tau_filter: tiempo de relajación típico (~20 ms para membranas)
        
        El filtro paso-bajo atenúa frecuencias altas, creando
        especificidad para frecuencias bajas (theta/alpha).
        """
        omega = 2 * np.pi * self.f_SR
        # Ganancia del filtro paso-bajo: máxima en bajas frecuencias
        gain = 1 / np.sqrt(1 + (omega * tau_filter)**2)
        # Desfase asociado al filtro
        phase_shift = -np.arctan(omega * tau_filter)
        
        return self.A_SR * gain * np.sin(2 * np.pi * self.f_SR * t + self.phi + phase_shift)


    def resonant_coupling_term(self, t, f_neural=8.0, Q_factor=3.0):
        """
        Término de acoplamiento con resonancia frecuencial.
        
        Modela la interacción entre campo externo y oscilaciones neuronales
        intrínsecas mediante un factor de calidad Lorentziano.
        
        Parámetros:
        -----------
        f_neural : float
            Frecuencia de oscilación neuronal intrínseca (Hz)
            - Theta: 4-8 Hz
            - Alpha: 8-12 Hz
            - Beta: 12-30 Hz
        Q_factor : float
            Factor de calidad de la resonancia (anchura del pico)
            - Q=1: resonancia ancha (poco selectiva)
            - Q=3: resonancia moderada (balance)
            - Q=10: resonancia estrecha (muy selectiva)
        
        Retorna:
        --------
        F_resonant : float
            Amplitud del acoplamiento modulado por resonancia
        """
        # Factor de resonancia Lorentziano
        # Máximo cuando f_SR == f_neural
        detuning = self.f_SR - f_neural
        bandwidth = f_neural / Q_factor
        resonance_factor = 1.0 / (1.0 + (detuning / bandwidth)**2)
        
        # Campo base
        F_base = self.schumann_field(t)
        
        # Acoplamiento resonante
        F_resonant = self.eta * resonance_factor * F_base
        
        return F_resonant, resonance_factor


    def drift_term_resonant(self, x, t, f_neural=8.0, Q_factor=3.0):
        """
        Término de deriva con acoplamiento resonante.
        Reemplaza drift_term cuando se usa resonancia.
        """
        dV_dx = self.potential_derivative(x)
        
        # Obtener acoplamiento resonante y factor de resonancia
        F_resonant, _ = self.resonant_coupling_term(t, f_neural, Q_factor)
        
        return (-dV_dx + F_resonant) / self.gamma
