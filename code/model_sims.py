
from dataclasses import dataclass
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
OUT = Path(r"D:\Economia\Agenda Pesquisa\fed_put\figs_loss")
OUT.mkdir(parents=True, exist_ok=True)

# %%

@dataclass
class Calibration:
    # Structural parameters
    beta: float = 0.97      # Discount rate
    sigma: float = 1.0      # Risk aversion / Intertemporal subs.
    kappa: float = 0.08     # Elasticity of inflation to output gap
    mu: float = 0.97        # Steady state P/D
    psi_ya_n: float = 1.0   # semielasticity of natural rate to technology

    # Shock persistence parameters
    rho_z: float = 0.80     # Demand
    rho_a: float = 0.90     # Technology
    rho_d: float = 0.90     # Dividend
    rho_u: float = 0.00     # Monetary

    # Shock standard deviations
    sd_z: float = 0.005
    sd_a: float = 0.005
    sd_d: float = 0.020
    sd_u: float = 0.002

    # Endogenous generation of natural rate parameters
    @property
    def chi_z(self) -> float:
        return 1.0 - self.rho_z

    @property
    def chi_a(self) -> float:
        return self.sigma * self.psi_ya_n * (self.rho_a - 1.0)

# %% Symmetric solution

# Symmetric solution denominator for state-variable coefs
def delta_generic(lmbda: float, phi_pi: float, phi_y: float, phi_m: float, cal: Calibration) -> float:
    term1 = (1.0 - cal.mu * lmbda)
    term2 = cal.kappa * (phi_pi - lmbda) + (1.0 - cal.beta * lmbda) * (phi_y + cal.sigma * (1.0 - lmbda))
    term3 = (1.0 - cal.beta * lmbda) * phi_m * cal.sigma * (1.0 - lmbda)
    return term1 * term2 + term3

def generic_state_coefficients(lmbda: float, eta: float, phi_pi: float, phi_y: float, phi_m: float, cal: Calibration):
    D = delta_generic(lmbda, phi_pi, phi_y, phi_m, cal)
    A_y = (1.0 - cal.beta * lmbda) * (1.0 - cal.mu * lmbda) * eta / D
    A_pi = cal.kappa * (1.0 - cal.mu * lmbda) * eta / D
    A_m = cal.sigma * (1.0 - lmbda) * (1.0 - cal.beta * lmbda) * eta / D
    return A_y, A_pi, A_m, D

def symmetric_coefficients(phi_pi: float, phi_y: float, phi_m: float, cal: Calibration) -> Dict[str, float]:
    a_z, b_z, c_z, Dz = generic_state_coefficients(cal.rho_z, cal.chi_z, phi_pi, phi_y, phi_m, cal)
    a_a, b_a, c_a, Da = generic_state_coefficients(cal.rho_a, cal.chi_a, phi_pi, phi_y, phi_m, cal)
    Dd = delta_generic(cal.rho_d, phi_pi, phi_y, phi_m, cal)
    a_d = -(1.0 - cal.beta * cal.rho_d) * phi_m * (1.0 - cal.mu) * cal.rho_d / Dd
    b_d = -cal.kappa * phi_m * (1.0 - cal.mu) * cal.rho_d / Dd
    c_d = ((1.0 - cal.mu) * cal.rho_d *
           (cal.kappa * (phi_pi - cal.rho_d) +
            (1.0 - cal.beta * cal.rho_d) * (phi_y + cal.sigma * (1.0 - cal.rho_d))) / Dd)
    Du = delta_generic(cal.rho_u, phi_pi, phi_y, phi_m, cal)
    a_u = -(1.0 - cal.beta * cal.rho_u) * (1.0 - cal.mu * cal.rho_u) / Du
    b_u = -cal.kappa * (1.0 - cal.mu * cal.rho_u) / Du
    c_u = -cal.sigma * (1.0 - cal.rho_u) * (1.0 - cal.beta * cal.rho_u) / Du
    return {
        "a_z": a_z, "b_z": b_z, "c_z": c_z,
        "a_a": a_a, "b_a": b_a, "c_a": c_a,
        "a_d": a_d, "b_d": b_d, "c_d": c_d,
        "a_u": a_u, "b_u": b_u, "c_u": c_u,
        "Delta_z": Dz, "Delta_a": Da, "Delta_d": Dd, "Delta_u": Du,
    }

def symmetric_stddevs(phi_pi: float, phi_y: float, phi_m: float, cal: Calibration) -> Dict[str, float]:
    coeff = symmetric_coefficients(phi_pi, phi_y, phi_m, cal)
    var_z = cal.sd_z**2 / (1.0 - cal.rho_z**2)
    var_a = cal.sd_a**2 / (1.0 - cal.rho_a**2)
    var_d = cal.sd_d**2 / (1.0 - cal.rho_d**2)
    var_u = cal.sd_u**2 / (1.0 - cal.rho_u**2)
    var_y = coeff["a_z"]**2 * var_z + coeff["a_a"]**2 * var_a + coeff["a_d"]**2 * var_d + coeff["a_u"]**2 * var_u
    var_pi = coeff["b_z"]**2 * var_z + coeff["b_a"]**2 * var_a + coeff["b_d"]**2 * var_d + coeff["b_u"]**2 * var_u
    var_m = coeff["c_z"]**2 * var_z + coeff["c_a"]**2 * var_a + coeff["c_d"]**2 * var_d + coeff["c_u"]**2 * var_u
    return {"sd_y": float(np.sqrt(var_y)), "sd_pi": float(np.sqrt(var_pi)), "sd_m": float(np.sqrt(var_m))}

def irf_symmetric(phi_pi: float, phi_y: float, phi_m: float, cal: Calibration,
                  horizon: int = 40, shock: str = "demand"):
    coeff = symmetric_coefficients(phi_pi, phi_y, phi_m, cal)
    h = np.arange(horizon + 1)
    z = np.zeros(horizon + 1); a = np.zeros(horizon + 1); d = np.zeros(horizon + 1); u = np.zeros(horizon + 1)
    if shock == "demand":
        shock_size = -cal.sd_z
        z = shock_size * (cal.rho_z ** h)
    elif shock == "technology":
        shock_size = -cal.sd_a
        a = shock_size * (cal.rho_a ** h)
    elif shock == "dividend":
        shock_size = -cal.sd_d
        d = shock_size * (cal.rho_d ** h)
    elif shock == "monetary":
        shock_size = -cal.sd_u
        u = shock_size * (cal.rho_u ** h)
    else:
        raise ValueError("shock must be one of: demand, technology, dividend, monetary")
    y = coeff["a_z"] * z + coeff["a_a"] * a + coeff["a_d"] * d + coeff["a_u"] * u
    pi = coeff["b_z"] * z + coeff["b_a"] * a + coeff["b_d"] * d + coeff["b_u"] * u
    m = coeff["c_z"] * z + coeff["c_a"] * a + coeff["c_d"] * d + coeff["c_u"] * u
    return {"h": h, "y": y, "pi": pi, "m": m}


# %% Asymmetric solution

def _solve_perfect_foresight_path(
        phi_pi: float, phi_y: float, phi_m: float,
        cal: Calibration, z0: float, a0: float, d0: float, u0: float,
        horizon: int = 100, asymmetric: bool = True, max_regime_iter: int = 240):
    h = np.arange(horizon + 1)
    z = z0 * (cal.rho_z ** h)
    a = a0 * (cal.rho_a ** h)
    d = d0 * (cal.rho_d ** h)
    u = u0 * (cal.rho_u ** h)
    rn = cal.chi_z * z + cal.chi_a * a

    gamma = np.zeros(horizon) if asymmetric else np.full(horizon, phi_m)
    x = np.zeros((horizon + 1, 3))
    B = np.array([
        [1.0, 1.0 / cal.sigma, 0.0],
        [0.0, cal.beta, 0.0],
        [0.0, 1.0, cal.mu],
    ], dtype=float)

    for _ in range(max_regime_iter):
        x[-1, :] = 0.0
        for t in range(horizon - 1, -1, -1):
            g = gamma[t]
            A = np.array([
                [1.0 + phi_y / cal.sigma, phi_pi / cal.sigma, g / cal.sigma],
                [-cal.kappa, 1.0, 0.0],
                [phi_y, phi_pi, 1.0 + g],
            ], dtype=float)
            c = np.array([
                (rn[t] - u[t]) / cal.sigma,
                0.0,
                rn[t] - u[t] + (1.0 - cal.mu) * d[t + 1],
            ], dtype=float)
            x[t, :] = np.linalg.solve(A, B @ x[t + 1, :] + c)

        if not asymmetric:
            break
        new_gamma = np.where(x[:-1, 2] < 0.0, phi_m, 0.0)
        if np.array_equal(new_gamma, gamma):
            break
        gamma = new_gamma
    return {"h": h, "y": x[:, 0], "pi": x[:, 1], "m": x[:, 2]}

def irf_asymmetric(phi_pi: float, phi_y: float, phi_m: float, cal: Calibration,
                   horizon: int = 80, shock: str = "demand"):
    z0 = a0 = d0 = u0 = 0.0
    if shock == "demand":
        shock_size = -cal.sd_z
        z0 = shock_size
    elif shock == "technology":
        shock_size = -cal.sd_a
        a0 = shock_size
    elif shock == "dividend":
        shock_size = -cal.sd_d
        d0 = shock_size
    elif shock == "monetary":
        shock_size = -cal.sd_u
        u0 = shock_size
    else:
        raise ValueError("shock must be one of: demand, technology, dividend, monetary")
    return _solve_perfect_foresight_path(phi_pi, phi_y, phi_m, cal, z0, a0, d0, u0, horizon=horizon, asymmetric=True)

def simulate_asymmetric(phi_pi: float, phi_y: float, phi_m: float, cal: Calibration,
                        periods: int=10000, burn_in: int = 1000, horizon: int = 80, seed: int = 0):
    rng = np.random.default_rng(seed)

    z = a = d = u = 0.0
    ys = np.empty(periods - burn_in)
    pis = np.empty(periods - burn_in)
    ms = np.empty(periods - burn_in)
    idx = 0

    for t in range(periods):
        sol = _solve_perfect_foresight_path(phi_pi, phi_y, phi_m, cal, z, a, d, u, horizon=horizon, asymmetric=True)
        y_t, pi_t, m_t = sol["y"][0], sol["pi"][0], sol["m"][0]
        if t >= burn_in:
            ys[idx] = y_t; pis[idx] = pi_t; ms[idx] = m_t; idx += 1
        z = cal.rho_z * z + rng.normal(scale=cal.sd_z)
        a = cal.rho_a * a + rng.normal(scale=cal.sd_a)
        d = cal.rho_d * d + rng.normal(scale=cal.sd_d)
        u = cal.rho_u * u + rng.normal(scale=cal.sd_u)
    return {"y": ys, "pi": pis, "m": ms}

def asymmetric_stddevs(phi_pi: float, phi_y: float, phi_m: float,cal: Calibration,
                       periods: int = 10000, burn_in: int = 1000, horizon: int = 80, seed: int = 0):
    sim = simulate_asymmetric(phi_pi, phi_y, phi_m, cal, periods=periods, burn_in=burn_in, horizon=horizon, seed=seed)
    return {"sd_y": float(np.std(sim["y"])), "sd_pi": float(np.std(sim["pi"])), "sd_m": float(np.std(sim["m"]))}


# %%  Plotters

def save_sd_plot(x, y1, y2, y3, y4, xlabel, ylabel, title, path):
    plt.figure(figsize=(8.2, 5.0))
    plt.plot(x, y1, linewidth=2, label=r"Symmetric, $\phi_y$ = 0")
    plt.plot(x, y2, linewidth=2, label=r"Symmetric, $\phi_y$ = 0.25")
    plt.plot(x, y3, linewidth=2, label=r"Asymmetric, $\phi_y$ = 0")
    plt.plot(x, y4, linewidth=2, label=r"Asymmetric, $\phi_y$ = 0.25")
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(path, dpi=180); plt.close()

def save_irf_plot(h, y1, y2, y3, y4, ylabel, label1, label2, label3, label4, title, path):
    plt.figure(figsize=(8.2, 5.0))
    plt.plot(h, y1, linewidth=2, label=label1)
    plt.plot(h, y2, linewidth=2, label=label2, linestyle='--')
    plt.plot(h, y3, linewidth=2, label=label3, linestyle=':')
    plt.plot(h, y4, linewidth=2, label=label4, linestyle='-.')
    plt.axhline(0.0, color="black", linewidth=0.5)
    plt.xlabel("Horizon"); plt.ylabel(ylabel); plt.title(title)
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(path, dpi=180); plt.close()



# %% Simulations

cal = Calibration()
phi_pi_baseline = 1.5
phi_y_baseline = 0.25
phi_y_simple = 0.00
phi_m_baseline = 0.25

phi_m_grid = np.linspace(0.0, 1.0, 21)

def sweep_stddevs_through_phi_m(phi_pi, phi_y, cal, seed=42):
    sym_list = []
    asym_list = []
    for phi_m in phi_m_grid:
        sym_list.append(symmetric_stddevs(phi_pi, phi_y, phi_m, cal))
        asym_list.append(asymmetric_stddevs(phi_pi, phi_y, phi_m, cal, seed=seed))
    sym = {k: np.array([d[k] for d in sym_list]) for k in ["sd_y", "sd_pi", "sd_m"]}
    asym = {k: np.array([d[k] for d in asym_list]) for k in ["sd_y", "sd_pi", "sd_m"]}
    return sym, asym

sd_sym_baseline, sd_asym_baseline = sweep_stddevs_through_phi_m(phi_pi_baseline, phi_y_baseline, cal)
sd_sym_simple, sd_asym_simple = sweep_stddevs_through_phi_m(phi_pi_baseline, phi_y_simple, cal)

# %%
# output gap sd figure, exploring change in phi_m
save_sd_plot(phi_m_grid, sd_sym_simple["sd_y"], sd_sym_baseline["sd_y"],
             sd_asym_simple['sd_y'], sd_asym_baseline['sd_y'],
             r"$\phi_m$",
             "Std. dev. of output gap",
             rf"Outputgap volatility vs $\phi_m$ ($\phi_\pi={phi_pi_baseline}$)",
               OUT / "phi_m_std_output_gap.png")

# inflation
save_sd_plot(phi_m_grid, sd_sym_simple["sd_pi"], sd_sym_baseline["sd_pi"],
             sd_asym_simple['sd_pi'], sd_asym_baseline['sd_pi'],
             r"$\phi_m$",
             "Std. dev. of inflation",
             rf"Inflation volatility vs $\phi_m$ ($\phi_\pi={phi_pi_baseline}$)",
               OUT / "phi_m_std_inflation.png")

# asset prices
save_sd_plot(phi_m_grid, sd_sym_simple["sd_m"], sd_sym_baseline["sd_m"],
             sd_asym_simple['sd_m'], sd_asym_baseline['sd_m'],
             r"$\phi_m$",
             "Std. dev. of asset price gap",
             rf"Asset price gap volatility vs $\phi_m$ ($\phi_\pi={phi_pi_baseline}$)",
               OUT / "phi_m_std_asset_prices.png")


# %%  Exploring standard deviation through variation in shock calibration for dividends

sd_d_grid = np.linspace(0.0, 0.02, 21)

def sweep_stddevs_thourgh_sd_d(phi_pi, phi_y, phi_m, cal, seed=42):
    sym_list = []
    asym_list = []
    for sd_d in sd_d_grid:
        cal.sd_d = sd_d
        sym_list.append(symmetric_stddevs(phi_pi, phi_y, phi_m, cal))
        asym_list.append(asymmetric_stddevs(phi_pi, phi_y, phi_m, cal, seed=seed))
    sym = {k: np.array([d[k] for d in sym_list]) for k in ["sd_y", "sd_pi", "sd_m"]}
    asym = {k: np.array([d[k] for d in asym_list]) for k in ["sd_y", "sd_pi", "sd_m"]}
    return sym, asym

sdd_sd_sym_baseline, sdd_sd_asym_baseline = sweep_stddevs_thourgh_sd_d(
    phi_pi_baseline, phi_y_baseline, phi_m_baseline, cal
    )

sdd_sd_sym_simple, sdd_sd_asym_simple = sweep_stddevs_thourgh_sd_d(
    phi_pi_baseline, phi_y_simple, phi_m_baseline, cal
    )

# output gap sd figure, exploring change in phi_m
save_sd_plot(sd_d_grid, sdd_sd_sym_simple["sd_y"], sdd_sd_sym_baseline["sd_y"],
             sdd_sd_asym_simple['sd_y'], sdd_sd_asym_baseline['sd_y'],
             r"$\sigma_d$",
             "Std. dev. of output gap",
             rf"Output-gap volatility vs $\sigma_d$ ($\phi_\pi={phi_pi_baseline}, \phi_m={phi_m_baseline}$)",
               OUT / "sdd_std_output_gap.png")

# inflation
save_sd_plot(sd_d_grid, sdd_sd_sym_simple["sd_pi"], sdd_sd_sym_baseline["sd_pi"],
             sdd_sd_asym_simple['sd_pi'], sdd_sd_asym_baseline['sd_pi'],
             r"$\sigma_d$",
             "Std. dev. of inflation",
             rf"inflation volatility vs $\sigma_d$ ($\phi_\pi={phi_pi_baseline}, \phi_m={phi_m_baseline}$)",
               OUT / "sdd_std_inflation.png")

# asset prices
save_sd_plot(sd_d_grid, sdd_sd_sym_simple["sd_m"], sdd_sd_sym_baseline["sd_m"],
             sdd_sd_asym_simple['sd_m'], sdd_sd_asym_baseline['sd_m'],
             r"$\sigma_d$",
             "Std. dev. of asset price gap",
             rf"Asset price volatility vs $\sigma_d$ ($\phi_\pi={phi_pi_baseline}, \phi_m={phi_m_baseline}$)",
               OUT / "sdd_std_asset_prices.png")

# %% Impulse responses
cal = Calibration()

def sweep_irf(phi_pi, phi_y, phi_m, cal):
    simple_dict = {}
    baseline_dict = {}
    pi_m_dict = {}
    all_dict = {}
    for shock_name in ["demand", "technology", 'monetary', 'dividend']:
        # Reacts to inflation
        irf_simple = irf_symmetric(phi_pi=phi_pi, phi_y=0, phi_m=0.0,
                                 cal=cal, horizon=40, shock=shock_name)
        # Reacts to inflation and y gap
        irf_bsl = irf_symmetric(phi_pi=phi_pi, phi_y=phi_y, phi_m=0.0,
                                 cal=cal, horizon=40, shock=shock_name)
        # Reacts to inflation and m
        irf_pi_m = irf_symmetric(phi_pi=phi_pi, phi_y=0, phi_m=phi_m_baseline,
                                 cal=cal, horizon=40, shock=shock_name)
        # Reacts to everything
        irf_all = irf_symmetric(phi_pi=phi_pi, phi_y=phi_y, phi_m=phi_m_baseline,
                                 cal=cal, horizon=40, shock=shock_name)

        simple_dict[shock_name] = irf_simple
        baseline_dict[shock_name] = irf_bsl
        pi_m_dict[shock_name] = irf_pi_m
        all_dict[shock_name] = irf_all
    return simple_dict, baseline_dict, pi_m_dict, all_dict

irf_simple, irf_bsl, irf_pi_m, irf_all = sweep_irf(
    phi_pi_baseline, phi_y_baseline, phi_m_baseline, cal
    )

for shock_name in ["demand", "technology", 'monetary', 'dividend']:
    for outcome, outcome_name in [('y', 'Output gap'),
                                  ('pi', 'Inflation'),
                                  ('m', 'Asset price gap')]:
        horizon = irf_bsl[shock_name]['h']

        simple = irf_simple[shock_name][outcome]
        bsl = irf_bsl[shock_name][outcome]
        pi_m = irf_pi_m[shock_name][outcome]
        _all = irf_all[shock_name][outcome]

        title = \
        rf'{outcome_name} IRF to a -1 s.d. {shock_name} shock under different rules ($\phi_\pi=1.5$)'
        ylabel = outcome_name
        label1 = r'($\phi_y, \phi_m$) = (0.0, 0.0)'
        label2 = rf'($\phi_y, \phi_m$) = ({phi_y_baseline}, 0.0)'
        label3 = rf'($\phi_y, \phi_m$) = (0.0, {phi_m_baseline})'
        label4 = rf'($\phi_y, \phi_m$) = ({phi_y_baseline}, {phi_m_baseline})'

        filename = f'irf_{outcome}_{shock_name}.png'
        save_irf_plot(
            h=horizon,
            y1=simple, label1=label1,
            y2=bsl, label2=label2,
            y3=pi_m, label3=label3,
            y4=_all, label4=label4,
            ylabel=ylabel, title=title, path = OUT / filename
            )


# %% sd_d = 0 --> reacting to asset prices is good
cal = Calibration()
cal.sd_d = 0  
no_d_irf_simple, no_d_irf_bsl, no_d_irf_pi_m, no_d_irf_all = sweep_irf(
phi_pi_baseline, phi_y_baseline, phi_m_baseline, cal
)


sd0_sym_baseline, sd0_asym_baseline = sweep_stddevs_through_phi_m(phi_pi_baseline, phi_y_baseline, cal)
sd0_sym_simple, sd0_asym_simple = sweep_stddevs_through_phi_m(phi_pi_baseline, phi_y_simple, cal)

# output gap sd figure, exploring change in phi_m
save_sd_plot(phi_m_grid, sd0_sym_simple["sd_y"], sd0_sym_baseline["sd_y"],
             sd0_asym_simple['sd_y'], sd0_asym_baseline['sd_y'],
             r"$\phi_m$",
             "Std. dev. of output gap absent financial shocks",
             rf"Output gap volatility vs $\phi_m$ ($\phi_\pi={phi_pi_baseline}$)",
               OUT / "phi_m_0sdd_std_output_gap.png")

# inflation
save_sd_plot(phi_m_grid, sd0_sym_simple["sd_pi"], sd0_sym_baseline["sd_pi"],
             sd0_asym_simple['sd_pi'], sd0_asym_baseline['sd_pi'],
             r"$\phi_m$",
             "Std. dev. of inflation absent financial shocks",
             rf"Inflation volatility vs $\phi_m$ ($\phi_\pi={phi_pi_baseline}$)",
               OUT / "phi_m_0sdd_std_inflation.png")

# asset prices
save_sd_plot(phi_m_grid, sd0_sym_simple["sd_m"], sd0_sym_baseline["sd_m"],
             sd0_asym_simple['sd_m'], sd0_asym_baseline['sd_m'],
             r"$\phi_m$",
             "Std. dev. of asset price gap absent financial shocks",
             rf"Asset price gap volatility vs $\phi_m$ ($\phi_\pi={phi_pi_baseline}$)",
               OUT / "phi_m_0sdd_std_asset_prices.png")

# %% Under monetary persistence
cal = Calibration()
cal.rho_u = 0.75

sd_sym_bsl_rho_u, sd_asym_bsl_rho_u = sweep_stddevs_through_phi_m(phi_pi_baseline, phi_y_baseline, cal)
sd_sym_simple_rho_u, sd_asym_simple_rho_u = sweep_stddevs_through_phi_m(phi_pi_baseline, phi_y_simple, cal)


# %% Loss function numerical exploration

# dimensions
# Fixed phi_y, phi_pi
# // increasing phi_m
# // loss value
# Series:
    # Symmetric reaction, loss doesnt take m into consideration
    # Asymmetric reaction, loss doesnt take m into consideration
    # Symmetric reaction, loss takes m into consideration
    # Asymmetric reaction, loss takes m into consideration
# Same chart with monetary persistence


def save_loss_plot(x, y1, y2, y3, y4, xlabel, ylabel, title, path):
    plt.figure(figsize=(8.2, 5.0))
    plt.plot(x, y1, linewidth=2, label=r"Symmetric, $l_m$ = 0.0")
    plt.plot(x, y2, linewidth=2, label=r"Symmetric, $l_m$ = 0.1")
    plt.plot(x, y3, linewidth=2, label=r"Asymmetric, $l_m$ = 0.0")
    plt.plot(x, y4, linewidth=2, label=r"Asymmetric, $l_m$ = 0.1")
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    plt.ylim(0.9, 1.6)
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(path, dpi=180); plt.close()



def sweep_loss_function(l_ms, sd_sym, sd_asym, l_y=1, l_pi=1):
    
    def loss_function(sd_y, sd_pi, sd_m, l_y, l_pi, l_m):
        loss_y = l_y*sd_y
        loss_pi = l_pi*sd_pi
        loss_m = l_m*sd_m
        loss = loss_y + loss_pi + loss_m
        try:  # if lists, normalize by first value
            rel_loss = loss / loss[0]
            return rel_loss
        except Exception:
            return loss
    
    losses_dict = {}
    for l_m in l_ms:
        loss_sym_bsl = loss_function(
            sd_sym['sd_y'],
            sd_sym['sd_pi'],
            sd_sym['sd_m'],
            l_y, l_pi, l_m)
        loss_asym_bsl = loss_function(
            sd_asym['sd_y'],
            sd_asym['sd_pi'],
            sd_asym['sd_m'],
            l_y, l_pi, l_m)
        losses_dict[f'sym_lm_{l_m}'] = loss_sym_bsl
        losses_dict[f'asym_lm_{l_m}'] = loss_asym_bsl
    
    return losses_dict

l_ms = [0, 0.1]
loss_lm_no_mon = sweep_loss_function(
    l_ms=l_ms,
    sd_sym=sd_sym_baseline,
    sd_asym=sd_asym_baseline
    )
# now rho_u
loss_lm_mon_pers =  sweep_loss_function(
    l_ms=l_ms,
    sd_sym=sd_sym_bsl_rho_u,
    sd_asym=sd_asym_bsl_rho_u
    )

# no reaction to GDP
loss_lm_no_mon_no_gdp = sweep_loss_function(
    l_ms=l_ms,
    sd_sym=sd_sym_simple,
    sd_asym=sd_asym_simple
    )

# mon perss, no reaction to gdp
loss_lm_mon_no_gdp = sweep_loss_function(
    l_ms=l_ms,
    sd_sym=sd_sym_simple_rho_u,
    sd_asym=sd_asym_simple_rho_u
    )

# no monetary pers.
save_loss_plot(x=phi_m_grid,
               y1=loss_lm_no_mon['sym_lm_0'],
               y2=loss_lm_no_mon['sym_lm_0.1'],
               y3=loss_lm_no_mon['asym_lm_0'],
               y4=loss_lm_no_mon['asym_lm_0.1'],
               xlabel=r"$\phi_m$",
               ylabel='L = ${y_{gap}}^2$ + ${\pi_{gap}}^2$ + $l_m$ ${m_{gap}}^2$',
               title="Relative loss vs $\phi_m$",
               path=OUT / "no_mon_pers.png")
             

# Monetary pers.
save_loss_plot(x=phi_m_grid,
               y1=loss_lm_mon_pers['sym_lm_0'],
               y2=loss_lm_mon_pers['sym_lm_0.1'],
               y3=loss_lm_mon_pers['asym_lm_0'],
               y4=loss_lm_mon_pers['asym_lm_0.1'],
               xlabel=r"$\phi_m$",
               ylabel='L = ${y_{gap}}^2$ + ${\pi_{gap}}^2$ + $l_m$ ${m_{gap}}^2$',
               title="Relative loss vs $\phi_m$ under monetary shock persistence",
               path=OUT / "mon_pers.png")


# No reaction to GDP
save_loss_plot(x=phi_m_grid,
               y1=loss_lm_no_mon_no_gdp['sym_lm_0'],
               y2=loss_lm_no_mon_no_gdp['sym_lm_0.1'],
               y3=loss_lm_no_mon_no_gdp['asym_lm_0'],
               y4=loss_lm_no_mon_no_gdp['asym_lm_0.1'],
               xlabel=r"$\phi_m$",
               ylabel='L = ${y_{gap}}^2$ + ${\pi_{gap}}^2$ + $l_m$ ${m_{gap}}^2$',
               title="Relative loss vs $\phi_m$ under $\phi_y$=0",
               path=OUT / "no_mon_pers_no_gdp.png")

# No reaction to GDP, monetary persistence
save_loss_plot(x=phi_m_grid,
               y1=loss_lm_mon_no_gdp['sym_lm_0'],
               y2=loss_lm_mon_no_gdp['sym_lm_0.1'],
               y3=loss_lm_mon_no_gdp['asym_lm_0'],
               y4=loss_lm_mon_no_gdp['asym_lm_0.1'],
               xlabel=r"$\phi_m$",
               ylabel='L = ${y_{gap}}^2$ + ${\pi_{gap}}^2$ + $l_m$ ${m_{gap}}^2$',
               title="Relative loss vs $\phi_m$ under $\phi_y$=0 and monetary persistence",
               path=OUT / "mon_pers_no_gdp.png")

