from __future__ import annotations

import copy
import numpy as np
from scipy.special import expit
from scipy.stats import multivariate_normal
from .distribution import sample_from_distribution, validate_distribution, covariance_from_parameters

SCENARIOS = ("same", "covariate", "translation", "add_remove", "relabel", "mixed")

DEFAULTS = {
    "same": {},
    "covariate": {"acceptance_floor": 0.10, "acceptance_scale": 0.80,"direction": [0.8, -0.5], "intercept": -0.3},
    "translation": {"shifts": {1: [-0.60, 0.35], 3: [0.45, -0.40], 5: [-0.40, 0.25]}},
    "add_remove": {"remove": [5], "add": [
        {"mean": [-1.85, 3.50], "cov_params": [0.74, 0.47, 26.0],
         "weight": 0.10, "label": 1, "regression_base_level": 0.40,
         "regression_local_amplitude": 0.15, "regression_gradient_angle_deg": 22.0}]},
    "relabel": {"flip_clusters": [2, 4]},
    "mixed": {"translations": {2: [0.35, -0.28]},
              "rotations_deg": {4: 10.0},
              "axis_multipliers": {0: [1.10, 1.00]},
              "weight_multipliers": {1: 1.22}},
}

DEFAULTS["mixed"] = {
    "add_remove": copy.deepcopy(DEFAULTS["add_remove"]),
    "translation": copy.deepcopy(DEFAULTS["translation"])
}
def _parameters(scenario: str, params: dict | None) -> dict:
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario {scenario}; choose one of {SCENARIOS}")

    return copy.deepcopy(DEFAULTS[scenario] if params is None else params)


def target_spec(source: dict, scenario: str, strength: float = 1.0,
                params: dict | None = None) -> tuple[dict, list[str]]:

    if not np.isfinite(strength) or strength < 0:
        raise ValueError("strength must be a finite, nonnegative number")
    cfg = _parameters(scenario, params)
    spec = copy.deepcopy(source)
    means = np.array(spec['means'], dtype=float)
    cov = np.array(spec['cov_params'], dtype=float)
    weights = np.array(spec['weights'], dtype=float)
    labels = np.array(spec['labels'], dtype=int)
    notes = []

    if scenario == 'same':
        notes = ['No parameter changes; new independent draws']
    elif scenario == 'covariate':

        floor = float(cfg["acceptance_floor"])
        scale = float(cfg["acceptance_scale"])
        width = float(cfg["boundary_width"])
    
        if not (0 < floor < 1 and 0 <= scale <= 1 - floor and width > 0):
            raise ValueError("Invalid covariate acceptance parameters")
    
        notes = [
            f"Boundary-focused covariate shift: width={width:.3f}",
            f"Acceptance floor={floor:.3f}, scale={scale:.3f}",
            "P_T(Y|X) = P_S(Y|X)"
        ]
        
        #a, b = float(cfg['acceptance_floor']), float(cfg['acceptance_scale'])
        #d = np.asarray(cfg['direction'], dtype=float)
        #intercept = float(cfg['intercept'])
        #if not (0 < a <= 1 and 0 <= b <= 1-a and d.shape == (2,)):
        #    raise ValueError('Require 0<acceptance_floor, 0<=acceptance_scale<=1-floor and direction length 2')
        #notes = [f"Acceptance a(x)={a:.3f}+{b:.3f} σ({strength:.2f}·({d[0]:+.2f}x1{d[1]:+.2f}x2{intercept:+.2f}))",
        #         'Feature-only selection: P_T(Y|X)=P_S(Y|X)']
    elif scenario == 'translation':
        for cluster, shift in cfg['shifts'].items():
            k = int(cluster)
            delta = strength * np.asarray(shift, dtype=float)
            if delta.shape != (2,): raise ValueError('Each translation must be length 2')
            means[k] += delta
            notes.append(f'C{k}: Δμ=({delta[0]:+.2f}, {delta[1]:+.2f})')
    elif scenario == 'add_remove':
        if strength > 0:
            for k in cfg.get('remove', []):
                k=int(k)
                weights[k] = 0.0
                notes.append(f'Remove C{k} (class {labels[k]})')
            for entry in cfg.get('add', []):
                new_idx = len(weights)
                means = np.vstack([means, np.asarray(entry['mean'], dtype=float)])
                cov = np.vstack([cov, np.asarray(entry['cov_params'], dtype=float)])
                labels = np.r_[labels, int(entry['label'])]
                weights = np.r_[weights, float(entry['weight'])]
                for field, key in [('regression_base_levels','regression_base_level'),
                                   ('regression_local_amplitudes','regression_local_amplitude'),
                                   ('regression_gradient_angles_deg','regression_gradient_angle_deg')]:
                    spec[field] = list(spec[field]) + [float(entry[key])]
                notes.append(f"Add C{new_idx}: class={entry['label']}, μ={entry['mean']}, "
                             f"axes/angle={entry['cov_params']}, raw weight={entry['weight']}")
        else:
            notes.append('strength=0: no components added or removed')
    elif scenario == 'relabel':
        ids = [int(k) for k in cfg['flip_clusters']]
        if strength > 0:
            for k in ids:
                before=int(labels[k]); labels[k]=1-before
                notes.append(f'C{k}: class {before}→{int(labels[k])}')
        else:
            notes.append('strength=0: no labels flipped')
            
    elif scenario == "mixed":

        if strength == 0:
            return copy.deepcopy(source), ["No target modifications"]
    
        # Stage 1: add/remove clusters
        intermediate, notes1 = target_spec(
            source,
            scenario="add_remove",
            strength=1.0,
            params=cfg["add_remove"]
        )
    
        # Stage 2: translate selected clusters
        final, notes2 = target_spec(
            intermediate,
            scenario="translation",
            strength=strength,
            params=cfg["translation"]
        )
    
        return final, (
            ["STAGE 1 — Cluster addition/removal"]
            + notes1
            + ["STAGE 2 — Cluster translations"]
            + notes2
        ) 
        
    elif scenario == 'mixed_o':
        for cluster, shift in cfg.get('translations', {}).items():
            k=int(cluster); delta=strength*np.asarray(shift,dtype=float)
            means[k]+=delta
            notes.append(f'C{k}: Δμ=({delta[0]:+.2f}, {delta[1]:+.2f})')
        for cluster, angle in cfg.get('rotations_deg', {}).items():
            k=int(cluster); rot=strength*float(angle)
            cov[k,2]+=rot
            notes.append(f'C{k}: principal rotation {rot:+.1f}°')
        for cluster, factors in cfg.get('axis_multipliers', {}).items():
            k=int(cluster); f=1+strength*(np.asarray(factors,dtype=float)-1)
            if np.any(f<=0): raise ValueError('Covariance axis multipliers must remain positive')
            cov[k,:2]*=f
            notes.append(f'C{k}: principal std multipliers ({f[0]:.3f}, {f[1]:.3f})')
        for cluster, factor in cfg.get('weight_multipliers', {}).items():
            k=int(cluster); f=1+strength*(float(factor)-1)
            if f<0: raise ValueError('Weight multiplier must remain nonnegative')
            weights[k]*=f
            notes.append(f'C{k}: weight multiplier ×{f:.3f}')
    
    if scenario not in ('same','covariate'):
        if weights.sum() <= 0: raise ValueError('All mixture weights removed')
        spec['means'] = means.tolist()
        spec['cov_params'] = cov.tolist()
        spec['labels'] = labels.tolist()
        spec['weights'] = (weights / weights.sum()).tolist()
    validate_distribution(spec)
    return spec, notes


def sample_target(source: dict, scenario: str, n: int, seed: int,
                  strength: float = 1.0, params: dict | None = None):
    """Return X, classification y, regression y, cluster IDs, target spec, notes."""
    spec, notes = target_spec(source, scenario, strength=strength, params=params)
    if scenario != 'covariate':
        return (*sample_from_distribution(spec, n=n, seed=seed), spec, notes)

    cfg = _parameters(scenario, params)
    #floor=float(cfg['acceptance_floor'])
    #scale=float(cfg['acceptance_scale'])
    #d=np.asarray(cfg['direction'],dtype=float)
    #intercept=float(cfg['intercept'])
        
    floor = float(cfg["acceptance_floor"])
    scale = float(cfg["acceptance_scale"])  
    width = float(cfg["boundary_width"])


    rng=np.random.default_rng(seed)
    Xs, ys, rs, cs = [], [], [], []
    remaining = n
    iteration = 0
    while remaining:
        size=max(320, int(remaining*3.6))
        X,y,r,c=sample_from_distribution(source,size,seed=int(rng.integers(1,2**31-1)))
        #accept=floor+scale*expit(strength*(X@d+intercept))
        p = bayes_probability(source, X)

        accept = floor + scale * np.exp(
            -0.5 * (strength * (p - 0.5) / width) ** 2
        )
        chosen=np.flatnonzero(rng.random(size)<accept)[:remaining]
        Xs.append(X[chosen]);ys.append(y[chosen]);rs.append(r[chosen]);cs.append(c[chosen])
        remaining-=len(chosen)
        iteration+=1
        if iteration > 10000: raise RuntimeError('Covariate sampler did not converge')
    return np.concatenate(Xs),np.concatenate(ys),np.concatenate(rs),np.concatenate(cs),spec,notes


def bayes_probability(spec: dict, X: np.ndarray) -> np.ndarray:

    X=np.atleast_2d(X)
    density=np.column_stack([
        w*multivariate_normal.pdf(X,mean=mu,cov=covariance_from_parameters(*par))
        for w,mu,par in zip(spec['weights'],spec['means'],spec['cov_params'])
    ])
    return (density@np.asarray(spec['labels'],dtype=float))/np.maximum(density.sum(axis=1),1e-300)
