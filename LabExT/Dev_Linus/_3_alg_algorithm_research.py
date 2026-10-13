import csv
import re

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator

from scipy.optimize import curve_fit
from scipy.linalg import cho_factor, cho_solve
from scipy.special import erf
from scipy.stats import qmc





def csv_to_arrays(file_name):
    program_folder = Path(__file__).resolve().parent

    parameter_list = []
    left_list = []
    right_list = []

    csv_file = program_folder / f'{file_name}.csv'

    with open(csv_file, newline='') as csvfile:
        data = list(csv.reader(csvfile, delimiter=','))
        for i in range(len(data)):
            if i <= 1:
                parameter_list.append(data[i])
            elif data[i][0]== 'left':
                left_list.append([float(v) for v in data[i][1:]])
            else:
                right_list.append([float(v) for v in data[i][1:]])

    left_array = np.asarray(left_list, dtype=np.float32)
    right_array = np.asarray(right_list, dtype=np.float32)

    match = re.search(r'grid dimension:(\d+)', parameter_list[0][2])
    N = int(match.group(1))

    ZL = csv_to_arrays_helper(left_array,N)
    ZR = csv_to_arrays_helper(right_array,N)

    X = np.arange(N) * 0.25
    Y = np.arange(N) * 0.25

    return X,Y,ZL,ZR

def csv_to_arrays_helper(used_array, number_of_points):
    Z = used_array[:, 2].reshape(number_of_points,number_of_points)
    Z = 10**(Z / 10) # dBm -> mW

    return Z

# ---------------------------------------------------------------------------
# 1D peak estimators (from fit_gaussian / estimate_1d_peak)
# ---------------------------------------------------------------------------

def gaussian_1d(x, a, mu, sigma, offset):
    """ 
        Gaussian function with given parameters
    
        returns Gaussian function value
    """
    return a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + offset

def fit_gaussian_1d(x_data, y_data):
    """ 
        fits a gaussian to measured x_data and y_data, measures a single dimension
    
        returns the ideal parameters for the Gaussian function and the covariance
    """
    x_data, y_data = np.asarray(x_data, float), np.asarray(y_data, float)
    a_init = y_data.max() - y_data.min()
    mu_init = x_data[np.argmax(y_data)]
    sigma_init = x_data.max() - x_data.min()
    offset_init = y_data.min()
    p0 = [a_init, mu_init, sigma_init, offset_init]
    bounds = ((0, -np.inf, 0, -np.inf), (np.inf, np.inf, np.inf, np.inf))
    popt, cov = curve_fit(gaussian_1d, x_data, y_data, p0=p0, bounds=bounds,
                           ftol=1e-8, maxfev=10000)
    return popt, np.sqrt(np.diag(cov))

def measured_maximum_peak(x_data, y_data):
    """
        finds the maximum value of power in a 1d list

        returns x value of maximum and the maximum, also returns the index of the measured maximum    
    """
    x_data, y_data = np.asarray(x_data, float), np.asarray(y_data, float)
    idx = int(np.argmax(y_data))
    return float(x_data[idx]), float(y_data[idx]), idx

def quadratic_peak_near_measured_maximum(x_data, y_data):
    """
        takes the x and power values then fits a quadratic polynomial to those points

        returns the x-value of the maximum of the interpolated function, as well as the value of the maximum
    """
    x_data, y_data = np.asarray(x_data, float), np.asarray(y_data, float)
    x_meas, y_meas, idx = measured_maximum_peak(x_data, y_data)

    if idx <= 0 or idx >= len(x_data) - 1:
        return x_meas, y_meas, "edge: using measured max"

    x3, y3 = x_data[idx-1:idx+2], y_data[idx-1:idx+2]
    try:
        a, b, c = np.polyfit(x3, y3, deg=2)
    except Exception:
        return x_meas, y_meas, "polyfit failed: using measured max"

    if not np.isfinite(a) or a >= 0:
        return x_meas, y_meas, "not concave: using measured max"

    x_vertex = -b / (2 * a)
    if x_vertex < x3.min() or x_vertex > x3.max():
        return x_meas, y_meas, "vertex outside neighborhood: using measured max"

    return float(x_vertex), float(a*x_vertex**2 + b*x_vertex + c), "quadratic interpolation"

def estimate_1d_peak(x_data, y_data, method="quadratic"):
    """
        takes the measured data and which algorithm is chosen
    
        returns target_x the estimated_power and what the algorithm used was.
    """

    x_data, y_data = np.asarray(x_data, float), np.asarray(y_data, float)
    method = method.lower()

    if "gauss" in method:
        popt, _ = fit_gaussian_1d(x_data, y_data)
        target = float(popt[1])
        return target, float(gaussian_1d(target, *popt)), "Gaussian fit"

    if "measured" in method and "quadratic" not in method:
        target, power, _ = measured_maximum_peak(x_data, y_data)
        return target, power, "measured max"

    target, power, msg = quadratic_peak_near_measured_maximum(x_data, y_data)
    return target, power, msg

# ---------------------------------------------------------------------------
# Baseline 1: per-dimension coordinate search (the original search_for_peak)
# ---------------------------------------------------------------------------

def coordinate_search(measure_fn, start_coordinates, radius, step_size, method="quadratic"):
    """
        takes starting coordinates and iterates along one axis, then along the next axis
        for each axis it saves the values and then picks the maximum according to the different peak estimators
  
        returns the x,y values of the maximum, this does not yet do the process for both stages
    """
    current = list(map(float, start_coordinates))
    n_samples = 0
    history = []

    for dimidx, p_start in enumerate(start_coordinates):
        d_range = np.arange(-radius, radius + step_size, step_size)
        IL_meas = np.empty(len(d_range))

        for i, d in enumerate(d_range):
            trial = current.copy()
            trial[dimidx] = d + p_start
            IL_meas[i] = measure_fn(trial)
            n_samples += 1

        if not np.all(np.isfinite(IL_meas)):
            continue  # same "give up on this dim" behavior as original

        d_best, est_power, msg = estimate_1d_peak(d_range, IL_meas, method=method)

        if abs(d_best) > 1.5 * radius:
            d_best = 0.0  # fall back to start, same safety check as original

        current[dimidx] = d_best + p_start
        history.append({"dim": dimidx, "d_range": d_range, "IL": IL_meas, "msg": msg})

    final_power = measure_fn(current)
    n_samples += 1
    return current, final_power, {"n_samples": n_samples, "history": history}

# ---------------------------------------------------------------------------
# Baseline 2: N-dimensional Bayesian optimization
# ---------------------------------------------------------------------------

def _normalise(X, bounds):
    """
        maps absolute coordinates into a interval the size of bounds

        returns the normalised coordinates, same shape as X
    """
    X, bounds = np.asarray(X, float), np.asarray(bounds, float)
    lo, hi = bounds[:, 0], bounds[:, 1]
    return (X - lo) / np.maximum(hi - lo, 1e-12) 

def _denormalise(Xn, bounds):
    """
        maps a interval the size of bounds into absolute coordinates
        
        returns the absolute coordinates, same shape as X
    """
    Xn, bounds = np.asarray(Xn, float), np.asarray(bounds, float)
    lo, hi = bounds[:, 0], bounds[:, 1]
    return lo + Xn * (hi - lo)

def _rbf_kernel(X1, X2, length_scale, signal_sigma=1.0):
    """
        Computes the squared-exponential (RBF) kernel between every point in
        X1 and every point in X2: a similarity score that decays smoothly
        with distance, scaled by length_scale per dimension. This is the
        function that encodes "how correlated do I expect two measurements
        to be, given how far apart their coordinates are" — it's the core
        assumption the whole Gaussian Process is built on.

        returns a (len(X1), len(X2)) matrix of kernel values
    """
    X1, X2 = np.atleast_2d(X1), np.atleast_2d(X2)
    ls = np.asarray(length_scale, float)
    if ls.size == 1:
        ls = np.full(X1.shape[1], float(ls))
    ls = np.maximum(ls, 1e-6)
    d = (X1[:, None, :] - X2[None, :, :]) / ls[None, None, :]
    return signal_sigma**2 * np.exp(-0.5 * np.sum(d*d, axis=2))

def _gp_predict(Xn_train, y_train, Xn_pred, length_scale, noise_sigma=0.05, signal_sigma=1.0):
    """
        Given every point measured so far (Xn_train, y_train), predicts the
        Gaussian Process posterior at a set of new, unmeasured candidate
        points (Xn_pred): both a mean estimate (mu, "what power do I expect
        here") and a standard deviation (sigma, "how unsure am I about that
        estimate"). Solved via Cholesky decomposition for numerical
        stability, with an escalating jitter fallback if the kernel matrix
        becomes near-singular (very common when two sampled points are
        nearly identical).

        returns (mu, sigma), both arrays of length len(Xn_pred)
    """
    y_mean, y_std = float(np.mean(y_train)), float(np.std(y_train)) or 1.0
    y_scaled = (y_train - y_mean) / y_std

    K = _rbf_kernel(Xn_train, Xn_train, length_scale, signal_sigma)
    K += (noise_sigma**2 + 1e-9) * np.eye(len(Xn_train))
    Ks = _rbf_kernel(Xn_pred, Xn_train, length_scale, signal_sigma)
    Kss_diag = np.full(len(Xn_pred), signal_sigma**2)

    jitter = 1e-10
    for _ in range(6):
        try:
            c, low = cho_factor(K + jitter*np.eye(len(K)), lower=True, check_finite=False)
            alpha = cho_solve((c, low), y_scaled, check_finite=False)
            v = cho_solve((c, low), Ks.T, check_finite=False)
            mu_scaled = Ks @ alpha
            var_scaled = Kss_diag - np.sum(Ks * v.T, axis=1)
            break
        except Exception:
            jitter *= 10
    else:
        invK = np.linalg.pinv(K)
        mu_scaled = Ks @ invK @ y_scaled
        var_scaled = Kss_diag - np.sum((Ks @ invK) * Ks, axis=1)

    sigma_scaled = np.sqrt(np.maximum(var_scaled, 1e-12))
    return y_mean + y_std*mu_scaled, y_std*sigma_scaled

def _expected_improvement(mu, sigma, y_best, xi=0.01):
    """
        The acquisition function: scores every candidate point by how much
        better than the current best (y_best) it's expected to be, blending
        "the model thinks this is high" (mu) with "the model is unsure here,
        so it might be a hidden peak" (sigma). xi is a small margin that
        discourages picking a point unless it's expected to be a genuine,
        meaningful improvement, not just noise. This is the mechanism that
        gives Bayesian optimization its robustness against local maxima: a
        point can score highly purely because sigma is large there, even if
        mu alone looks unremarkable.

        returns one EI score per candidate point
    """
    sigma = np.maximum(sigma, 1e-12)
    improvement = mu - y_best - xi
    z = improvement / sigma
    cdf = 0.5 * (1 + erf(z/np.sqrt(2)))
    pdf = np.exp(-0.5*z*z) / np.sqrt(2*np.pi)
    return improvement*cdf + sigma*pdf

def _space_filling_points(n_points, n_dim, rng):
    """
        Generates n_points candidate positions in the unit box, spread out
        to cover the space evenly rather than clustering (via Sobol
        sequences, a deterministic low-discrepancy sampling method), falling
        back to plain uniform random points if scipy's Sobol sampler isn't
        available. Used both for the initial exploration samples and for
        building the candidate cloud each iteration.

        returns an (n_points, n_dim) array in [0,1]^n_dim
    """
    n_points = max(int(n_points), 1)
    if qmc is not None:
        try:
            m = int(np.ceil(np.log2(n_points)))
            sampler = qmc.Sobol(d=n_dim, scramble=True, seed=int(rng.integers(0, 2**31-1)))
            return sampler.random_base2(m=m)[:n_points]
        except Exception:
            pass
    return rng.random((n_points, n_dim))

def _candidate_cloud(n_candidates, n_dim, rng):
    """
        Builds the full set of "next point" candidates to score with
        Expected Improvement each iteration: mostly space-filling points,
        plus the exact center and a few points along each axis near the
        center. The extra center/axis points bias the search toward not
        ignoring the immediate local neighborhood just because the random
        space-filling points happened to miss it.

        returns an array of candidate points in [0,1]^n_dim
    """
    X = _space_filling_points(n_candidates, n_dim, rng)
    pts = []
    for dim in range(n_dim):
        for val in (0.25, 0.75):
            p = np.full(n_dim, 0.5); p[dim] = val
            pts.append(p)
    center = np.full((1, n_dim), 0.5)
    X = np.vstack([X, center, np.asarray(pts)]) if pts else np.vstack([X, center])
    return np.clip(X, 0.0, 1.0)

def bayesian_search(measure_fn, start_coordinates, search_radius,
                     n_initial=12, n_iter=30, n_candidates=4000,
                     length_scale=0.35, noise_sigma=0.05, xi=0.01, seed=None):
    """
        Runs the full Bayesian optimization loop: measure the start point,
        take a handful of space-filling initial samples, then repeatedly
        fit a GP to everything measured so far, score a cloud of candidate
        points with Expected Improvement, measure the best-scoring one, and
        repeat. Ends at whichever point actually measured the highest power
        (not the GP's predicted optimum, which could be wrong), so the
        result is never worse than what was genuinely observed.

        measure_fn(coords: list[float]) -> float power

        returns (best_coordinates, best_power, info_dict)
    """
    start = np.asarray(start_coordinates, float)
    n_dim = len(start)
    bounds = np.column_stack([start - search_radius, start + search_radius])
    rng = np.random.default_rng(seed)

    X_abs, y = [start.copy()], [measure_fn(start)]

    Xn_init = _space_filling_points(max(n_initial - 1, 0), n_dim, rng)
    for x in _denormalise(Xn_init, bounds):
        X_abs.append(np.asarray(x)); y.append(measure_fn(x))

    X_abs, y = np.asarray(X_abs), np.asarray(y)
    ls_vec = np.full(n_dim, float(length_scale))
    min_dist = 1e-3

    for _ in range(n_iter):
        Xn_train = _normalise(X_abs, bounds)
        Xn_cand = _candidate_cloud(n_candidates, n_dim, rng)

        mu, sigma = _gp_predict(Xn_train, y, Xn_cand, ls_vec, noise_sigma)
        ei = _expected_improvement(mu, sigma, y_best=float(np.max(y)), xi=xi)

        # don't re-propose a point essentially identical to one already measured
        dists = np.linalg.norm(Xn_cand[:, None, :] - Xn_train[None, :, :], axis=2)
        ei[np.min(dists, axis=1) < min_dist] = -np.inf
        if not np.any(np.isfinite(ei)):
            break

        x_next = _denormalise(Xn_cand[int(np.argmax(ei))][None, :], bounds)[0]
        X_abs = np.vstack([X_abs, x_next])
        y = np.append(y, measure_fn(x_next))

    best_idx = int(np.argmax(y))
    return list(X_abs[best_idx]), float(y[best_idx]), {
        "n_samples": len(y), "sampled_coordinates": X_abs.tolist(), "sampled_values": y.tolist()
    }

# ---------------------------------------------------------------------------
# Different Algorithms
# ---------------------------------------------------------------------------

# --------------------------------------------------------------------------- Grid Search Metric

def grid_search(Zfunction, start):
    maxval = -np.inf
    maxcoords = list(start)

    for i in np.arange(41)*0.25:
        for j in np.arange(41)*0.25:
            if Zfunction([i,j]) >= maxval:
                maxval = Zfunction([i,j])
                maxcoords = [i,j]

    samples = 41 * 41

    return maxcoords,float(maxval),{"n_samples": samples}

# --------------------------------------------------------------------------- Bisection Algorithm

def single_dim_bisection(Zfunction, stat_dim, static_pos, start, radius, max_iter, tolerance):
    middle = start
    p1 = middle-radius
    p2 = middle+radius
    n = 0
    quarter1 = 0
    quarter3 = 0

    if stat_dim == 0:
        while n <= max_iter and np.abs(Zfunction([static_pos, p1]) - Zfunction([static_pos, p2]) > tolerance):
            middle = (p2+p1)/2
            quarter1 = (middle+p1)/2
            quarter3 = (p2+middle)/2
            qv1 = Zfunction([static_pos, quarter1])
            qv3 = Zfunction([static_pos, quarter3])
            if qv1 > qv3:
                p2 = middle
            else:
                p1 = middle
            n += 1

        return (p1 + p2) / 2

    if stat_dim == 1:
        while n <= max_iter and np.abs(Zfunction([p1, static_pos]) - Zfunction([p2, static_pos]) > tolerance):
            middle = (p2+p1)/2
            quarter1 = (middle+p1)/2
            quarter3 = (p2+middle)/2
            qv1 = Zfunction([quarter1, static_pos])
            qv3 = Zfunction([quarter3, static_pos])
            if qv1 > qv3:
                p2 = middle
            else:
                p1 = middle
            n += 1

        return (p1 + p2) / 2

def bisection(Zfunction, start, radius):
    n = 100
    max_y = single_dim_bisection(Zfunction, 0, 5.0, start[1], radius, n, 0)
    max_x = single_dim_bisection(Zfunction, 1, 5.0, start[0], radius, n, 0)
    max_y = single_dim_bisection(Zfunction, 0, max_x, start[1], radius, n, 0)
    max_x = single_dim_bisection(Zfunction, 1, max_y, start[0], radius, n, 0)

    maxcoords = [max_x,max_y]
    maxval = Zfunction(maxcoords)

    return maxcoords,float(maxval),{"n_samples": 4*n}

# --------------------------------------------------------------------------- Golden Section Algorithm

def single_dim_golden_section(Zfunction, stat_dim, static_pos, start, radius, max_iter, tolerance):
    inv_phi = (np.sqrt(5) - 1) / 2  # ~0.618034
    
    p1 = start - radius
    p2 = start + radius
    
    # Helper to construct coordinates depending on static dimension
    def eval_z(val):
        return Zfunction([static_pos, val]) if stat_dim == 0 else Zfunction([val, static_pos])
    
    # Initial interior points
    q1 = p2 - inv_phi * (p2 - p1)
    q2 = p1 + inv_phi * (p2 - p1)
    
    fq1 = eval_z(q1)
    fq2 = eval_z(q2)
    
    for _ in range(max_iter):
        if np.abs(p2 - p1) < tolerance:
            break
            
        if fq1 > fq2:  # Searching for maximum
            p2 = q2
            q2 = q1
            fq2 = fq1
            q1 = p2 - inv_phi * (p2 - p1)
            fq1 = eval_z(q1)  # Only 1 new evaluation per loop
        else:
            p1 = q1
            q1 = q2
            fq1 = fq2
            q2 = p1 + inv_phi * (p2 - p1)
            fq2 = eval_z(q2)  # Only 1 new evaluation per loop

    return (p1 + p2) / 2

def golden_section(Zfunction, start, radius):
    n = 20
    max_y = single_dim_golden_section(Zfunction, 0, 5.0, start[1], radius, n, 0)
    max_x = single_dim_golden_section(Zfunction, 1, 5.0, start[0], radius, n, 0)
    #max_y = single_dim_golden_section(Zfunction, 0, max_x, start[1], radius, n, 0)
    #max_x = single_dim_golden_section(Zfunction, 1, max_y, start[0], radius, n, 0)

    maxcoords = [max_x,max_y]
    maxval = Zfunction(maxcoords)

    return maxcoords,float(maxval),{"n_samples": 2*(n+2)}

# --------------------------------------------------------------------------- Gradient Ascent Algorithm

def fixed_gradient_ascent(Zfunction, start, radius):
    x_points = np.arange(0,10)
    x_measured = []

    n = 0

    for x in x_points:
        x_measured.append([Zfunction([x,start[1]]),x,start[1]])
        n += 1

    x_measured = np.array(x_measured)
    best_row = x_measured[np.argmax(x_measured[:, 0])]
    x_max = best_row[1]

    y_points = np.arange(0,10)
    y_measured = []

    for y in y_points:
        y_measured.append([Zfunction([x_max,y]),x_max,y])
        n += 1

    y_measured = np.array(y_measured)
    best_row = y_measured[np.argmax(y_measured[:, 0])]
    y_max = best_row[2]

    candidate_coord = [x_max, y_max]
    candidate_measurement = Zfunction([x_max,y_max])

    delta = 0.75
    x,y = candidate_coord[0], candidate_coord[1]

    while delta >= 0.25:
        xdp, xdm, ydp, ydm = Zfunction([x+delta,y]), Zfunction([x-delta,y]), Zfunction([x,y+delta]), Zfunction([x,y-delta])


        candidates = [
            (xdp, [x+delta, y]),
            (xdm, [x-delta, y]),
            (ydp, [x, y+delta]),
            (ydm, [x, y-delta]),
        ]

        best_val, best_coord = max(candidates, key=lambda c: c[0])

        n += 4

        if best_val > candidate_measurement:
            candidate_coord = best_coord
            candidate_measurement = best_val

        else:
            delta = delta/2

    return candidate_coord,float(candidate_measurement),{"n_samples": n}

# --------------------------------------------------------------------------- Dual Sweep Algorithm        

def _sweep_all_dimensions(measure_fn, current, radius, step_size, method, bounds=None):
    """bounds: optional list of (lo, hi) tuples, one per dimension, to clip
    trial coordinates against. If None, no clipping is applied."""
    n_samples = 0
    history = []

    for dimidx, p_start in enumerate(current.copy()):
        d_range = np.arange(-radius, radius + step_size, step_size)
        IL_meas = np.empty(len(d_range))

        for i, d in enumerate(d_range):
            trial = current.copy()
            trial_val = d + p_start
            if bounds is not None:
                lo, hi = bounds[dimidx]
                trial_val = min(max(trial_val, lo), hi)
            trial[dimidx] = trial_val
            IL_meas[i] = measure_fn(trial)
            n_samples += 1

        if not np.all(np.isfinite(IL_meas)):
            continue

        d_best, est_power, msg = estimate_1d_peak(d_range, IL_meas, method=method)

        if abs(d_best) > 1.5 * radius:
            d_best = 0.0

        new_pos = d_best + p_start
        if bounds is not None:
            lo, hi = bounds[dimidx]
            new_pos = min(max(new_pos, lo), hi)
        current[dimidx] = new_pos

        history.append({"dim": dimidx, "d_range": d_range, "IL": IL_meas, "msg": msg})

    return n_samples, history

def coordinate_search_dual(measure_fn, start_coordinates, radius, step_size, method="quadratic", n_passes=2, bounds=None):
    current = list(map(float, start_coordinates))
    n_samples = 0
    all_history = []

    for pass_idx in range(n_passes):
        pass_samples, pass_history = _sweep_all_dimensions(measure_fn, current, radius, step_size, method, bounds=bounds)
        n_samples += pass_samples
        for entry in pass_history:
            entry["pass"] = pass_idx
        all_history.extend(pass_history)

    final_power = measure_fn(current)
    n_samples += 1
    return current, final_power, {"n_samples": n_samples, "history": all_history}

# ---------------------------------------------------------------------------
# Calling the algorithms
# ---------------------------------------------------------------------------

def run_all_methods(Zfunction, X, Y, Z, side_label, n_bo_seeds=5):
    """Run every method once (Bayesian n_bo_seeds times) against one surface.
    Returns a list of result dicts, one row per method/seed."""
    start = [5.0,5.0]
    radius = 4.0
    bounds = [(0.0, 10.0), (0.0, 10.0)]

    rows = []
    for method in ["measured", "gaussian", "quadratic"]:
        coords, power, info = coordinate_search(Zfunction, start, radius, step_size=0.5, method=method)
        rows.append({
            "side": side_label, "method": f"coordinate_{method}",
            "seed": None, "best_x": coords[0], "best_y": coords[1],
            "best_power": power, "n_samples": info["n_samples"],
        })

    for method in ["measured", "gaussian", "quadratic"]:
            coords, power, info = coordinate_search_dual(Zfunction, start, radius, step_size=0.5, method=method, bounds=bounds)
            rows.append({
                "side": side_label, "method": f"coordinate_{method}_dual",
                "seed": None, "best_x": coords[0], "best_y": coords[1],
                "best_power": power, "n_samples": info["n_samples"],
            })

    coords, power, info = bisection(Zfunction, start, radius)
    rows.append({
                "side": side_label, "method": "bisection",
                "seed": None, "best_x": coords[0], "best_y": coords[1],
                "best_power": power, "n_samples": info["n_samples"],
            })

    coords, power, info = golden_section(Zfunction, start, radius)
    rows.append({
                "side": side_label, "method": "golden_section",
                "seed": None, "best_x": coords[0], "best_y": coords[1],
                "best_power": power, "n_samples": info["n_samples"],
            })

    coords, power, info = fixed_gradient_ascent(Zfunction, start, radius)
    rows.append({
                "side": side_label, "method": "fixed_gradient_ascent",
                "seed": None, "best_x": coords[0], "best_y": coords[1],
                "best_power": power, "n_samples": info["n_samples"],
            })
    
    coords, power, info = grid_search(Zfunction, start)
    rows.append({
                "side": side_label, "method": "grid_search",
                "seed": None, "best_x": coords[0], "best_y": coords[1],
                "best_power": power, "n_samples": info["n_samples"],
            })

    for seed in range(n_bo_seeds):
        coords, power, info = bayesian_search(Zfunction, start, radius, n_iter=30, seed=seed)
        rows.append({
            "side": side_label, "method": "bayesian",
            "seed": seed, "best_x": coords[0], "best_y": coords[1],
            "best_power": power, "n_samples": info["n_samples"],
        })

    return rows

def find_matching_files(folder, id_strings):
    """Return the filename stems (no .csv extension) of every CSV in
    folder whose name contains id_string."""
    names = []
    for i in id_strings:
        names.extend([p.stem for p in folder.glob('*.csv') if i in p.stem])
    return names

def run_batch(folder, id_strings, output_csv):
    all_rows = []
    for file_stem in find_matching_files(folder, id_strings):  # reuse your earlier helper
        X, Y, ZL, ZR = csv_to_arrays(file_stem)
        Lfunction = RegularGridInterpolator((X, Y), ZL, method='pchip')
        Rfunction = RegularGridInterpolator((X, Y), ZR, method='pchip')

        for side_label, func, Z in [("left", Lfunction, ZL), ("right", Rfunction, ZR)]:
            measure_fn = lambda coords, f=func: float(f(coords))
            try:
                rows = run_all_methods(measure_fn, X, Y, Z, side_label)
                for row in rows:
                    row["file"] = file_stem
                all_rows.extend(rows)
            except Exception as e:
                print(f"Failed on {file_stem} ({side_label}): {e}")

    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["file", "side", "method", "seed",
                                                "best_x", "best_y", "best_power", "n_samples"])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Wrote {len(all_rows)} rows to {output_csv}")

if __name__ == "__main__":
    folder = Path(__file__).resolve().parent
    run_batch(folder, id_strings=["ID1","ID3","ID5","ID7"], output_csv=folder / "_3_algorithm_comparison_results.csv")


