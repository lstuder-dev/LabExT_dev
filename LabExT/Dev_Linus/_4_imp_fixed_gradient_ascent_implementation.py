import numpy as np




def fixed_gradient_ascent(measure_fn, start_coordinates, radius, number_of_initial_points, max_iterations, 
                          initial_delta, delta_tolerance, max_samples):
    
    start_coordinates = [float(v) for v in start_coordinates]
    rx, ry = (radius,radius) if np.ndim(radius) == 0 else radius
    x_bounds = (start_coordinates[0]-rx, start_coordinates[0]+rx)
    y_bounds = (start_coordinates[1]-ry, start_coordinates[1]+ry)
    X_abs, power_measurememts = [],[]


    def measure(x,y):
        value = measure_fn([x,y])
        X_abs.append([float(x),float(y)])
        power_measurememts.append(value)
        return value if np.isfinite(value) else -np.inf

    def best_so_far():
        vals = np.where(np.isfinite(power_measurememts), power_measurememts, -np.inf)
        k = int(np.argmax(vals))
        return list(X_abs[k]), float(vals[k])

    measure(*start_coordinates)

    x_points = np.linspace(x_bounds[0],x_bounds[1],number_of_initial_points)
    x_measured = [measure(x, start_coordinates[1]) for x in x_points]
    x_max = float(x_points[np.argmax(x_measured)])


    y_points = np.linspace(y_bounds[0],y_bounds[1],number_of_initial_points)
    for y in y_points:
        measure(x_max,y)

    candidate_coord, candidate_power = best_so_far()

    delta, iter = initial_delta, 0
    
    while delta >= delta_tolerance and iter < max_iterations and len(power_measurememts) < max_samples:
        x,y = candidate_coord

        candidates = [(x+delta, y), (x-delta, y), (x, y+delta), (x, y-delta)]
        candidates = [c for c in candidates if x_bounds[0] <= c[0] <= x_bounds[1]
                                  and y_bounds[0] <= c[1] <= y_bounds[1]]

        values = [measure(cx,cy) for cx,cy in candidates]

        if values and max(values) > candidate_power:
            candidate_power = max(values)
            candidate_coord = list(candidates[int(np.argmax(values))])
        else:
            delta /= 2

        iter += 1


    info = {
        "method": 'Fixed Gradient Ascent',
        "start coordinates": start_coordinates,
        "bounds": [list(x_bounds),list(y_bounds)],
        "n_samples": len(power_measurememts),
        "n_iterations": iter,
        "sampled coordinates": X_abs,
        "sampled powers dBm": power_measurememts,
    }

    return [float(v) for v in candidate_coord], float(candidate_power), info


def make_sub_measure(measure_all, base, dims):
    """Expose some dimensions of a larger search space as a small one."""
    base = list(base)
    def sub_measure(sub):
        full = base.copy()
        for d, v in zip(dims, sub):
            full[d] = float(v)
        return measure_all(full)
    return sub_measure


def _run_stagewise(self, algo, algo_kwargs, start_coordinates, radius_um,
                   pause_time_ms=200, n_averages=1, label="stagewise"):
    start = list(map(float, start_coordinates))
    current = start.copy()
    dims_per_stage = [(0, 1)] if len(start) == 2 else [(0, 1), (2, 3)]

    def measure_all(coords):
        return self._measure_power_at_coordinates(coords, pause_time_ms, n_averages)

    start_power = measure_all(start)
    n_samples = 1
    per_stage = []
    try:
        for dims in dims_per_stage:
            sub_measure = make_sub_measure(measure_all, current, dims)
            coords, _, info = algo(sub_measure, [current[d] for d in dims], radius_um, **algo_kwargs)
            for d, v in zip(dims, coords):
                current[d] = v
            n_samples += info["n_samples"]
            per_stage.append({"dims": list(dims), **info})

        final_power = measure_all(current)
        n_samples += 1
        if not np.isfinite(final_power) or final_power < start_power:
            current, final_power = start.copy(), start_power
            self._move_stages_absolute(current)
    except Exception:
        self.logger.exception("%s failed; moving back to start.", label)
        self._move_stages_absolute(start)
        raise

    info = {"method": label, "start power dBm": float(start_power),
            "n_samples": n_samples, "per stage": per_stage}
    return current, float(final_power), info