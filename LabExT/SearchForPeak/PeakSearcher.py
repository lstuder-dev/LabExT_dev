#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LabExT  Copyright (C) 2021  ETH Zurich and Polariton Technologies AG
This program is free software and comes with ABSOLUTELY NO WARRANTY; for details see LICENSE file.
"""

import json
import logging
import os
import time
from typing import Type

import numpy as np
from scipy.optimize import curve_fit
from scipy.linalg import cho_factor, cho_solve
from scipy.special import erf

#TODO: omuell added for testing
import matplotlib.pyplot as plt

try:
    # Sobol points give a better space-filling initial design/candidate cloud than pure random points.
    # The code below falls back to random sampling if this import is unavailable in the LabExT env.
    from scipy.stats import qmc
except Exception:  # pragma: no cover - depends on installed SciPy version
    qmc = None

from LabExT.Measurements.MeasAPI import *
from LabExT.Movement.MotorProfiles import trapezoidal_velocity_profile_by_integration
from LabExT.Movement.MoverNew import MoverNew
from LabExT.Movement.config import CoordinateSystem
from LabExT.Movement.Transformations import StageCoordinate
from LabExT.Utils import get_configuration_file_path
from LabExT.View.Controls.PlotControl import PlotData
from LabExT.ViewModel.Utilities.ObservableList import ObservableList


class PeakSearcher(Measurement):
    """
    ## Search for Peak

    Executes a Search for Peak for a standard IL measurement with one or two stages (left and right) and only x and y coordinates.
    This Measurement is NOT a 'normal' measurement and should NOT be used in an experiment routine.

    #### Details
    An optical signal generated at an optical source passes through the DUT and into a power meter. The optical fibers carrying said signal are mounted onto
    remotely controllable stages (in our case SmarAct Piezo Stages). In this routine, these stages mechanically sweep over a given range, the insertion loss is measured in regular intervals.
    The sweep is conducted in x and y direction separately.

    The Search for Peak measurement routine relies on the assumption that around the transmission maximum of a grating coupler, the transmission forms a 2D gaussian (w.r.t x and y position).
    Thus after having collected data for each axis, a 1D gaussian is fitted to the data and the stages are moved to the maximum of the gaussian.

    There are two types of Search for Peak available:
    - **stepped SfP**: suitable for all types of fibers/fiber arrays and all power meter models. The given range is mechanically stepped over, the measurement
    stops at each point given by the `search step size` parameter, waits the time given by the `search fiber stabilization time` parameter to let fiber vibrations
    dissipate and then records a data point. This type is universally applicable but also very slow.
    - **fast SfP**: suitable only for fiber arrays and the Keysight N7744a power meter models. The given range is mechanically continuously sweeped over, the power meter
    collects regular data points (amount is given by `Number of points`). Those data points are then related to a physical position taking into account the acceleration
    of the stages. This type of Search for Peak is significantly faster than the stepped SfP and provides the user with a massively increased amount of data.
    At the moment, this type only works with the Keysight N7744a power meter. Usage with single mode fibers is possible, but untested.


    #### Example Setup

    ```
    Laser -in-> DUT -out-> Power Meter
    ```
    The `-xx->` arrows denote where the remotely controllable stages are placed. In the case of a fiber array, `-in->` and `-out->` denote the same stage, as both input and output of the DUT are
    included in the fiber array. In the case of two single fibers, `-in->` and `-out->` denote two separate stages.

    ### Parameters

    #### Laser Parameters
    - **Laser wavelength**: wavelength of the laser in [nm].
    - **Laser power**: power of the laser in [dBm].

    #### Power Meter Parameters
    - **Power Meter range**: range of the power in [dBm].

    #### Stage Parameters
    - **Search radius**: Radius arond the current position the algorithm sweeps over in [um].
    - **SfP type**: Type of Search for Peak to use. Options are `stepped SfP` and `swept SfP`, see above for more detail.
    - **(stepped SfP only) Search step size**: Distance between every data point in [um].
    - **(stepped SfP only) Search fiber stabilization time**: Idle time between the stage having reached the target position and the measurement start. Meant to allow fiber oscillations to dissipate.
    - **(swept SfP only) Search time**: Time the mechanical movement across the set measurement range should take in [s].
    - **(swept SfP only) Number of points**: Number of points to collect at the power meter for each separate sweep.

    All parameters labelled `stepped SfP only` are ignored when choosing the swept SfP, all parameters labelled `swept SfP only` are ignored when choosing the stepped SfP.
    """

    DIMENSION_NAMES_TWO_STAGES = ['Left X', 'Left Y', 'Right X', 'Right Y']
    DIMENSION_NAMES_SINGLE_STAGE = ['X', 'Y']

    def __init__(
        self,
        *args,
        mover: Type[MoverNew] = None,
        parent=None,
        **kwargs
    ) -> None:
        """Constructor

        Parameters
        ----------
        mover : Mover
            Reference to the Mover class for Piezo stages.
        """
        super().__init__(*args, **kwargs)  # calling parent constructor

        self._parent = parent
        self.name = "SearchForPeak-2DGaussianFit"
        self.settings_filename = "PeakSearcher_settings.json"
        self.mover = mover

        self.logger = logging.getLogger()

        # gather all plots for the plotting GUIs
        self.plots_left = ObservableList()
        self.plots_right = ObservableList()

        # chosen instruments for IL measurement
        self.instr_laser = None
        self.instr_powermeter = None
        self.initialized = False

        self.logger.info(
            'Initialized Search for Peak with method: ' + str(self.name))

    @property
    def settings_path_full(self):
        return get_configuration_file_path(self.settings_filename)

    def set_experiment(self, experiment):
        """Helper function to keep all initializations in the right order
        This line cannot be included in __init__
        """
        self._experiment = experiment

    @staticmethod
    def _gaussian(xdata, a, mu, sigma, offset):
        return a * np.exp(-(xdata - mu) ** 2 / (2 * sigma ** 2)) + offset

    @staticmethod
    def _gaussian_param_initial_guess(x_data, y_data):
        """
        Crudely estimates initial parameters for a gaussian fitting on 2-dimensional data.
        """
        a_init = y_data.max() - y_data.min()
        # mu_init = np.sum(x_data * y_data) / np.sum(y_data)
        mu_init = x_data[np.argmax(y_data)]
        # sigma_init = np.sqrt(np.sum(y_data * (x_data - mu_init) ** 2 / np.sum(y_data)))
        # assume that sigma spans the sampled interval
        sigma_init = x_data.max() - x_data.min()
        offset_init = y_data.min()

        return [a_init, mu_init, sigma_init, offset_init]

    def fit_gaussian(self, x_data, y_data):
        """Fits a gaussian function of four parameters to the given x and y data.

        Parameters
        ----------
        x_data : np.ndarray
            the set of independent data points
        y_data : np.ndarray
            the set of dependent data points

        Returns
        -------
        popt: 4-tuple
            a (amplitude of gauss peak), mu (mean of gauss), sigma (std dev of gauss), offset (y-axis offset baseline)
        perr_std_dev: np.ndarray
            a 4-vector giving the estimated std deviations of the parameters, the lower the better

        Raises
        ------
        RuntimeError: when the fitting fails to converge.
        """

        # make sure the input data is in numpy arrays
        x_data = np.array(x_data)
        y_data = np.array(y_data)

        # we cannot fit on empty vectors
        assert len(x_data) > 0
        assert len(y_data) > 0

        pinit = PeakSearcher._gaussian_param_initial_guess(x_data, y_data)

        # define bounds for the fitting parameters
        a_bounds = (0, np.inf)  # allow only positive gaussians, i.e. hills, not valleys
        mu_bounds = (-np.inf, np.inf)
        sigma_bounds = (0, np.inf)
        offset_bounds = (-np.inf, np.inf)

        lower_bounds = (a_bounds[0], mu_bounds[0], sigma_bounds[0], offset_bounds[0])
        upper_bounds = (a_bounds[1], mu_bounds[1], sigma_bounds[1], offset_bounds[1])

        # fit a gaussian to the data
        popt, cov = curve_fit(PeakSearcher._gaussian,
                              x_data,
                              y_data,
                              p0=pinit,
                              bounds=(lower_bounds, upper_bounds),
                              ftol=1e-8,
                              maxfev=10000)

        self.logger.debug('Gaussian Fit:')
        self.logger.debug('a -- mu -- sigma -- offset')
        self.logger.debug(str(popt))

        perr_std_dev = np.sqrt(np.diag(cov))

        return popt, perr_std_dev

    # -------------------------------------------------------------------------
    # Safer 1D peak estimation helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _measured_maximum_peak(x_data, y_data):
        """Return the actually measured maximum point.

        This is the most conservative replacement for the old Gaussian fit: it
        makes no assumption about the optical mode shape. It is therefore robust
        for asymmetric or multi-lobed grating-coupler profiles, but it is limited
        by the scan step size and can be sensitive to a single noisy data point.
        """
        x_data = np.asarray(x_data, dtype=float)
        y_data = np.asarray(y_data, dtype=float)
        idx = int(np.argmax(y_data))
        return float(x_data[idx]), float(y_data[idx]), idx

    @staticmethod
    def _quadratic_peak_near_measured_maximum(x_data, y_data):
        """Estimate a sub-step peak position from the three points around max.

        This does *not* assume a Gaussian profile. It only assumes that the very
        local neighborhood of a smooth maximum can be approximated by a parabola.
        If the maximum is at the edge, or if the local parabola opens upwards,
        the function falls back to the actually measured maximum.
        """
        x_data = np.asarray(x_data, dtype=float)
        y_data = np.asarray(y_data, dtype=float)
        x_meas, y_meas, idx = PeakSearcher._measured_maximum_peak(x_data, y_data)

        if idx <= 0 or idx >= len(x_data) - 1:
            return x_meas, y_meas, None, "maximum is at scan edge; using measured maximum"

        x3 = x_data[idx - 1:idx + 2]
        y3 = y_data[idx - 1:idx + 2]

        try:
            # y = ax^2 + bx + c.  For a true local maximum, a must be negative.
            a, b, c = np.polyfit(x3, y3, deg=2)
        except Exception:
            return x_meas, y_meas, None, "local quadratic fit failed; using measured maximum"

        if not np.isfinite(a) or not np.isfinite(b) or a >= 0:
            return x_meas, y_meas, None, "local parabola is not a maximum; using measured maximum"

        x_vertex = -b / (2 * a)

        # Only trust the interpolation inside the directly sampled neighborhood.
        if x_vertex < min(x3) or x_vertex > max(x3):
            return x_meas, y_meas, None, "quadratic vertex outside local neighborhood; using measured maximum"

        y_vertex = float(a * x_vertex ** 2 + b * x_vertex + c)
        poly_coeff = np.array([a, b, c], dtype=float)
        return float(x_vertex), y_vertex, poly_coeff, "local quadratic interpolation around measured maximum"

    def estimate_1d_peak(self, x_data, y_data, method="quadratic around measured max"):
        """Estimate a 1D peak while keeping the old return conventions.

        The old Search-for-Peak code expected a Gaussian-like parameter vector
        ``[amplitude, mu, sigma, offset]`` because it moved to ``popt[1]``.
        This helper keeps that convention even for non-Gaussian peak choices:
        ``mu`` is always the chosen target position, while the other values are
        harmless descriptive placeholders. This means the surrounding code and
        the result dictionary can stay almost unchanged.
        """
        x_data = np.asarray(x_data, dtype=float)
        y_data = np.asarray(y_data, dtype=float)
        method = str(method or "quadratic around measured max")

        if len(x_data) == 0 or len(y_data) == 0:
            raise RuntimeError("Cannot estimate a peak from empty data.")
        if len(x_data) != len(y_data):
            raise RuntimeError("x_data and y_data must have the same length.")
        if not np.all(np.isfinite(y_data)) or not np.all(np.isfinite(x_data)):
            raise RuntimeError("Cannot estimate a peak from non-finite data.")

        method_l = method.lower()

        # Option 1: keep the original behavior for comparison.
        if "gauss" in method_l:
            popt, perr_std_dev = self.fit_gaussian(x_data, y_data)
            target = float(popt[1])
            estimated_power = float(PeakSearcher._gaussian(target, *popt))
            fit_x = np.linspace(x_data.min(), x_data.max(), num=max(len(x_data) * 5, 50))
            fit_y = PeakSearcher._gaussian(fit_x, *popt)
            return list(map(float, popt)), perr_std_dev, target, estimated_power, fit_x, fit_y, "Gaussian fit"

        # Option 2: pure measured maximum, no model at all.
        if "measured" in method_l and "quadratic" not in method_l:
            target, estimated_power, _ = PeakSearcher._measured_maximum_peak(x_data, y_data)
            fit_x, fit_y = None, None
            fit_msg = "Using measured maximum; no Gaussian shape assumption."

        # Option 3: recommended default: measured maximum plus local parabolic
        # interpolation for sub-step resolution. Falls back to measured max.
        else:
            target, estimated_power, poly_coeff, reason = PeakSearcher._quadratic_peak_near_measured_maximum(x_data, y_data)
            fit_msg = "Using " + reason + "; no Gaussian shape assumption."
            if poly_coeff is not None:
                # Plot only the local quadratic, not a fake full-profile model.
                idx = int(np.argmax(y_data))
                x_local = x_data[idx - 1:idx + 2]
                fit_x = np.linspace(x_local.min(), x_local.max(), num=50)
                fit_y = np.polyval(poly_coeff, fit_x)
            else:
                fit_x, fit_y = None, None

        # Gaussian-compatible placeholders for the existing result dictionary.
        amplitude = float(np.nanmax(y_data) - np.nanmin(y_data))
        sigma_like = float(max(np.nanmax(x_data) - np.nanmin(x_data), 1e-12))
        offset = float(np.nanmin(y_data))
        popt = [amplitude, float(target), sigma_like, offset]
        perr_std_dev = None
        return popt, perr_std_dev, float(target), float(estimated_power), fit_x, fit_y, fit_msg

    # -------------------------------------------------------------------------
    # Bayesian optimization helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _normalise_to_unit_box(X, bounds):
        """Map absolute coordinates to [0, 1]^N for numerical stability."""
        X = np.asarray(X, dtype=float)
        bounds = np.asarray(bounds, dtype=float)
        lo = bounds[:, 0]
        hi = bounds[:, 1]
        span = np.maximum(hi - lo, 1e-12)
        return (X - lo) / span

    @staticmethod
    def _denormalise_from_unit_box(Xn, bounds):
        """Map [0, 1]^N coordinates back to absolute stage coordinates."""
        Xn = np.asarray(Xn, dtype=float)
        bounds = np.asarray(bounds, dtype=float)
        lo = bounds[:, 0]
        hi = bounds[:, 1]
        return lo + Xn * (hi - lo)

    @staticmethod
    def _squared_exponential_kernel(X1, X2, length_scale, signal_sigma=1.0, cutoff=0.0):
        """Squared-exponential/RBF kernel used by the small in-file GP.

        This is the vectorised version of the kernel in the postdoc's
        ``BayesianEstimation_Ndimensions.py``. That script minimises a test
        parabola on a discrete grid; here we maximise measured optical power and
        generate candidate points in a bounded local search box.
        """
        X1 = np.atleast_2d(np.asarray(X1, dtype=float))
        X2 = np.atleast_2d(np.asarray(X2, dtype=float))
        length_scale = np.asarray(length_scale, dtype=float)
        if length_scale.size == 1:
            length_scale = np.full(X1.shape[1], float(length_scale))
        length_scale = np.maximum(length_scale, 1e-6)

        d = (X1[:, None, :] - X2[None, :, :]) / length_scale[None, None, :]
        K = float(signal_sigma) ** 2 * np.exp(-0.5 * np.sum(d * d, axis=2))
        if cutoff and cutoff > 0:
            K[K < cutoff] = 0.0
        return K

    @staticmethod
    def _gp_predict_unit_box(Xn_train, y_train, Xn_pred, length_scale, noise_sigma=0.05,
                             signal_sigma=1.0, kernel_cutoff=0.0):
        """Gaussian-process prediction in the unit box.

        The y-values are internally centered/scaled, because dBm values may be
        around -60...0 dBm while the GP formulas are numerically happier around
        order unity.
        """
        Xn_train = np.asarray(Xn_train, dtype=float)
        y_train = np.asarray(y_train, dtype=float)
        Xn_pred = np.asarray(Xn_pred, dtype=float)

        y_mean = float(np.mean(y_train))
        y_std = float(np.std(y_train))
        if y_std < 1e-9:
            y_std = 1.0
        y_scaled = (y_train - y_mean) / y_std

        K = PeakSearcher._squared_exponential_kernel(
            Xn_train, Xn_train, length_scale=length_scale,
            signal_sigma=signal_sigma, cutoff=kernel_cutoff)
        K = K + (float(noise_sigma) ** 2 + 1e-9) * np.eye(len(Xn_train))
        Ks = PeakSearcher._squared_exponential_kernel(
            Xn_pred, Xn_train, length_scale=length_scale,
            signal_sigma=signal_sigma, cutoff=kernel_cutoff)
        Kss_diag = np.full(len(Xn_pred), float(signal_sigma) ** 2)

        # Cholesky solve is more stable than explicitly inverting K.
        jitter = 1e-10
        for _ in range(6):
            try:
                c, low = cho_factor(K + jitter * np.eye(len(K)), lower=True, check_finite=False)
                alpha = cho_solve((c, low), y_scaled, check_finite=False)
                v = cho_solve((c, low), Ks.T, check_finite=False)
                mu_scaled = Ks @ alpha
                var_scaled = Kss_diag - np.sum(Ks * v.T, axis=1)
                break
            except Exception:
                jitter *= 10
        else:
            # Last-resort fallback. Slower/less stable, but prevents a lab run
            # from crashing if K becomes ill-conditioned.
            invK = np.linalg.pinv(K)
            mu_scaled = Ks @ invK @ y_scaled
            var_scaled = Kss_diag - np.sum((Ks @ invK) * Ks, axis=1)

        sigma_scaled = np.sqrt(np.maximum(var_scaled, 1e-12))
        mu = y_mean + y_std * mu_scaled
        sigma = y_std * sigma_scaled
        return mu, sigma

    @staticmethod
    def _expected_improvement_for_maximisation(mu, sigma, y_best, xi=0.01):
        """Expected-improvement acquisition function for maximising power."""
        mu = np.asarray(mu, dtype=float)
        sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-12)
        improvement = mu - float(y_best) - float(xi)
        z = improvement / sigma
        normal_cdf = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
        normal_pdf = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
        return improvement * normal_cdf + sigma * normal_pdf

    @staticmethod
    def _space_filling_points(n_points, n_dim, rng):
        """Generate points in [0, 1]^N for initial BO samples/candidates."""
        n_points = int(max(n_points, 1))
        n_dim = int(n_dim)
        if qmc is not None:
            try:
                # Sobol requires powers of two, then we trim.
                m = int(np.ceil(np.log2(n_points)))
                sampler = qmc.Sobol(d=n_dim, scramble=True, seed=int(rng.integers(0, 2 ** 31 - 1)))
                return sampler.random_base2(m=m)[:n_points]
            except Exception:
                pass
        return rng.random((n_points, n_dim))

    @staticmethod
    def _candidate_cloud(n_candidates, n_dim, rng):
        """Candidate set used to maximise EI without a nested optimizer."""
        X = PeakSearcher._space_filling_points(n_candidates, n_dim, rng)

        # Add center and axis-biased points. This makes the BO less likely to
        # ignore the obvious local region around the current alignment.
        center = np.full((1, n_dim), 0.5)
        axis_points = []
        for dim in range(n_dim):
            for val in (0.25, 0.75):
                p = np.full(n_dim, 0.5)
                p[dim] = val
                axis_points.append(p)
        if axis_points:
            X = np.vstack([X, center, np.asarray(axis_points)])
        else:
            X = np.vstack([X, center])
        return np.clip(X, 0.0, 1.0)

    def _param_value(self, name, default):
        """Read a LabExT parameter safely, keeping compatibility with old settings."""
        # Wrapper methods below can force a mode without requiring a GUI setting.
        if name == 'Peak search optimizer':
            forced = getattr(self, '_forced_peak_optimizer', None)
            if forced is not None:
                return forced
        try:
            if name in self.parameters and self.parameters[name] is not None:
                value = self.parameters[name].value
                if value is not None:
                    return value
        except Exception:
            pass
        return default

    def _measure_power_at_coordinates(self, coordinates, pause_time_ms=200, n_averages=1):
        """Move to absolute stage coordinates and return median dBm power.

        The power meter unit is set to dBm in ``search_for_peak``. Maximising dBm
        is equivalent to maximising linear power, but usually gives a smoother
        numerical target for the GP.
        """
        coordinates = list(map(float, coordinates))
        self._move_stages_absolute(coordinates)
        if pause_time_ms and pause_time_ms > 0:
            time.sleep(float(pause_time_ms) / 1000.0)

        values = []
        for _ in range(int(max(n_averages, 1))):
            val = float(self.instr_powermeter.power)
            if np.isfinite(val):
                values.append(val)
        if not values:
            return -300.0
        return float(np.median(values))

    def _run_bayesian_optimisation(self, start_coordinates, search_radius_um,
                                   pause_time_ms=200, n_initial=12, n_iter=30,
                                   n_averages=1, n_candidates=4000,
                                   length_scale=0.35, noise_sigma=0.05,
                                   xi=0.01, seed=None, label="Bayesian"):
        """Run local N-dimensional Bayesian optimisation around a start point.

        For a two-stage setup N=4: [left x, left y, right x, right y]. For a
        single-stage setup N=2. The function deliberately works in absolute
        stage coordinates but normalises internally to [0, 1]^N.
        """
        start_coordinates = np.asarray(start_coordinates, dtype=float)
        n_dim = len(start_coordinates)
        search_radius_um = float(search_radius_um)
        bounds = np.column_stack([
            start_coordinates - search_radius_um,
            start_coordinates + search_radius_um,
        ])

        rng = np.random.default_rng(None if seed in (None, "") else int(seed))

        X_abs = []
        y = []

        # Always measure the current/start point first. This protects against
        # BO making things worse: the best sampled point is never worse than the
        # current alignment, except for measurement drift/noise.
        y0 = self._measure_power_at_coordinates(start_coordinates, pause_time_ms, n_averages)
        X_abs.append(start_coordinates.copy())
        y.append(y0)

        # Space-filling initial design inside the local box. We skip points that
        # are numerically identical to the center already measured.
        Xn_init = PeakSearcher._space_filling_points(max(n_initial - 1, 0), n_dim, rng)
        X_init = PeakSearcher._denormalise_from_unit_box(Xn_init, bounds)
        for x in X_init:
            p = self._measure_power_at_coordinates(x, pause_time_ms, n_averages)
            X_abs.append(np.asarray(x, dtype=float))
            y.append(p)
            self.logger.debug(f"{label} initial sample: power={p:.3f} dBm at {list(map(float, x))}")

        X_abs = np.asarray(X_abs, dtype=float)
        y = np.asarray(y, dtype=float)

        length_scale_vec = np.full(n_dim, float(length_scale))
        min_dist = 1e-3  # unit-box distance; avoids proposing identical points

        for iteration in range(int(max(n_iter, 0))):
            Xn_train = PeakSearcher._normalise_to_unit_box(X_abs, bounds)
            Xn_cand = PeakSearcher._candidate_cloud(n_candidates, n_dim, rng)

            mu, sigma = PeakSearcher._gp_predict_unit_box(
                Xn_train, y, Xn_cand,
                length_scale=length_scale_vec,
                noise_sigma=noise_sigma,
                signal_sigma=1.0,
                kernel_cutoff=0.0,
            )
            ei = PeakSearcher._expected_improvement_for_maximisation(
                mu, sigma, y_best=float(np.max(y)), xi=xi)

            # Avoid sampling almost the same position repeatedly.
            distances = np.linalg.norm(Xn_cand[:, None, :] - Xn_train[None, :, :], axis=2)
            ei[np.min(distances, axis=1) < min_dist] = -np.inf

            if not np.any(np.isfinite(ei)):
                self.logger.warning(f"{label}: acquisition function had no finite candidate; stopping early.")
                break

            x_next_n = Xn_cand[int(np.argmax(ei))]
            x_next = PeakSearcher._denormalise_from_unit_box(x_next_n[None, :], bounds)[0]
            p_next = self._measure_power_at_coordinates(x_next, pause_time_ms, n_averages)

            X_abs = np.vstack([X_abs, x_next])
            y = np.append(y, p_next)

            best_idx = int(np.argmax(y))
            self.logger.debug(
                f"{label} iteration {iteration + 1}/{n_iter}: "
                f"measured={p_next:.3f} dBm, best={y[best_idx]:.3f} dBm "
                f"at {list(map(float, X_abs[best_idx]))}")

        best_idx = int(np.argmax(y))
        best_coordinates = list(map(float, X_abs[best_idx]))
        best_power = float(y[best_idx])

        # End at the best actually measured coordinate, not at the GP-predicted
        # maximum. This is safer for lab automation and makes comparisons fair.
        self._move_stages_absolute(best_coordinates)

        info = {
            "method": label,
            "start coordinates": list(map(float, start_coordinates)),
            "bounds": bounds.tolist(),
            "best measured coordinates": best_coordinates,
            "best measured power dBm": best_power,
            "sampled coordinates": X_abs.tolist(),
            "sampled powers dBm": y.tolist(),
            "n initial": int(n_initial),
            "n iterations": int(n_iter),
            "n averages": int(n_averages),
            "candidate count per iteration": int(n_candidates),
            "kernel": "squared exponential / RBF",
            "kernel length scale in unit box": list(map(float, length_scale_vec)),
            "noise sigma dB": float(noise_sigma),
            "expected improvement xi dB": float(xi),
        }
        return best_coordinates, best_power, info


    def _add_bayesian_projection_plots(self, bayes_info, label_prefix="Bayesian"):
        """Populate the existing left/right plot containers for Bayesian runs.

        The original coordinate-search routine fills ``plots_left`` and
        ``plots_right`` with 1D scans: coordinate offset on x, measured dBm on y.
        A true 4D Bayesian optimisation does not produce such independent scans,
        because every sample changes several coordinates jointly. To keep the GUI
        useful and keep the same two plot containers, we plot diagnostic 1D
        *projections* instead:

        - scatter points: all BO samples projected onto one coordinate axis;
        - line: GP posterior mean along that coordinate while all other
          coordinates are fixed at the best measured position;
        - x marker: the best actually measured position.

        This means the plot should be read as a diagnostic slice/projection, not
        as a literal axis-by-axis sweep like the old SfP plots.
        """
        try:
            X_abs = np.asarray(bayes_info.get("sampled coordinates", []), dtype=float)
            y = np.asarray(bayes_info.get("sampled powers dBm", []), dtype=float)
            bounds = np.asarray(bayes_info.get("bounds", []), dtype=float)
            center = np.asarray(bayes_info.get("start coordinates", []), dtype=float)
            best = np.asarray(bayes_info.get("best measured coordinates", []), dtype=float)
        except Exception as exc:
            self.logger.warning(f"Could not create Bayesian diagnostic plots: {exc}")
            return

        if X_abs.ndim != 2 or y.ndim != 1 or len(X_abs) == 0 or len(X_abs) != len(y):
            self.logger.warning("Could not create Bayesian diagnostic plots: invalid sample arrays.")
            return

        n_dim = X_abs.shape[1]
        if bounds.shape != (n_dim, 2):
            self.logger.warning("Could not create Bayesian diagnostic plots: invalid bounds.")
            return
        if center.size != n_dim:
            center = X_abs[0].copy()
        if best.size != n_dim:
            best = X_abs[int(np.argmax(y))].copy()

        color_strings = ['C' + str(i) for i in range(10)]
        best_power = float(np.max(y))

        # Read the BO hyperparameters back from the info dict so the diagnostic
        # slice uses the same GP assumptions as the actual optimiser.
        length_scale = bayes_info.get("kernel length scale in unit box", 0.35)
        noise_sigma = bayes_info.get("noise sigma dB", 0.05)

        try:
            Xn_train = PeakSearcher._normalise_to_unit_box(X_abs, bounds)
        except Exception as exc:
            self.logger.warning(f"Could not normalise Bayesian samples for plotting: {exc}")
            return

        for dimidx in range(n_dim):
            if hasattr(self, '_dimension_names') and dimidx < len(self._dimension_names):
                dimension_name = self._dimension_names[dimidx]
            else:
                dimension_name = f"Dimension {dimidx + 1}"

            color = color_strings[dimidx % len(color_strings)]

            # Projection of all BO samples onto this coordinate axis. The x-axis
            # is relative to the BO centre/start, matching the old SfP d_range
            # convention as closely as possible.
            meas_plot = PlotData(
                ObservableList(), ObservableList(),
                'scatter', color=color,
                label=f"{label_prefix} {dimension_name} samples")
            meas_plot.x.extend((X_abs[:, dimidx] - center[dimidx]).tolist())
            meas_plot.y.extend(y.tolist())

            # GP posterior mean slice through the best measured point. This is
            # the Bayesian analogue of the old fitted curve, but it is a slice of
            # the N-dimensional surrogate, not an independent 1D fit.
            slice_plot = PlotData(
                ObservableList(), ObservableList(),
                color=color,
                label=f"{label_prefix} {dimension_name} GP slice")
            try:
                x_line_abs = np.linspace(bounds[dimidx, 0], bounds[dimidx, 1], num=100)
                X_line_abs = np.tile(best, (len(x_line_abs), 1))
                X_line_abs[:, dimidx] = x_line_abs
                X_line_n = PeakSearcher._normalise_to_unit_box(X_line_abs, bounds)
                mu_line, _ = PeakSearcher._gp_predict_unit_box(
                    Xn_train, y, X_line_n,
                    length_scale=length_scale,
                    noise_sigma=noise_sigma,
                    signal_sigma=1.0,
                    kernel_cutoff=0.0,
                )
                slice_plot.x.extend((x_line_abs - center[dimidx]).tolist())
                slice_plot.y.extend(mu_line[:-1].tolist())
                slice_plot.y.append(float(mu_line[-1]))
            except Exception as exc:
                self.logger.warning(f"Could not create GP slice for {dimension_name}: {exc}")

            # Mark the best actually measured BO point. This mirrors the x marker
            # in the original coordinate-search plots.
            opt_pos_plot = PlotData(
                ObservableList(), ObservableList(),
                marker='x', markersize=10, color=color,
                label=f"{label_prefix} {dimension_name} best")
            opt_pos_plot.x.extend([float(best[dimidx] - center[dimidx])])
            opt_pos_plot.y.append(best_power)

            if dimidx < n_dim / 2:
                self.plots_left.append(meas_plot)
                self.plots_left.append(slice_plot)
                self.plots_left.append(opt_pos_plot)
            else:
                self.plots_right.append(meas_plot)
                self.plots_right.append(slice_plot)
                self.plots_right.append(opt_pos_plot)

            # TODO: omuell added to test
            plt.plot(meas_plot)
            plt.show()

    @staticmethod
    def get_default_parameter():
        return {
            'Laser wavelength': MeasParamInt(value=1550, unit='nm'),
            'Laser power': MeasParamFloat(value=0.0, unit='dBm'),
            'Power Meter range': MeasParamFloat(value=0.0, unit='dBm'),
            'Search radius': MeasParamFloat(value=5.0, unit='um'),
            # Keeps the external call syntax unchanged: call search_for_peak(),
            # then select the algorithm from the GUI/settings.
            'Peak search optimizer': MeasParamList(options=[
                'coordinate search',
                'coordinate search + Bayesian refiner',
                'Bayesian 4D'
            ]),
            # Controls how each 1D coordinate sweep chooses its target. The
            # recommended default avoids assuming that the profile is Gaussian.
            '(coordinate SfP) Peak estimator': MeasParamList(options=[
                'quadratic around measured max',
                'measured max',
                'Gaussian fit'
            ]),
            # Bayesian optimization parameters. The radius is relative to the
            # starting point for true 4D BO, and relative to the coordinate-search
            # result for the refiner.
            '(Bayesian) Search radius': MeasParamFloat(value=2.0, unit='um'),
            '(Bayesian) Initial samples': MeasParamInt(value=12),
            '(Bayesian) Iterations': MeasParamInt(value=30),
            '(Bayesian) Measurements per point': MeasParamInt(value=1),
            '(Bayesian) Candidate points per iteration': MeasParamInt(value=4000),
            '(Bayesian) Kernel length scale': MeasParamFloat(value=0.35, unit='unit-box'),
            '(Bayesian) Noise sigma': MeasParamFloat(value=0.05, unit='dB'),
            '(Bayesian) Expected improvement xi': MeasParamFloat(value=0.01, unit='dB'),
            '(Bayesian) Random seed': MeasParamInt(value=0),
            'SfP type': MeasParamList(options=['stepped SfP', 'swept SfP (FA & N7744a PM models only)']),
            '(stepped SfP only) Search step size': MeasParamFloat(value=0.5, unit='um'),
            '(stepped SfP only) Search fiber stabilization time': MeasParamInt(value=200, unit='ms'),
            '(swept SfP only) Search time': MeasParamFloat(value=2.0, unit='s'),
            '(swept SfP only) Number of points': MeasParamInt(value=500)
        }

    @staticmethod
    def get_wanted_instrument():
        return ['Laser', 'Power Meter']

    def search_for_peak(self):
        """Main Search For Peak routine
        Uses a 2D gaussian fit for all four dimensions.

        Returns
        -------
        dict
            A dict containing the parameters used for the SFP, the estimated through power,
            and gaussian fitting information.
        """
        # double check if mover is actually enabled
        if self.mover.left_calibration is None and self.mover.right_calibration is None:
            raise RuntimeError(
                "The Search for Peak requires at least one left or right stage configured.")

        if self.mover.left_calibration and self.mover.right_calibration:
            self._dimension_names = self.DIMENSION_NAMES_TWO_STAGES
        else:
            self._dimension_names = self.DIMENSION_NAMES_SINGLE_STAGE

        # load laser and powermeter
        self.instr_powermeter = self.get_instrument('Power Meter')
        self.instr_laser = self.get_instrument('Laser')

        # double check if instruments are initialized, otherwise throw error
        if self.instr_powermeter is None:
            raise RuntimeError('Search for Peak Power Meter not yet defined!')
        if self.instr_laser is None:
            raise RuntimeError('Search for Peak Laser not yet defined!')

        # initialize plotting
        self.plots_left.clear()
        self.plots_right.clear()

        # open connection to instruments
        self.instr_laser.open()
        self.instr_powermeter.open()

        self.logger.debug('Executing Search for Peak with the following parameters: {:s}'.format(
            "\n".join([str(name) + " = " + str(param.value) + " " + str(param.unit) for name, param in
                       self.parameters.items()])
        ))

        # setup results dictionary and save all parameters
        results = {
            'name': self.name,
            'parameter': {},
            'start location': None,
            'start through power': None,
            'optimized location': None,
            'optimized through power': None,
            'fitting information': {}
        }
        for param_name, cfg_param in self.parameters.items():
            results['parameter'][param_name] = str(cfg_param.value) + str(cfg_param.unit)

        # send user specified parameters to instruments
        self.instr_laser.wavelength = self.parameters['Laser wavelength'].value
        self.instr_laser.power = self.parameters['Laser power'].value
        self.instr_powermeter.unit = 'dBm'
        self.instr_powermeter.wavelength = self.parameters['Laser wavelength'].value
        self.instr_powermeter.range = self.parameters['Power Meter range'].value

        # get stage speed for later reference
        v0 = self.mover.speed_xy
        acc0 = self.mover.acceleration_xy

        # stop all previous logging
        self.instr_powermeter.logging_stop()

        # switch on laser
        with self.instr_laser:
            with self.mover.set_stages_coordinate_system(CoordinateSystem.STAGE):
                # read parameters for SFP
                sfp_type = self.parameters.get('SfP type').value
                radius_us = self.parameters.get('Search radius').value

                # parameters specifically for stepped sfp
                stepsize_us = self.parameters['(stepped SfP only) Search step size'].value
                pause_time_ms = self.parameters['(stepped SfP only) Search fiber stabilization time'].value

                # parameters specifically for swept SfP
                t_sweep = self.parameters.get('(swept SfP only) Search time').value
                no_points = int(self.parameters.get('(swept SfP only) Number of points').value)

                # New optimizer selection. Defaults preserve the old external syntax:
                # the user still calls search_for_peak(), and this setting decides
                # whether we do coordinate search, coordinate search + BO refinement,
                # or direct local Bayesian optimisation in all dimensions.
                peak_optimizer = self._param_value('Peak search optimizer', 'coordinate search')
                peak_estimator = self._param_value('(coordinate SfP) Peak estimator', 'quadratic around measured max')
                bayes_radius_us = float(self._param_value('(Bayesian) Search radius', min(radius_us, 2.0)))
                bayes_n_initial = int(self._param_value('(Bayesian) Initial samples', 12))
                bayes_n_iter = int(self._param_value('(Bayesian) Iterations', 30))
                bayes_n_avg = int(self._param_value('(Bayesian) Measurements per point', 1))
                bayes_n_candidates = int(self._param_value('(Bayesian) Candidate points per iteration', 4000))
                bayes_length_scale = float(self._param_value('(Bayesian) Kernel length scale', 0.35))
                bayes_noise_sigma = float(self._param_value('(Bayesian) Noise sigma', 0.05))
                bayes_xi = float(self._param_value('(Bayesian) Expected improvement xi', 0.01))
                bayes_seed = int(self._param_value('(Bayesian) Random seed', 0))
                if bayes_seed == 0:
                    bayes_seed = None

                # define parameters
                # the sweep velocity is the distance passed (twice the search
                # radius) divided by the sweep time
                v_sweep_ums = 2 * radius_us / t_sweep
                avg_time = t_sweep / float(no_points)
                unit = 'dBm'

                # find the current positions of the stages as starting point for
                # SFP
                _left_start_coordinates = []
                _right_start_coordinates = []
                if self.mover.left_calibration:
                    _left_start_coordinates = self.mover.left_calibration.get_position().to_list()[
                        :2]
                if self.mover.right_calibration:
                    _right_start_coordinates = self.mover.right_calibration.get_position().to_list()[
                        :2]
                start_coordinates = _left_start_coordinates + _right_start_coordinates
                current_coordinates = start_coordinates.copy()

                self.logger.debug(f"Start Position: {start_coordinates}")

                estimated_through_power = -99.0

                # get start statistics
                results['start location'] = start_coordinates.copy()
                results['start through power'] = self.instr_powermeter.power

                if str(peak_optimizer).lower() == 'bayesian 4d':
                    # True joint optimisation: all stage coordinates are varied
                    # together, so this does not assume separability between x/y
                    # or input/output fiber positions.
                    current_coordinates, estimated_through_power, bayes_info = self._run_bayesian_optimisation(
                        start_coordinates=start_coordinates,
                        search_radius_um=bayes_radius_us,
                        pause_time_ms=pause_time_ms,
                        n_initial=bayes_n_initial,
                        n_iter=bayes_n_iter,
                        n_averages=bayes_n_avg,
                        n_candidates=bayes_n_candidates,
                        length_scale=bayes_length_scale,
                        noise_sigma=bayes_noise_sigma,
                        xi=bayes_xi,
                        seed=bayes_seed,
                        label='Bayesian 4D' if len(start_coordinates) == 4 else 'Bayesian ND',
                    )
                    results['fitting information']['Bayesian 4D'] = bayes_info
                    self._add_bayesian_projection_plots(bayes_info, label_prefix='Bayesian 4D')

                else:
                        # do sweep for every dimension
                    # color cycle strings for matplotlib
                    color_strings = ['C' + str(i) for i in range(10)]
                    for dimidx, p_start in enumerate(start_coordinates):

                        dimension_name = self._dimension_names[dimidx]

                        # create new plotting dataset for measurement
                        meas_plot = PlotData(ObservableList(), ObservableList(),
                                            'scatter', color=color_strings[dimidx])
                        fit_plot = PlotData(ObservableList(), ObservableList(),
                                            color=color_strings[dimidx], label=dimension_name)
                        opt_pos_plot = PlotData(ObservableList(), ObservableList(),
                                                marker='x', markersize=10, color=color_strings[dimidx])
                        if dimidx < len(start_coordinates) / 2:
                            self.plots_left.append(meas_plot)
                            self.plots_left.append(fit_plot)
                            self.plots_left.append(opt_pos_plot)
                        else:
                            self.plots_right.append(meas_plot)
                            self.plots_right.append(fit_plot)
                            self.plots_right.append(opt_pos_plot)

                        # differentiate between the two types of SfP
                        if sfp_type == 'swept SfP (FA & N7744a PM models only)':
                            allowed_pm_classes = ['PowerMeterN7744A', 'PowerMeterSimulator']
                            # complain if user selects a Power Meter that is not
                            # compatible with new Search for Peak
                            if self.instr_powermeter.__class__.__name__ not in allowed_pm_classes:
                                raise RuntimeError(
                                    'swept SfP is only compatible with Keysight N7744A PM models, not {}'.format(
                                    self.instr_powermeter.__class__.__name__))
                            # move stage to initial position and setup
                            current_coordinates[dimidx] = p_start - radius_us
                            self._move_stages_absolute(current_coordinates)

                            # setup power meter logging feature
                            # autogain attribute exists only for N7744A, no effect on
                            # other
                            self.instr_powermeter.autogain = False
                            self.instr_powermeter.range = self.parameters['Power Meter range'].value
                            self.instr_powermeter.unit = unit
                            self.instr_powermeter.averagetime = avg_time
                            self.instr_powermeter.logging_setup(
                                n_measurement_points=no_points,
                                triggered=True,
                                trigger_each_meas_separately=False)
                            self.instr_powermeter.logging_start()

                            # take a tiny break
                            time.sleep(0.1)

                            current_coordinates[dimidx] = p_start + radius_us
                            # empirically determined acceleration
                            acc_umps2 = 50
                            self.mover.speed_xy = v_sweep_ums
                            self.mover.acceleration_xy = acc_umps2
                            # start logging at powermeter
                            self.instr_powermeter.trigger()
                            # mover_time_lower = time.time()
                            self._move_stages_absolute(current_coordinates)
                            # mover_time_upper = time.time()

                            while self.instr_powermeter.logging_busy():
                                time.sleep(0.1)
                            pm_data = self.instr_powermeter.logging_get_data()

                            # pay attention to unit here
                            IL_meas = pm_data

                            # calculate the estimated movement profile, given constant
                            # acceleration of the stages
                            _, d_range, _, _ = trapezoidal_velocity_profile_by_integration(start_position_m=-radius_us,
                                                                                        stop_position_m=radius_us,
                                                                                        max_speed_mps=v_sweep_ums,
                                                                                        const_acceleration_mps2=acc_umps2,
                                                                                        n_output_points=len(IL_meas))

                            # plot it
                            meas_plot.x = d_range
                            meas_plot.y = IL_meas

                        elif sfp_type == 'stepped SfP':
                            # create range of N measurement points from x-Delta to
                            # x+Delta
                            d_range = np.arange(-radius_us, radius_us +
                                                stepsize_us, stepsize_us)

                            # go through all measurement points for this coordinate and
                            # record IL
                            IL_meas = np.empty(len(d_range))

                            for measidx, d_current in enumerate(d_range):
                                # move stages to currently probed coordinate
                                current_coordinates[dimidx] = d_current + p_start
                                self._move_stages_absolute(current_coordinates)

                                # take a break to let fiber-vibration die off
                                time.sleep(pause_time_ms / 1000)

                                # take IL measurement
                                loss = self.instr_powermeter.power

                                # save data
                                # do not trigger plot update just yet
                                meas_plot.x.extend([d_current])
                                meas_plot.y.append(loss)

                                IL_meas[measidx] = loss

                        else:
                            raise ValueError(
                                'invalid SfP type given! Options are `stepped SfP` or `swept SfP`.')

                        self.logger.debug('SFP results:')
                        self.logger.debug('coordinates:' + str(d_range))
                        self.logger.debug('IL: ' + str(IL_meas))

                        # default assignments before SFP decision
                        optimized_target = 0
                        popt = None
                        perr_std_dev = None
                        fit_msg = None
                        sfp_msg = None

                        # 1st decision: did the power meter always return useful data?
                        if ~np.all(np.isfinite(IL_meas)):
                            sfp_msg = f'SFP failed on dimension {dimension_name} because not all measured IL values are finite.' + \
                                    ' Change of power meter range required. Moving back to start point.'
                            self.logger.warning(sfp_msg)
                        else:
                            # 2nd decision: estimate the peak. The default no longer
                            # assumes a Gaussian. It either takes the measured maximum
                            # or a local quadratic interpolation around that maximum.
                            try:
                                (popt, perr_std_dev, d_best, estimated_through_power,
                                 fit_x, fit_y, fit_msg) = self.estimate_1d_peak(
                                    d_range, IL_meas, method=peak_estimator)
                            except RuntimeError as err:
                                popt = PeakSearcher._gaussian_param_initial_guess(d_range, IL_meas)
                                d_best = float(popt[1])
                                estimated_through_power = float(np.nanmax(IL_meas))
                                perr_std_dev = None
                                fit_x, fit_y = None, None
                                fit_msg = f"Peak estimation failed ({err}). Using measured maximum."
                                self.logger.warning(fit_msg)

                            # 3rd decision: judge feasibility of the estimated target.
                            # This keeps the original safety behavior: do not move far
                            # outside the commanded scan range based on a model.
                            if abs(d_best) > 1.5 * radius_us:
                                optimized_target = 0
                                estimated_through_power = float(np.interp(optimized_target, d_range, IL_meas))
                                sfp_msg = 'Movement would be more than 1.5x search radius. Moving back to start point.'
                                self.logger.warning(sfp_msg)
                            else:
                                optimized_target = float(d_best)
                                sfp_msg = f'Moving to optimized fiber location.'

                            # Plot the optional estimator curve. For measured-max mode
                            # there is intentionally no fake fit curve.
                            if fit_x is not None and fit_y is not None:
                                fit_plot.x.extend(fit_x)
                                fit_plot.y.extend(fit_y[0:-1])
                                fit_plot.y.append(fit_y[-1])

                            # Mark the point where we move to in any case.
                            opt_pos_plot.x.extend([optimized_target])
                            opt_pos_plot.y.append(estimated_through_power)

                        # inform user and store the fitting information
                        self.logger.debug(
                            f"Search for peak for dimension {dimension_name} finished. "
                            f"Fitter message: {fit_msg} -- SFP decision: {sfp_msg} "
                            f"Moving to location: {optimized_target:.3f}um with estimated through power"
                            f" of {estimated_through_power:.1f}dBm.")

                        results['fitting information'][dimension_name] = {
                            'optimized parameters': list(popt) if popt is not None else None,
                            'parameter estimation error std dev': list(perr_std_dev) if perr_std_dev is not None else None,
                            'fitter message': str(fit_msg),
                            'sfp decision': str(sfp_msg)}

                        # reset speed and acceleration to original
                        self.mover.speed_xy = v0
                        self.mover.acceleration_xy = acc0

                        # final move of fiber in this dimensions final decision
                        current_coordinates[dimidx] = optimized_target + p_start
                        self._move_stages_absolute(current_coordinates)

                if str(peak_optimizer).lower() == 'coordinate search + bayesian refiner':
                    # Experimental local BO refinement after the normal coordinate
                    # sweep. This is usually the safest first comparison because
                    # the existing SfP gets close, then BO only explores a small
                    # local N-dimensional box around that result.
                    current_coordinates, estimated_through_power, bayes_info = self._run_bayesian_optimisation(
                        start_coordinates=current_coordinates,
                        search_radius_um=bayes_radius_us,
                        pause_time_ms=pause_time_ms,
                        n_initial=bayes_n_initial,
                        n_iter=bayes_n_iter,
                        n_averages=bayes_n_avg,
                        n_candidates=bayes_n_candidates,
                        length_scale=bayes_length_scale,
                        noise_sigma=bayes_noise_sigma,
                        xi=bayes_xi,
                        seed=bayes_seed,
                        label='Bayesian refiner',
                    )
                    results['fitting information']['Bayesian refiner'] = bayes_info
                    self._add_bayesian_projection_plots(bayes_info, label_prefix='Bayesian refiner')

        # close instruments
        self.instr_laser.close()
        self.instr_powermeter.close()

        # save final result to log
        loc_str = " x ".join(["{:.3f}um".format(p)
                             for p in current_coordinates])
        self.logger.info(
            f"Search for peak finished: maximum estimated output power of {estimated_through_power:.1f}dBm"
            f" at {loc_str:s}.")

        # save end result and return
        results['optimized location'] = current_coordinates.copy()
        results['optimized through power'] = estimated_through_power

        return results

    def _search_for_peak_with_forced_optimizer(self, optimizer_name):
        """Run search_for_peak() with a forced optimizer mode.

        This gives you explicit methods for lab comparisons while keeping the
        normal LabExT external syntax and return dictionary unchanged.
        """
        old_forced = getattr(self, '_forced_peak_optimizer', None)
        self._forced_peak_optimizer = optimizer_name
        try:
            return self.search_for_peak()
        finally:
            self._forced_peak_optimizer = old_forced

    def search_for_peak_bayesian_4d(self):
        """Experimental true joint Bayesian optimisation in all x/y dimensions.

        For two stages this is genuinely 4D: [left x, left y, right x, right y].
        It returns the same result dictionary as search_for_peak().
        """
        return self._search_for_peak_with_forced_optimizer('Bayesian 4D')

    def search_for_peak_bayesian_refiner(self):
        """Experimental BO refinement after the normal coordinate search.

        This first runs the selected coordinate-search peak estimator and then
        performs a local N-dimensional Bayesian refinement around the found
        position. It returns the same result dictionary as search_for_peak().
        """
        return self._search_for_peak_with_forced_optimizer('coordinate search + Bayesian refiner')

    def search_for_peak_coordinate_only(self):
        """Run only the coordinate-search baseline with the selected 1D estimator.

        Useful as a control measurement when comparing against the Bayesian
        refiner and the true joint Bayesian optimisation.
        """
        return self._search_for_peak_with_forced_optimizer('coordinate search')

    def _move_stages_absolute(self, coordinates: list):
        with self.mover.set_stages_coordinate_system(CoordinateSystem.STAGE):
            if self.mover.left_calibration and self.mover.right_calibration:
                leftz = self.mover.left_calibration.get_position().z
                rightz = self.mover.right_calibration.get_position().z
                assert len(coordinates) == 4
                self.mover.left_calibration.move_absolute(
                    StageCoordinate.from_list(coordinates[:2] + [leftz]))
                self.mover.right_calibration.move_absolute(
                    StageCoordinate.from_list(coordinates[2:] + [rightz]))
            elif self.mover.left_calibration:
                leftz = self.mover.left_calibration.get_position().z
                assert len(coordinates) == 2
                self.mover.left_calibration.move_absolute(
                    StageCoordinate.from_list(coordinates + [leftz]))
            elif self.mover.right_calibration:
                rightz = self.mover.right_calibration.get_position().z
                assert len(coordinates) == 2
                self.mover.right_calibration.move_absolute(
                    StageCoordinate.from_list(coordinates + [rightz]))
            else:
                raise RuntimeError()

    def update_params_from_savefile(self):
        if not os.path.isfile(self.settings_path_full):
            self.logger.debug(f"SFP Parameter save file at {self.settings_path_full} not found. "
                              f"Using default parameters.")
            return

        with open(self.settings_path_full, 'r') as json_file:
            data = json.loads(json_file.read())

        for param_name, param_value in data["data"].items():
            self.parameters[param_name].value = param_value

    def algorithm(self, device, data, instruments, parameters):
        raise NotImplementedError()
