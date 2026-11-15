"""Spatial+ (Dupont, Wood & Augustin 2022) with one penalized smooth per site.

Pinned here:

- the fast fits (ridge columns plus a Woodbury correction) equal explicit
  dense penalized least squares: the hat matrix's quadratic forms, residuals,
  edf, GCV and REML (up to a constant), and the criteria's gradients;
- one group, no covariates: β̂⁺ equals a dense reference with the same basis
  and penalty, each λ chosen by a fine GCV grid, to 1e-6;
- λ = 0 gives the unpenalized basis adjustment's slope (Dupont §4);
- a Dupont-type confounded field: the penalized spatial model without the
  residual step is biased, Spatial+ keeps under a fifth of that bias, under
  GCV and REML;
- two distant groups of different roughness get different λ, and without
  covariates step 1 equals fitting each group alone;
- the resamples refit both steps at fixed λ: weight 1 everywhere gives the
  full-sample slope, Poisson weights an explicit weighted penalized refit;
- check criteria write rows marked ``check``; the split rows carry the
  method and the criterion and are summarised per criterion;
- a run with ``spatial_plus`` writes ``spatial_plus.csv`` and Spatial+ rows
  on every split beside unadjusted standard rows.
"""

import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for _p in (HERE, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import geopandas  # noqa: F401  (pyproj before torchvision, see vision.py)
import numpy as np
from scipy.optimize import minimize_scalar

from geofuse import spatial_plus as sp
from geofuse.jobs import outer_splits


def _field(seed=0, n1=250, n2=200, k=60):
    rng = np.random.default_rng(seed)
    xy = np.vstack([rng.random((n1, 2)) * 8000, rng.random((n2, 2)) * 6000 + 1e6])
    labels = np.r_[np.zeros(n1, int), np.ones(n2, int)]
    z = np.sin(xy[:, 0] / 1500) + np.cos(xy[:, 1] / 2000)
    cov = rng.normal(size=(len(xy), 2))
    x = 0.5 * z + 0.3 * cov[:, 0] + 0.5 * rng.normal(size=len(xy))
    y = 0.3 * x + z + 0.2 * cov[:, 1] + rng.normal(size=len(xy))
    return xy, labels, cov, x, y, sp.build(xy, labels, k=k)


def _dense(sm, dense, lam):
    """The explicit design ``[dense, trends, functions]`` and its penalty."""
    n = sm.n
    parts = [] if dense is None else [dense]
    pen = [] if dense is None else [np.zeros(dense.shape[1])]
    for g in sm.groups:
        T = np.zeros((n, g.Qt.shape[1]))
        T[g.rows] = g.Qt
        parts.append(T)
        pen.append(np.zeros(T.shape[1]))
    for j, g in enumerate(sm.groups):
        C = np.zeros((n, g.m))
        C[g.rows] = g.C
        parts.append(C)
        pen.append(np.full(g.m, lam[j]))
    return np.column_stack(parts), np.concatenate(pen)


def _hat(X, P):
    return X @ np.linalg.pinv(X.T @ X + np.diag(P)) @ X.T


class FitTests(unittest.TestCase):
    def test_the_fast_fit_is_dense_penalized_least_squares(self):
        _xy, _l, cov, x, y, sm = _field()
        design = sp.Design(sm, cov)
        top = np.log(design.s.max())
        lam = np.exp([top - 5, top - 2])
        X, P = _dense(sm, cov, lam)
        H = _hat(X, P)
        n = len(y)
        resid = y - H @ y
        self.assertAlmostEqual(
            design.quad(x, y, lam), x @ (np.eye(n) - H) @ y, delta=1e-8
        )
        np.testing.assert_allclose(
            design.residual(x, lam), x - H @ x, atol=1e-10
        )
        self.assertAlmostEqual(design.edf(lam)[0], np.trace(H), places=8)
        gcv = n * (resid @ resid) / (n - np.trace(H)) ** 2
        self.assertAlmostEqual(
            design.criterion(y, np.log(lam), "gcv")[0], gcv, places=10
        )

    def test_reml_differs_from_the_explicit_one_by_a_constant(self):
        _xy, _l, cov, _x, y, sm = _field()
        design = sp.Design(sm, cov)
        top = np.log(design.s.max())

        def explicit(lam):
            X, P = _dense(sm, cov, lam)
            A = X.T @ X + np.diag(P)
            theta = np.linalg.solve(A, X.T @ y)
            pen_rss = float(((y - X @ theta) ** 2).sum() + theta @ (P * theta))
            nu = len(y) - design.r0
            pos = P[P > 0]
            return (
                nu * np.log(pen_rss / nu)
                + np.linalg.slogdet(A)[1]
                - np.log(pos).sum()
            )

        values = []
        for shift in (-6.0, -3.0, 0.0):
            lam = np.exp([top + shift, top + shift - 1])
            values.append(explicit(lam) - design.criterion(y, np.log(lam), "reml")[0])
        self.assertAlmostEqual(values[0], values[1], places=6)
        self.assertAlmostEqual(values[0], values[2], places=6)

    def test_the_gradients(self):
        _xy, _l, cov, x, y, sm = _field()
        for dense in (None, cov, np.column_stack([cov, x])):
            design = sp.Design(sm, dense)
            rho = np.log(design.s.max()) - np.array([6.0, 3.0])
            for kind in sp.CRITERIA:
                _v, grad = design.criterion(y, rho, kind)
                h = 1e-5
                fd = [
                    (
                        design.criterion(y, rho + h * e, kind)[0]
                        - design.criterion(y, rho - h * e, kind)[0]
                    )
                    / (2 * h)
                    for e in np.eye(2)
                ]
                np.testing.assert_allclose(grad, fd, rtol=1e-5, atol=1e-8)

    def test_lambda_zero_is_the_unpenalized_basis_adjustment(self):
        _xy, _l, cov, x, y, sm = _field()
        lam = np.full(2, 1e-14)
        design = sp.Design(sm, cov)
        beta = design.quad(x, y, lam) / design.quad(x, x, lam)
        X, _ = _dense(sm, cov, lam)
        ols = np.linalg.lstsq(np.column_stack([x, X]), y, rcond=None)[0][0]
        self.assertAlmostEqual(beta, ols, delta=1e-10)


class OneGroupReferenceTests(unittest.TestCase):
    def test_beta_equals_a_dense_reference_with_lambda_by_a_fine_grid(self):
        rng = np.random.default_rng(4)
        n = 400
        xy = rng.random((n, 2)) * 10_000
        z = np.sin(xy[:, 0] / 1800) * np.cos(xy[:, 1] / 2500)
        x = z + 0.6 * rng.normal(size=n)
        y = 0.25 * x + z + rng.normal(size=n)
        sm = sp.build(xy, np.zeros(n, int), k=80)
        step1 = sp.exposure_residual(sm, x, "gcv")
        fit = sp.outcome_fit(sm, y, step1["r"], None, "gcv")

        def gcv_choice(v, dense):
            def g(rho):
                X, P = _dense(sm, dense, [np.exp(rho)])
                H = _hat(X, P)
                r = v - H @ v
                return n * (r @ r) / (n - np.trace(H)) ** 2

            top = np.log(sp.Design(sm, dense).s.max())
            grid = np.linspace(top - 30, top + 8, 400)
            best = grid[int(np.argmin([g(r) for r in grid]))]
            res = minimize_scalar(
                g, bounds=(best - 0.2, best + 0.2), method="bounded",
                options={"xatol": 1e-10},
            )
            return np.exp(res.x)

        lam_x = gcv_choice(x, None)
        X, P = _dense(sm, None, [lam_x])
        r = x - _hat(X, P) @ x
        lam_y = gcv_choice(y, r[:, None])
        X2, P2 = _dense(sm, r[:, None], [lam_y])
        coef = np.linalg.solve(X2.T @ X2 + np.diag(P2), X2.T @ y)
        self.assertAlmostEqual(fit["beta"], coef[0], delta=1e-6 * abs(coef[0]))


class ConfoundingTests(unittest.TestCase):
    """x = 0.5 z + small noise, y = β x + z + ε with a smooth confounder z: the
    penalized spatial model is biased, Spatial+ much less (Dupont et al. 2022,
    §3). Spatial+ keeps a small bias here (measured over 100 replicates: GCV
    0.027 = 2.9 Monte Carlo SE, REML 0.016 = 1.7 SE, against 0.157 and 0.110
    for the plain penalized model): step 1 shrinks part of z back into the
    residual."""

    REPS = int(os.environ.get("GEOFUSE_SPATIAL_PLUS_REPS", "100"))

    def test_spatial_plus_removes_most_of_the_penalty_bias(self):
        beta = 0.5
        for kind in sp.CRITERIA:
            plus, plain = [], []
            for rep in range(self.REPS):
                rng = np.random.default_rng(1000 + rep)
                n = 1000
                xy = rng.random((n, 2)) * 10_000
                z = np.sin(xy[:, 0] / 2500) + np.cos(xy[:, 1] / 3000)
                x = 0.5 * z + 0.15 * rng.normal(size=n)
                y = beta * x + z + 0.5 * rng.normal(size=n)
                sm = sp.build(xy, np.zeros(n, int), k=100)
                r = sp.exposure_residual(sm, x, kind)["r"]
                plus.append(sp.outcome_fit(sm, y, r, None, kind)["beta"])
                plain.append(sp.outcome_fit(sm, y, x, None, kind)["beta"])
            plus, plain = np.array(plus) - beta, np.array(plain) - beta
            se_plus = plus.std(ddof=1) / np.sqrt(len(plus))
            se_plain = plain.std(ddof=1) / np.sqrt(len(plain))
            self.assertLess(abs(plus.mean()), 3.5 * se_plus, kind)
            self.assertLess(abs(plus.mean()), 0.2 * abs(plain.mean()), kind)
            self.assertGreater(abs(plain.mean()), 8 * se_plain, kind)


class GroupTests(unittest.TestCase):
    def test_groups_get_their_own_lambda_and_step_one_fits_each_alone(self):
        rng = np.random.default_rng(7)
        a = rng.random((300, 2)) * 8000
        b = rng.random((300, 2)) * 8000 + 5e5
        xy = np.vstack([a, b])
        x = np.r_[np.sin(a[:, 0] / 4000), np.sin(b[:, 0] / 300) * np.cos(b[:, 1] / 300)]
        x = x + 0.3 * rng.normal(size=len(x))
        labels, rule = sp.group_labels(xy)
        self.assertEqual(rule["n_groups"], 2)
        sm = sp.build(xy, labels, k=80)
        step = sp.exposure_residual(sm, x, "gcv")
        self.assertGreater(abs(np.log(step["lam"][0]) - np.log(step["lam"][1])), 1.0)
        for j in range(2):
            alone = sp.build(xy[labels == j], np.zeros(300, int), k=80)
            r = sp.exposure_residual(alone, x[labels == j], "gcv")["r"]
            np.testing.assert_allclose(step["r"][labels == j], r, atol=1e-10)

    def test_a_column_defines_the_groups(self):
        xy = np.random.default_rng(0).random((10, 2))
        labels, rule = sp.group_labels(xy, values=np.array(list("aabbbccaab")))
        self.assertEqual(rule["rule"], "column")
        self.assertEqual(len(set(labels)), 3)


class ReplicateTests(unittest.TestCase):
    def test_replicates_refit_both_steps_at_fixed_lambda(self):
        _xy, _l, cov, x, y, sm = _field()
        step1 = sp.exposure_residual(sm, x, "gcv")
        fit = sp.outcome_fit(sm, y, step1["r"], cov, "gcv")
        rng = np.random.default_rng(1)
        W = np.vstack([np.ones(len(y)), rng.poisson(1.0, size=(3, len(y)))]).astype(
            float
        )
        reps = sp.replicate_slopes(
            sm,
            W,
            [{"x": x, "lam": step1["lam"]}],
            [y],
            [{"exposure": 0, "outcome": 0, "lam": fit["lam"]}],
            covariates=cov,
        )
        self.assertAlmostEqual(reps["slope"][0, 0], fit["beta"], delta=1e-10)
        X, _ = _dense(sm, cov, fit["lam"])
        for b in range(1, 4):
            w = W[b]
            r = np.empty(len(y))
            for j, g in enumerate(sm.groups):
                Xg = np.column_stack([g.Qt, g.C])
                Pg = np.r_[np.zeros(g.Qt.shape[1]), np.full(g.m, step1["lam"][j])]
                wg = w[g.rows]
                th = np.linalg.solve(
                    Xg.T @ (Xg * wg[:, None]) + np.diag(Pg), Xg.T @ (wg * x[g.rows])
                )
                r[g.rows] = x[g.rows] - Xg @ th
            _X2, P2 = _dense(sm, np.column_stack([r, cov]), fit["lam"])
            X2 = np.column_stack([r, cov, X[:, cov.shape[1] :]])
            th = np.linalg.lstsq(
                X2.T @ (X2 * w[:, None]) + np.diag(P2), X2.T @ (w * y), rcond=None
            )[0]
            self.assertAlmostEqual(reps["slope"][b, 0], th[0], delta=1e-10)


class PoolTests(unittest.TestCase):
    def test_worker_processes_give_the_serial_resamples(self):
        _xy, _l, cov, x, y, sm = _field(k=40)
        rng = np.random.default_rng(3)
        n = len(y)
        from geofuse import resampling

        W = resampling.row_weights(rng.integers(0, n, size=(70, n)), n)
        G = len(sm.groups)
        args = (
            sm,
            W,
            [{"x": x, "lam": np.full(G, 1.0)}],
            [y],
            [{"exposure": 0, "outcome": 0, "lam": np.full(G, 2.0)}],
        )
        serial = sp.replicate_slopes(*args, covariates=cov, workers=1)
        pooled = sp.replicate_slopes(*args, covariates=cov, workers=2)
        for key in ("slope", "delta"):
            np.testing.assert_allclose(pooled[key], serial[key], rtol=1e-12, atol=0)


class ReportTests(unittest.TestCase):
    def _report(self, **kw):
        xy, labels, cov, x, y, _sm = _field(k=40)
        rng = np.random.default_rng(2)
        other = x + rng.normal(size=len(x))
        grip = 0.1 * x + rng.normal(size=len(x))
        return sp.report(
            xy,
            {"cgi": ("cgi", x), "ndvi": ("channel", other)},
            y,
            {"grip": grip},
            cov,
            labels,
            k=40,
            scales={
                "cgi": {"iqr": 0.5, "span": 1.5},
                "ndvi": {"iqr": 1.0, "span": 3.0},
            },
            n_boot=30,
            **kw,
        )

    def test_check_criteria_write_rows_marked_check(self):
        out = self._report(criterion="gcv", checks=["reml"])
        crits = {(r["criterion"], r["check"]) for r in out["slopes"]}
        self.assertEqual(crits, {("gcv", False), ("reml", True)})
        self.assertTrue(out["lambda_fixed_in_resamples"])
        row = next(
            r
            for r in out["slopes"]
            if r["exposure"] == "cgi" and r["role"] == "target" and not r["check"]
        )
        self.assertAlmostEqual(row["per_iqr"]["estimate"], 0.5 * row["estimate"])
        self.assertLessEqual(row["ci_low"], row["estimate"])
        self.assertGreater(row["se_vp"], 0)
        self.assertGreater(row["se_ve"], 0)
        for key in ("d", "delta_d", "paired"):
            self.assertTrue(out[key], key)
            self.assertIsNotNone(out[key][0]["ci_low"], key)

    def test_split_rows_and_summary(self):
        out = self._report(criterion="gcv", checks=["reml"])
        rows = []
        for k in range(3):
            for r in outer_splits.spatial_plus_rows(out):
                r.update(split=k, test_share=0.25)
                rows.append(r)
        quantities = {r["quantity"] for r in rows}
        self.assertTrue(
            {
                "held_out_per_sd0",
                "exposure_per_sd0",
                "exposure_negative_control_per_sd0",
                "negative_control_d",
                "specificity_difference",
                "cgi_minus_channel",
            }
            <= quantities
        )
        self.assertTrue(all(r["spatial_method"] == "spatial_plus" for r in rows))
        summary = outer_splits.summarise(rows)
        beta = [e for e in summary if e["quantity"] == "held_out_per_sd0"]
        self.assertEqual({e["spatial_criterion"] for e in beta}, {"gcv", "reml"})
        self.assertEqual({e["spatial_check"] for e in beta}, {False, True})
        table = outer_splits.spatial_plus_table(out)
        self.assertEqual({r["step"] for r in table}, {"exposure", "outcome"})

    def test_the_command_line_summary(self):
        from geofuse.cli.results import _spatial_plus_lines

        out = self._report(criterion="gcv", checks=["reml"])
        lines = _spatial_plus_lines(out)
        self.assertIn("penalties by gcv (checks: reml)", lines[0])
        self.assertTrue(
            any(line.lstrip().startswith("cgi (reml, check)") for line in lines)
        )
        self.assertEqual(
            _spatial_plus_lines({"skipped": "binary outcome"}),
            ["  Spatial+         skipped (binary outcome)"],
        )


class RunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_fusion_run import _write_study, run_study

        cls.folder = tempfile.mkdtemp(prefix="geofuse-spatial-plus-")
        _write_study(cls.folder)
        cls.ctx, cls.bundle = run_study(
            cls.folder,
            """
            mode = "cross-sectional"
            ndvi = "ndvi_2015.tif"
            gvi = "gvi.gpkg"
            """,
            spatial_adjust_method="spatial_plus",
            spatial_adjust_select="df_outcome_ic",
            spatial_plus_k=30,
            spatial_plus_checks=["reml"],
            spatial_adjust_eps_m=50_000.0,
            outer_splits=2,
            comparators=["ndvi_here", "noise"],
            report_paired_bootstrap=20,
        )

    def test_the_report_and_its_table(self):
        plus = self.bundle["spatial_plus"]
        self.assertNotIn("skipped", plus)
        self.assertEqual(plus["criterion"], "gcv")
        path = os.path.join(self.bundle["report_dir"], "spatial_plus.csv")
        self.assertTrue(os.path.exists(path), path)

    def test_every_split_has_spatial_plus_rows_beside_unadjusted_ones(self):
        rows = self.bundle["outer_splits"]["splits"]
        plus = [r for r in rows if r.get("spatial_method") == "spatial_plus"]
        self.assertEqual({r["split"] for r in plus}, {0, 1})
        standard = [r for r in rows if r["quantity"] == "held_out_beta"]
        self.assertTrue(standard)
        self.assertTrue(all("spatial_method" not in r for r in standard))


if __name__ == "__main__":
    unittest.main()
