use ndarray::prelude::*;

mod vector;

use vector::V;

#[derive(Debug, Copy, Clone, PartialEq)]
struct ConvexHullPoint<D: Copy> {
    x: usize,
    y: f64,
    /// Slope between this and the previous point
    s: f64,
    /// Custom data
    data: D,
}

/// 1D convex hull of a set of points
/// (0, f(0)), (1, f(1)), ...,
/// Array `data` contains custom values for individual points.
fn convex_hull_1d<D: Copy>(
    f: ArrayView1<f64>,
    data: ArrayView1<D>,
    hull: &mut Vec<ConvexHullPoint<D>>,
) {
    assert_eq!(data.raw_dim(), f.raw_dim());
    // The algorithm adds points one at a time and removes previous points until the resulting shape
    // is convex. Since every point is considered exactly once and possibly removed at most once,
    // the algorithm complexity is linear in the input size.
    let mut points = f.iter().zip(data).enumerate();

    let (x, (&y, &data)) = points.next().expect("No points");
    let s = f64::NEG_INFINITY;
    let mut prev = ConvexHullPoint { x, y, s, data };

    for (x, (&y, &data)) in points {
        // Add point (x, y) to the hull.
        // Remove previous points in the hull until the result is convex: prev.s < s.
        let s = loop {
            let s = (y - prev.y) / (x - prev.x) as f64;
            if prev.s < s {
                break s;
            }
            // This cannot panic since if p is empty then prev.s = f64::NEG_INFINITY
            prev = hull.pop().unwrap();
        };
        hull.push(prev);
        prev = ConvexHullPoint { x, y, s, data };
    }
    hull.push(prev);
}

/// n dimensional Legendre-Fenchel transform
/// on a regular grid with gridstep `h` of the same size for both input and output.
///
/// It keeps track of argmax (at which node the maximum occured) and returns the provided custom
/// data for that node. This can be used for example for computing the push-forward map in the
/// c-transform.
///
/// Algorithm based on Lucet, Faster than the Fast Legendre Transform, the Linear-time
/// Legendre Transform, Numerical Algorithms 16 (1997)
///
/// The complexity is linear in the number of nodes.
///
/// **Panics** if the arrays are not all of the same shape.
pub fn legendre_fenchel<D: Copy, Dim: Dimension>(
    mut in_out: ArrayViewMut<f64, Dim>,
    mut data: ArrayViewMut<D, Dim>,
    h: f64,
) {
    assert_eq!(in_out.raw_dim(), data.raw_dim());
    let mut p = Vec::new(); // buffer to avoid allocations

    // Perform 1d Legendre-Fenchel transforms along each axis repeatedly.
    for axis in (0..in_out.ndim()).rev() {
        // The sign of the result must be flipped except for the last processed dimension.
        let sign = if axis > 0 { -1. } else { 1. };
        azip!((phi_star_row in in_out.lanes_mut(Axis(axis)), data_row in data.lanes_mut(Axis(axis))) {

            p.clear();
            convex_hull_1d(phi_star_row.view(), data_row.view(), &mut p);

            // Legendre-Fenchel of the 1D convex hull
            let mut p_iter = p.iter();
            let mut active_point = p_iter.next().expect("convex hull should be nonempty");
            let mut next = p_iter.next();

            azip!((index j, phi_star in phi_star_row, data in data_row) {
                while let Some(point) = next {
                    if j as f64 * (h * h) <= point.s {
                        break;
                    }
                    active_point = point;
                    next = p_iter.next();
                }

                *phi_star = sign * (j as f64 * active_point.x as f64 * (h * h) - active_point.y);
                *data = active_point.data;
            });
        });
    }
}

/// 2D c-transform with quadratic cost: c(x,y) = |x - y|^2/2.
pub fn ctransform2<D: Copy>(mut in_out: ArrayViewMut2<f64>, id: ArrayViewMut2<D>, h: f64) {
    azip!((index (i, j), phi in in_out.view_mut()) {
        let i = i as f64;
        let j = j as f64;
        *phi = 0.5 * (h * h) * (i * i + j * j) - *phi
    });

    legendre_fenchel(in_out.view_mut(), id, h);

    azip!((index (i, j), phi in in_out.view_mut()) {
        let i = i as f64;
        let j = j as f64;
        *phi = 0.5 * (h * h) * (i * i + j * j) - *phi
    });
}

/// Euclidean distance
fn dist(a: V<[f64; 2]>, b: V<[f64; 2]>) -> f64 {
    ((a.0[0] - b.0[0]).powi(2) + (a.0[1] - b.0[1]).powi(2)).sqrt()
}

/// Push-forward T_#φ μ
///
/// The map T_#φ is given by x - \nabla \phi^c.
///
/// Follows the algorithm in <https://github.com/Math-Jacobs/bfm/blob/c93de454c49c6958e0a8e18b285c2e005b55507f/python/src/main.cpp#L345> with a few modifications.
///
/// In particular, \nabla \phi^c at the cell corner (X below) is estimated using a simpler finite
/// difference using the values of \phi^c in the centers of the 4 adjacent cells (o below).
///
/// ```text
/// +---+---+
/// | o | o |
/// +---X---+
/// | o | o |
/// +---+---+
/// ```
///
pub fn push_forward2(
    mut t_mu: ArrayViewMut2<f64>,
    mu: ArrayView2<f64>,
    phi_c: ArrayView2<f64>,
    h: f64,
) {
    assert_eq!(t_mu.raw_dim(), mu.raw_dim());
    assert_eq!(t_mu.raw_dim(), phi_c.raw_dim());
    t_mu.fill(0.);

    let (ni, nj) = phi_c.dim();

    // map each cell
    for ((i, j), mu) in mu.indexed_iter() {
        if *mu == 0. {
            continue;
        }
        // approximate T_#φ in each corner of the cell
        let mut t_phi = [[V([0.; 2]); 2]; 2];
        for k in 0..2 {
            for l in 0..2 {
                // corner index
                let ci = i + k;
                let cj = j + l;
                // values of phi_c at the centers of the adjecent cells
                // Note: By restricting the index to the range 0...ni - 1, we automatically get 0 normal
                // derivative on the boundary of the domain.
                let a = phi_c[(
                    ci.saturating_sub(1).min(ni - 1),
                    cj.saturating_sub(1).min(nj - 1),
                )];
                let b = phi_c[(
                    ci.saturating_sub(0).min(ni - 1),
                    cj.saturating_sub(1).min(nj - 1),
                )];
                let c = phi_c[(
                    ci.saturating_sub(1).min(ni - 1),
                    cj.saturating_sub(0).min(nj - 1),
                )];
                let d = phi_c[(
                    ci.saturating_sub(0).min(ni - 1),
                    cj.saturating_sub(0).min(nj - 1),
                )];
                // x - \nabla \phi^c using a simple 2nd order central finite difference
                t_phi[k][l] = V([
                    ci as f64 * h - 0.5 / h * (b + d - a - c),
                    cj as f64 * h - 0.5 / h * (c + d - a - b),
                ]);
            }
        }
        let x_stretch = dist(t_phi[0][0], t_phi[1][0]).max(dist(t_phi[0][1], t_phi[1][1]));
        let y_stretch = dist(t_phi[0][0], t_phi[0][1]).max(dist(t_phi[1][0], t_phi[1][1]));

        let x_samples = ((x_stretch / h).ceil() as usize).max(1);
        let y_samples = ((y_stretch / h).ceil() as usize).max(1);

        // distribute the cell mass uniformly over all samples
        let mass = *mu / (x_samples as f64 * y_samples as f64);

        for k in 0..x_samples {
            for l in 0..y_samples {
                let a = (k as f64 + 0.5) / (x_samples as f64);
                let b = (l as f64 + 0.5) / (y_samples as f64);

                // sample position in [0,1]²
                let V([x, y]) = (1. - a) * (1. - b) * t_phi[0][0]
                    + (1. - a) * b * t_phi[0][1]
                    + a * (1. - b) * t_phi[1][0]
                    + a * b * t_phi[1][1];

                // position with respect to cell grid
                let xcell = x / h - 0.5;
                let ycell = y / h - 0.5;

                // indices of the nearest 4 cells
                //
                // +---(tio, tjo)
                // |           |
                // |           |
                // (ti, tj)----+
                let ti = (xcell.floor().max(0.) as usize).min(ni - 1);
                let tj = (ycell.floor().max(0.) as usize).min(nj - 1);

                let tio = (ti + 1).min(ni - 1);
                let tjo = (tj + 1).min(nj - 1);

                let a = xcell.fract();
                let b = ycell.fract();

                // distribute the sample mass among the 4 adjecent cells
                t_mu[(ti, tj)] += (1. - a) * (1. - b) * mass;
                t_mu[(tio, tj)] += a * (1. - b) * mass;
                t_mu[(ti, tjo)] += (1. - a) * b * mass;
                t_mu[(tio, tjo)] += a * b * mass;
            }
        }
    }
}

/// Push-forward T_#φ μ
///
/// Uses the idea in _Jacobs, Lee, Leger, 2021, The back-and-forth method for Wasserstein gradient
/// flows_
///
/// Requires φ to be c-concave (for example φ = \psi^c). Assumes that φ has zero normal derivative
/// on the domain boundary.
///
/// μ is given at the centers of cells.
///
/// We have (T_#φ)^{-1} = x - \nabla \phi and therefore
///
/// t_mu(x) = mu(x - \nabla \phi) det(I - D^2\phi)
pub fn push_forward2_ver2(
    mut t_mu: ArrayViewMut2<f64>,
    mu: ArrayView2<f64>,
    phi: ArrayView2<f64>,
    h: f64,
) {
    assert_eq!(t_mu.raw_dim(), mu.raw_dim());
    assert_eq!(t_mu.raw_dim(), phi.raw_dim());

    let (ni, nj) = phi.dim();

    // map each cell
    for ((i, j), t_mu) in t_mu.indexed_iter_mut() {
        // 9 neighboring cells
        let umm = phi[(i.saturating_sub(1), j.saturating_sub(1))];
        let um_ = phi[(i.saturating_sub(1), j)];
        let ump = phi[(i.saturating_sub(1), (j + 1).min(nj - 1))];
        let u_m = phi[(i, j.saturating_sub(1))];
        let u__ = phi[(i, j)];
        let u_p = phi[(i, (j + 1).min(nj - 1))];
        let upm = phi[((i + 1).min(ni - 1), j.saturating_sub(1))];
        let up_ = phi[((i + 1).min(ni - 1), j)];
        let upp = phi[((i + 1).min(ni - 1), (j + 1).min(nj - 1))];
        // \nabla\phi
        let dphix = (up_ - um_) / (2. * h);
        let dphiy = (u_p - u_m) / (2. * h);
        // det (I - D^2\phi)
        let det = (1. - (up_ - 2. * u__ + um_) / (h * h)) * (1. - (u_p - 2. * u__ + u_m) / (h * h))
            - ((upp + umm - ump - upm) / (4. * h * h)).powi(2);

        // x - \nabla\phi with respect to the cell grid
        let xcell = i as f64 - dphix / h;
        let ycell = j as f64 - dphiy / h;

        // indices of the nearest 4 cells
        //
        // +---(tio, tjo)
        // |           |
        // |           |
        // (ti, tj)----+
        let ti = (xcell.floor().max(0.) as usize).min(ni - 1);
        let tj = (ycell.floor().max(0.) as usize).min(nj - 1);

        let tio = (ti + 1).min(ni - 1);
        let tjo = (tj + 1).min(nj - 1);

        let a = xcell.fract();
        let b = ycell.fract();

        // interpolate the density value
        let mu_inter = mu[(ti, tj)] * (1. - a) * (1. - b)
            + mu[(tio, tj)] * a * (1. - b)
            + mu[(ti, tjo)] * (1. - a) * b
            + mu[(tio, tjo)] * a * b;

        *t_mu = mu_inter * det;
    }
}

pub fn det(mut det: ArrayViewMut2<f64>, phi: ArrayView2<f64>, h: f64) {
    assert_eq!(det.raw_dim(), phi.raw_dim());

    let (ni, nj) = phi.dim();

    // map each cell
    for ((i, j), det) in det.indexed_iter_mut() {
        // 9 neighboring cells
        let umm = phi[(i.saturating_sub(1), j.saturating_sub(1))];
        let um_ = phi[(i.saturating_sub(1), j)];
        let ump = phi[(i.saturating_sub(1), (j + 1).min(nj - 1))];
        let u_m = phi[(i, j.saturating_sub(1))];
        let u__ = phi[(i, j)];
        let u_p = phi[(i, (j + 1).min(nj - 1))];
        let upm = phi[((i + 1).min(ni - 1), j.saturating_sub(1))];
        let up_ = phi[((i + 1).min(ni - 1), j)];
        let upp = phi[((i + 1).min(ni - 1), (j + 1).min(nj - 1))];
        // det (I - D^2\phi)
        *det = (1. - (up_ - 2. * u__ + um_) / (h * h)) * (1. - (u_p - 2. * u__ + u_m) / (h * h))
            - ((upp + umm - ump - upm) / (4. * h * h)).powi(2);
        let dphix = (up_ - um_) / (2. * h);
        let dphiy = (u_p - u_m) / (2. * h);
        // *det = dphiy;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn convex_hull_test() {
        let mut p = Vec::new();
        convex_hull_1d(
            (&[0., 1., -2., 1., 1., 1.]).into(),
            (&[(); 6]).into(),
            &mut p,
        );
        assert_eq!(
            &p,
            &[
                ConvexHullPoint {
                    x: 0,
                    y: 0.,
                    s: f64::NEG_INFINITY,
                    data: ()
                },
                ConvexHullPoint {
                    x: 2,
                    y: -2.,
                    s: -1.,
                    data: ()
                },
                ConvexHullPoint {
                    x: 5,
                    y: 1.,
                    s: 1.,
                    data: ()
                },
            ]
        );
        p.clear();
        convex_hull_1d((&[0., 1.]).into(), (&[(); 2]).into(), &mut p);
        assert_eq!(
            &p,
            &[
                ConvexHullPoint {
                    x: 0,
                    y: 0.,
                    s: f64::NEG_INFINITY,
                    data: ()
                },
                ConvexHullPoint {
                    x: 1,
                    y: 1.,
                    s: 1.,
                    data: ()
                },
            ]
        );
        p.clear();
        convex_hull_1d((&[0.]).into(), (&[()]).into(), &mut p);
        assert_eq!(
            &p,
            &[ConvexHullPoint {
                x: 0,
                y: 0.,
                s: f64::NEG_INFINITY,
                data: ()
            }]
        );
    }

    #[test]
    fn legendre_fenchel_1d_test() {
        let mut phi_star = vec![0.; 5];
        let mut data: Vec<_> = (0..5).collect();

        legendre_fenchel((&mut phi_star).into(), (&mut data).into(), 1.);

        assert_eq!(phi_star, vec![0., 4., 8., 12., 16.]);
        assert_eq!(data, vec![0, 4, 4, 4, 4]);

        let mut phi_star = vec![0.; 5];
        let mut data: Vec<_> = (0..5).collect();

        legendre_fenchel((&mut phi_star).into(), (&mut data).into(), 0.5);

        assert_eq!(phi_star, vec![0., 1., 2., 3., 4.]);
        assert_eq!(data, vec![0, 4, 4, 4, 4]);

        let mut phi_star = (0..5).map(|i| (i as f64).powi(2) * 0.5).collect::<Vec<_>>();
        let phi = phi_star.clone();
        let mut data: Vec<_> = (0..5).collect();
        legendre_fenchel((&mut phi_star).into(), (&mut data).into(), 1.);

        assert_eq!(phi_star, phi);
    }

    #[test]
    fn legendre_fenchel_2d_test() {
        let mut phi_star = Array2::zeros((3, 3));
        let mut data: Vec<_> = (0..9).collect();

        legendre_fenchel(
            phi_star.view_mut(),
            aview_mut1(&mut data).into_shape((3, 3)).unwrap(),
            1.,
        );

        assert_eq!(
            phi_star,
            arr2(&[[0.0, 2.0, 4.0], [2.0, 4.0, 6.0], [4.0, 6.0, 8.0]])
        );

        // LF is invariant on |x|^2/2
        let n = 5;
        let mut phi_star = Array2::from_shape_fn((n, n), |(x, y)| 0.5 * (x * x + y * y) as f64);
        let phi = phi_star.clone();
        let mut data: Vec<_> = (0..n * n).collect();

        legendre_fenchel(
            phi_star.view_mut(),
            aview_mut1(&mut data).into_shape(phi.raw_dim()).unwrap(),
            1.,
        );

        assert_eq!(phi_star, phi);
    }

    #[test]
    fn legendre_fenchel_3d_test() {
        // LF is invariant on |x|^2/2
        let n = 4;
        let mut phi_star =
            Array3::from_shape_fn((n, n, n), |(x, y, z)| 0.5 * (x * x + y * y + z * z) as f64);
        let phi = phi_star.clone();
        let mut data: Vec<_> = (0..n * n * n).collect();

        legendre_fenchel(
            phi_star.view_mut(),
            aview_mut1(&mut data).into_shape(phi.raw_dim()).unwrap(),
            1.,
        );

        assert_eq!(phi_star, phi,);
    }

    #[test]
    fn push_forward2_test() {
        let phi_c: Array2<f64> = Array2::zeros((2, 2));
        let mu: Array2<f64> = Array2::ones(phi_c.raw_dim());
        let mut t_mu: Array2<f64> = Array2::zeros(mu.raw_dim());

        push_forward2(t_mu.view_mut(), mu.view(), phi_c.view(), 1.);

        assert_eq!(t_mu, mu);
    }
}
