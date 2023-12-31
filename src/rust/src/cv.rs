use crate::linalg::*;
use crate::regularise::*;
use ndarray::{prelude::*, Zip};
use rand::seq::SliceRandom;
use rand::{rngs::StdRng, SeedableRng};
use statrs::distribution::{ContinuousCDF, StudentsT};
use statrs::statistics::Statistics;
use std::io::{self, Error, ErrorKind};

/// Description ...
/// Takes a random number generator from the caller function to allow runs to be exactly repeatable in the same machine
fn k_split(
    row_idx: &Vec<usize>,
    mut k: usize,
    rng: &mut StdRng,
) -> io::Result<(Vec<usize>, usize, usize)> {
    let n = row_idx.len();
    if (k >= n) | (n <= 2) {
        return Err(Error::new(ErrorKind::Other, "The number of splits, i.e. k, needs to be less than the number of pools, n, and n > 2. We are aiming for fold sizes of 10 or greater."));
    }
    let mut s = (n as f64 / k as f64).floor() as usize;
    while s < 10 {
        if n < 20 {
            println!("Warning: number of pools is less than 20, so we're using k=2.");
            k = 2;
            s = (n as f64 / k as f64).floor() as usize;
            break;
        }
        k -= 1;
        s = (n as f64 / k as f64).floor() as usize;
    }
    let mut g = (0..k)
        .flat_map(|x| std::iter::repeat(x).take(s))
        .collect::<Vec<usize>>();
    if n - s > 0 {
        for _i in 0..(n - s) {
            g.push(k);
        }
    }
    let mut shuffle: Vec<usize> = row_idx.clone();
    shuffle.shuffle(rng);
    // println!("shuffle={:?}", shuffle);
    let mut out: Vec<usize> = Vec::new();
    for i in 0..n {
        out.push(g[shuffle[i]]);
    }
    Ok((out, k, s))
}

pub fn pearsons_correlation(x: ArrayView1<f64>, y: ArrayView1<f64>) -> io::Result<(f64, f64)> {
    let n = x.len();
    if n != y.len() {
        return Err(Error::new(
            ErrorKind::Other,
            "Input vectors are not the same size.",
        ));
    }
    let mu_x = x.mean();
    let mu_y = y.mean();
    let x_less_mu_x = x.map(|x| x - mu_x);
    let y_less_mu_y = y.map(|y| y - mu_y);
    let x_less_mu_x_squared = x_less_mu_x.map(|x| x.powf(2.0));
    let y_less_mu_y_squared = y_less_mu_y.map(|y| y.powf(2.0));
    let numerator = (x_less_mu_x * y_less_mu_y).sum();
    let denominator = x_less_mu_x_squared.sum().sqrt() * y_less_mu_y_squared.sum().sqrt();
    let r_tmp = numerator / denominator;
    let r = match r_tmp.is_nan() {
        true => 0.0,
        false => r_tmp,
    };
    let sigma_r_denominator = (1.0 - r.powf(2.0)) / (n as f64 - 2.0);
    if sigma_r_denominator <= 0.0 {
        // Essentially no variance in r2, hence very significant
        return Ok((r, f64::EPSILON));
    }
    let sigma_r = sigma_r_denominator.sqrt();
    let t = r / sigma_r;
    let d = StudentsT::new(0.0, 1.0, n as f64 - 2.0).unwrap();
    let pval = 2.00 * (1.00 - d.cdf(t.abs()));
    Ok((r, pval))
}

pub fn mean_bias_error(x: ArrayView1<f64>, y: ArrayView1<f64>) -> io::Result<f64> {
    let (n, m) = (x.len(), y.len());
    assert_eq!(n, m);
    let mbe = (&x - &y).sum() / (n as f64);
    Ok(mbe)
}

pub fn mean_absolute_error(x: ArrayView1<f64>, y: ArrayView1<f64>) -> io::Result<f64> {
    let (n, m) = (x.len(), y.len());
    assert_eq!(n, m);
    let mae = (&x - &y).iter().fold(0.0, |sum, &z| sum + z.abs()) / (n as f64);
    Ok(mae)
}

pub fn mean_squared_error(x: ArrayView1<f64>, y: ArrayView1<f64>) -> io::Result<f64> {
    let (n, m) = (x.len(), y.len());
    assert_eq!(n, m);
    let mse = (&x - &y).iter().fold(0.0, |sum, &z| sum + z.powf(2.0)) / (n as f64);
    Ok(mse)
}

pub fn root_mean_squared_error(x: ArrayView1<f64>, y: ArrayView1<f64>) -> io::Result<f64> {
    let (n, m) = (x.len(), y.len());
    assert_eq!(n, m);
    let mse = (&x - &y).iter().fold(0.0, |sum, &z| sum + z.powf(2.0)) / (n as f64);
    Ok(mse.sqrt())
}

/// Note: 1-cor, rmse, mae, and mse in combination or singly result in the same performance for some reason probably related to how we select for the best parameters below
fn error_index(
    b_hat: ArrayView1<f64>,
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    idx_validation: &Vec<usize>,
) -> io::Result<f64> {
    let (n, p) = (idx_validation.len(), x.ncols());
    let p_ = b_hat.len();
    if p != p_ {
        return Err(Error::new(
            ErrorKind::Other,
            "The X matrix is incompatible with b_hat.",
        ));
    }
    let idx_b_hat = &(0..p).collect();
    let y_true: Array1<f64> = Array1::from_shape_vec(
        n,
        idx_validation.iter().map(|&i| y[i]).collect::<Vec<f64>>(),
    )
    .unwrap();
    let y_pred: Array1<f64> = multiply_views_xx(
        x.view(),
        b_hat.to_owned().into_shape((p, 1)).unwrap().view(),
        idx_validation,
        idx_b_hat,
        idx_b_hat,
        &vec![0],
    )
    .unwrap()
    .column(0)
    .to_owned();
    let (cor, _pval) = pearsons_correlation(y_true.view(), y_pred.view()).unwrap();
    let _mbe = mean_bias_error(y_true.view(), y_pred.view()).unwrap();
    let mae = mean_absolute_error(y_true.view(), y_pred.view()).unwrap();
    let mse = mean_squared_error(y_true.view(), y_pred.view()).unwrap();
    let rmse = root_mean_squared_error(y_true.view(), y_pred.view()).unwrap();
    let error_index = ((1.0 - cor.abs()) + mae + mse + rmse) / 4.0;
    // let error_index = ((1.0 - cor.abs()) + mae + mse) / 3.0;
    // let error_index = ((1.0 - cor.abs()) + rmse) / 2.0;
    // let error_index = 1.0 - cor.abs();
    // let error_index = rmse;
    // let error_index = mae;
    // let error_index = mse;
    // let error_index = _mbe;
    // let error_index = 10.0;
    // let mut rng = StdRng::seed_from_u64(42069);
    // let dist_unif = statrs::distribution::Uniform::new(0.0, 1.0).unwrap();
    // let error_index = dist_unif.sample(&mut rng);
    Ok(error_index)
}

// NOTE: glmnet is blazinbgly fast because of its warm-start (probably starting with OLS estimates)
// resulting to quick convergence, i.e. only a single or a few coordinate-descent steps were needed.
// This means it's essentially a single step to find betas at a lambda-alpha parameter pair!
// What we can do better it to apply warm-start with the parameter-pair selection,
// e.g. start with large lambda and if the difference in performce between a lambda step is less than some threshold,
// then we stop and say that's the best lambda?!

// Also, ponder/test standardisation of X...

pub fn penalised_lambda_path_with_k_fold_cross_validation(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    row_idx: &Vec<usize>,
    alpha: f64,
    lambda_step_size: f64,
    r: usize,
) -> io::Result<(Array1<f64>, f64, f64)> {
    let (_n, _p) = (row_idx.len(), x.ncols());
    let max_usize: usize = (1.0 / lambda_step_size).round() as usize;

    let parameters_path_alpha: Array1<f64> = if alpha < 0.0 {
        (0..(max_usize + 1))
        .into_iter()
        .rev() // maybe moving this way we can be more blazingly fast?
        .map(|x| (x as f64) / (max_usize as f64))
        .collect()
    } else {
        Array1::from_elem(1, alpha)
    };
    
    let parameters_path_lambda: Array1<f64> = (0..(max_usize + 1))
        .into_iter()
        .rev() // maybe moving this way we can be more blazingly fast?
        .map(|x| (x as f64) / (max_usize as f64))
        .collect();
    let a = parameters_path_alpha.len();
    let l = parameters_path_lambda.len();
    // If alpha < 0.0 then optimise for alpha in addtion to lambda, else use the user-secified alpha for ridge, lasso or somewhere in between
    let alpha_path: Array2<f64> = Array2::from_shape_vec(
                (a, l),
                parameters_path_alpha
                    .clone()
                    .iter()
                    .flat_map(|&x| std::iter::repeat(x).take(l))
                    .collect(),
            )
            .unwrap();
    let lambda_path: Array2<f64> = Array2::from_shape_vec(
        (a, l),
        std::iter::repeat(parameters_path_lambda.clone())
            .take(a)
            .flat_map(|x| x)
            .collect(),
    )
    .unwrap();

    println!("parameters_path_alpha={:?}", parameters_path_alpha);
    println!("parameters_path_lambda={:?}", parameters_path_lambda);

    println!("alpha_path={:?}", alpha_path);
    println!("lambda_path={:?}", lambda_path);

    // Using a random number generator from the caller function to allow runs to be exactly repeatable in the same machine
    // This enables the optimisation for alpha to work predictable well and better than either ridge-like or lasso-like, i.e. alpha=0 and alpha=1 when q is between monogenic and polygenic
    let mut rng = StdRng::seed_from_u64(0);
    let (_, nfolds, _s) = k_split(row_idx, 10, &mut rng).unwrap();
    let mut performances: Array4<f64> = Array4::from_elem((r, nfolds, a, l), f64::NAN);
    for rep in 0..r {
        // let mut rng = StdRng::seed_from_u64(rep as u64);
        let (groupings, _, _) = k_split(row_idx, 10, &mut rng).unwrap();
        // println!("groupings={:?}", groupings);
        for fold in 0..nfolds {
            let idx_validation: Vec<usize> = groupings
                .iter()
                .enumerate()
                .filter(|(_, x)| *x == &fold)
                .map(|(i, _)| row_idx[i])
                .collect();
            let idx_training: Vec<usize> = groupings
                .iter()
                .enumerate()
                .filter(|(_, x)| *x != &fold)
                .map(|(i, _)| row_idx[i])
                .collect();
            let b_hat = ols(
                x.view(),
                y.view(),
                &idx_training,
                &(0..x.ncols()).collect::<Vec<usize>>(),
            )
            .unwrap();
            let mut errors: Array2<f64> = Array2::from_elem((a, l), f64::NAN);
            let mut b_hats: Array2<Array1<f64>> =
                Array2::from_elem((a, l), Array1::from_elem(1, f64::NAN));
            Zip::from(&mut errors)
                .and(&mut b_hats)
                .and(&alpha_path)
                .and(&lambda_path)
                .par_for_each(|err, b, &alfa, &lambda| {
                    let b_hat_new: Array1<f64> = expand_and_contract(&b_hat, alfa, lambda).unwrap();
                    *err = error_index(b_hat_new.view(), x, y, &idx_validation).unwrap();
                    *b = b_hat_new;
                });
            // Append performances, i.e. error index: f(1-cor, rmse, mae, etc...)
            for i in 0..a {
                for j in 0..l {
                    performances[(rep, fold, i, j)] = errors[(i, j)];
                }
            }
        }
    }
    // Account for overfit cross-validation folds, i.e. filter them out, or just use mode of the lambda and alphas?
    let mut alpha_path_counts: Array1<usize> = Array1::from_elem(l, 0);
    let mut lambda_path_counts: Array1<usize> = Array1::from_elem(l, 0);
    for rep in 0..r {
        let mean_error_per_rep_across_folds: Array2<f64> = performances
            .slice(s![rep, .., .., ..])
            .mean_axis(Axis(0))
            .unwrap();
        let min_error = mean_error_per_rep_across_folds.fold(
            mean_error_per_rep_across_folds[(0, 0)],
            |min, &x| {
                if x < min {
                    x
                } else {
                    min
                }
            },
        );
        let ((idx_0, idx_1), _) = mean_error_per_rep_across_folds
            .indexed_iter()
            .find(|((_i, _j), &x)| x == min_error)
            .unwrap();
        for i in 0..a {
            if alpha_path[(idx_0, idx_1)] == parameters_path_alpha[i] {
                alpha_path_counts[i] += 1;
            }
        }
        for i in 0..l {
            if lambda_path[(idx_0, idx_1)] == parameters_path_lambda[i] {
                lambda_path_counts[i] += 1;
            }
        }
    }
    // Find the mode alpha and lambda
    let alpha_max_count = alpha_path_counts.fold(0, |max, &x| if x > max { x } else { max });
    let (alpha_idx, _) = alpha_path_counts
        .indexed_iter()
        .find(|(_a, &x)| x == alpha_max_count)
        .unwrap();
    let lambda_max_count = lambda_path_counts.fold(0, |max, &x| if x > max { x } else { max });
    let (lambda_idx, _) = lambda_path_counts
        .indexed_iter()
        .find(|(_a, &x)| x == lambda_max_count)
        .unwrap();
    let alpha = parameters_path_alpha[alpha_idx];
    let lambda = parameters_path_lambda[lambda_idx];
    ///////////////////////////////////
    // // Equivalent to just simply finding the minimum - for n=100; p=1000; and q={2, 10}
    // let mean_error_per_rep_across_folds = performances
    //     .mean_axis(Axis(0))
    //     .unwrap()
    //     .mean_axis(Axis(0))
    //     .unwrap();
    // let min_error = mean_error_per_rep_across_folds.view().min();
    // let ((i, j), _) = mean_error_per_rep_across_folds
    //             .indexed_iter()
    //             .find(|((_i, _j), &x)| x == min_error)
    //             .unwrap();
    // let alpha = alpha_path[(i,j)];
    // let lambda = lambda_path[(i,j)];
    // ///////////////////////////////////
    let b_hat: Array1<f64> = ols(x, y, row_idx, &(0..x.ncols()).collect::<Vec<usize>>()).unwrap();
    let b_hat_penalised: Array1<f64> = expand_and_contract(&b_hat, alpha, lambda).unwrap();
    Ok((b_hat_penalised, alpha, lambda))
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;
    #[test]
    fn test_cv() {
        // Inputs
        let row_idx: Vec<usize> = (0..10).collect();
        let k = 2;
        // Outputs
        let mut rng = StdRng::seed_from_u64(0);
        let (idx, k, s) = k_split(&row_idx, k, &mut rng).unwrap();
        // Assertions
        assert_eq!(idx.iter().fold(0, |sum, &x| sum + x), s);
        assert_eq!(k, 2);
        assert_eq!(s, 5);
        // assert_eq!(0, 1);
    }
}
