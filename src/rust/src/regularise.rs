use ndarray::{prelude::*, Zip};
use statrs::statistics::Statistics;
use std::io;

pub fn expand_and_contract(
    b_hat: &Array1<f64>,
    alpha: f64,
    lambda: f64,
) -> io::Result<Array1<f64>> {
    let intercept = b_hat[0];
    let p = b_hat.len();
    // Norm 1 or norm 2 (exclude the intercept)
    let mut normed1: Array1<f64> = Array1::from_elem(p-1, f64::NAN);
    let mut normed2: Array1<f64> = Array1::from_elem(p-1, f64::NAN);
    let mut normed: Array1<f64> = Array1::from_elem(p-1, f64::NAN);
    let mut b_hat_new: Array1<f64> = Array1::from_elem(p-1, f64::NAN); // excluding the intercept
    Zip::from(&mut normed1)
        .and(&mut normed2)
        .and(&mut normed)
        .and(&mut b_hat_new)
        .and(&b_hat.slice(s![1..p]))
        .par_for_each(|n1, n2, n, b_new, &b| {
            *n1 = b.abs();
            *n2 = b.powf(2.0);
            *n = ((1.00 - alpha) * *n2) + (alpha * *n1);
            *b_new = b;
        });
    // Find estimates that will be penalised using the proxy b_hat norms
    let normed_min = normed.view().min();
    let normed_max = normed.view().max();
    let mut subtracted_penalised: Array1<f64> = Array1::from_elem(p-1, 0.0);
    let mut added_penalised: Array1<f64> = Array1::from_elem(p-1, 0.0);
    let mut subtracted_depenalised: Array1<f64> = Array1::from_elem(p-1, 0.0);
    let mut added_depenalised: Array1<f64> = Array1::from_elem(p-1, 0.0);
    Zip::from(&mut b_hat_new)
        .and(&mut subtracted_penalised)
        .and(&mut added_penalised)
        .and(&mut subtracted_depenalised)
        .and(&mut added_depenalised)
        .and(&normed)
        .par_for_each(|b, sp, ap, sd, ad, &n| {
            let s = (n - normed_min) / (normed_max - normed_min);
            if s < lambda {
                if *b >= 0.0 {
                    *sp = n;
                } else {
                    *ap = n;
                }
                *b = b.signum() * vec![(b.abs() - n), 0.0].max();
            } else {
                if *b >= 0.0 {
                    *sd = n;
                } else {
                    *ad = n;
                }
            }
        });

    let mut subpen = subtracted_penalised.sum();
    let mut adpen = added_penalised.sum();
    let sudep = subtracted_depenalised.sum();
    let addep = added_depenalised.sum();

    // Account for the absence of available slots to transfer the contracted effects into
    if (subpen > 0.0) & (sudep == 0.0) {
        adpen -= subpen;
        subpen = 0.0;
    } else if (adpen > 0.0) & (addep == 0.0) {
        subpen -= adpen;
        adpen = 0.0;
    }

    Zip::from(&mut b_hat_new)
        .and(&normed)
        .and(&subtracted_depenalised)
        .and(&added_depenalised)
        .par_for_each(|b, &n, &sd, &ad| {
            let q = if (sd != 0.0) | (ad != 0.0) {
                if *b >= 0.0 {
                    subpen * n / sudep
                } else {
                    -adpen * n / addep
                }
            } else {
                0.0
            };
            *b += q;
        });
    let mut out: Array1<f64> = Array1::from_elem(p, f64::NAN);
        out[0] = intercept;
    for i in 1..p {
        out[i] = b_hat_new[i-1];
    }
    
    Ok(out)
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_penalise() {
        let x: Array1<f64> =
            Array1::from_shape_vec(7, vec![5.0, 0.4, 0.0, 1.0, 0.1, 1.0, 0.0]).unwrap();
        assert_eq!(
            expand_and_contract(&x, 1.00, 0.5).unwrap(),
            Array1::from_shape_vec(7, vec![5.0, 0.0, 0.0, 1.25, 0.0, 1.25, 0.0]).unwrap()
        );
        let x: Array1<f64> =
            Array1::from_shape_vec(7, vec![5.0, -0.4, 0.0, -1.0, -0.1, -1.0, 0.0]).unwrap();
        assert_eq!(
            expand_and_contract(&x, 1.00, 0.5).unwrap(),
            Array1::from_shape_vec(7, vec![5.0, 0.0, 0.0, -1.25, 0.0, -1.25, 0.0]).unwrap()
        );
        let x: Array1<f64> =
            Array1::from_shape_vec(7, vec![5.0, -0.4, 0.0, 1.0, -0.1, 1.0, 0.0]).unwrap();
        assert_eq!(
            expand_and_contract(&x, 1.00, 0.5).unwrap(),
            Array1::from_shape_vec(7, vec![5.0, 0.0, 0.0, 0.75, 0.0, 0.75, 0.0]).unwrap()
        );
        let x: Array1<f64> =
            Array1::from_shape_vec(7, vec![5.0, 0.4, 0.0, -1.0, 0.1, -1.0, 0.0]).unwrap();
        assert_eq!(
            expand_and_contract(&x, 1.00, 0.5).unwrap(),
            Array1::from_shape_vec(7, vec![5.0, 0.0, 0.0, -0.75, 0.0, -0.75, 0.0]).unwrap()
        );
        let x: Array1<f64> =
            Array1::from_shape_vec(7, vec![5.0, 0.4, 0.0, 1.0, -0.1, -1.0, 0.0]).unwrap();
        assert_eq!(
            expand_and_contract(&x, 1.00, 0.5).unwrap(),
            Array1::from_shape_vec(7, vec![5.0, 0.0, 0.0, 1.4, 0.0, -1.10, 0.0]).unwrap()
        );
    }
}
