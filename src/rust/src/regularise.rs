use ndarray::prelude::*;
use statrs::statistics::Statistics;
use std::io;

pub fn varsel(
    b_hat_proxy: ArrayView1<f64>,
    alpha: f64,
    lambda: f64,
) -> io::Result<Vec<usize>> {
    // Clone b_hat
    let p = b_hat_proxy.len();
    // Proxy norm 1 or norm 2 (exclude the intercept) for finding the loci that need to be penalised
    let normed1_proxy: Array1<f64> = b_hat_proxy.slice(s![1..p]).map(|&x| x.abs());
    let normed2_proxy = b_hat_proxy.slice(s![1..p]).map(|&x| x.powf(2.0));
    let normed_proxy = ((1.00 - alpha) * normed2_proxy / 1.00) + (alpha * normed1_proxy);
    // Find estimates that will be penalised using the proxy b_hat norms
    let normed_proxy_min = normed_proxy.view().min();
    let normed_proxy_max = normed_proxy.view().max();
    let normed_proxy_scaled: Array1<f64> = (&normed_proxy - normed_proxy_min) / (normed_proxy_max - normed_proxy_min);
    // let idx_depenalised = normed_proxy_scaled
    //     .iter()
    //     .enumerate()
    //     .filter(|(_, &value)| value >= lambda)
    //     .map(|(index, _)| index + 1)
    //     .collect::<Vec<usize>>();
    let mut idx_depenalised: Vec<usize> = vec![0];
    for i in 0..normed_proxy_scaled.len() {
        if normed_proxy_scaled[i] >= lambda {
            idx_depenalised.push(i + 1);
        }
    }
    Ok(idx_depenalised)
}

pub fn expand_and_contract(
    b_hat: &Array1<f64>,
    b_hat_proxy: &Array1<f64>,
    alpha: f64,
    lambda: f64,
) -> io::Result<Array1<f64>> {
    // Clone b_hat
    let mut b_hat: Array1<f64> = b_hat.clone();
    let p = b_hat.len();
    // Exclude the intercept from penalisation
    let intercept = b_hat[0];
    // Norm 1 or norm 2 (exclude the intercept)
    let normed1: Array1<f64> = b_hat.slice(s![1..p]).map(|&x| x.abs());
    let normed2 = b_hat.slice(s![1..p]).map(|&x| x.powf(2.0));
    let normed = ((1.00 - alpha) * normed2 / 1.00) + (alpha * normed1);
    // Proxy norm 1 or norm 2 (exclude the intercept) for finding the loci that need to be penalised
    let normed1_proxy: Array1<f64> = b_hat_proxy.slice(s![1..p]).map(|&x| x.abs());
    let normed2_proxy = b_hat_proxy.slice(s![1..p]).map(|&x| x.powf(2.0));
    let normed_proxy = ((1.00 - alpha) * normed2_proxy / 1.00) + (alpha * normed1_proxy);
    // Find estimates that will be penalised using the proxy b_hat norms
    let normed_proxy_min = normed_proxy.view().min();
    let normed_proxy_max = normed_proxy.view().max();
    let normed_proxy_scaled: Array1<f64> = (&normed_proxy - normed_proxy_min) / (normed_proxy_max - normed_proxy_min);
    // let normed_proxy_scaled: Array1<f64> = (&normed_proxy - 0.0) / (normed_proxy_max - 0.0);
    let idx_penalised = normed_proxy_scaled
        .iter()
        .enumerate()
        .filter(|(_, &value)| value < lambda)
        .map(|(index, _)| index)
        .collect::<Vec<usize>>();
    let idx_depenalised = normed_proxy_scaled
        .iter()
        .enumerate()
        .filter(|(_, &value)| value >= lambda)
        .map(|(index, _)| index)
        .collect::<Vec<usize>>();
    // Penalise: contract using the non-proxy b_hat norms
    let mut subtracted_penalised = 0.0;
    let mut added_penalised = 0.0;
    for i in idx_penalised.into_iter() {
        if b_hat[i + 1] >= 0.0 {
            subtracted_penalised += normed[i];
        } else {
            added_penalised += normed[i];
        }
        b_hat[i + 1] = b_hat[i + 1].signum() * vec![(b_hat[i + 1].abs() - normed[i]), 0.0].max();
    }
    // Find total depenalised (expanded) values
    let mut subtracted_depenalised = 0.0;
    let mut added_depenalised = 0.0;
    for i in idx_depenalised.clone().into_iter() {
        if b_hat[i + 1] >= 0.0 {
            subtracted_depenalised += normed[i];
        } else {
            added_depenalised += normed[i];
        }
    }
    // println!("BEFORE: @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@");
    // println!("PENALISED:");
    // println!("subtracted_penalised={}", subtracted_penalised);
    // println!("added_penalised={}", added_penalised);
    // println!("DEPENALISED:");
    // println!("subtracted_depenalised={}", subtracted_depenalised);
    // println!("added_depenalised={}", added_depenalised);
    // Account for the absence of available slots to transfer the contracted effects into
    if (subtracted_penalised > 0.0) & (subtracted_depenalised == 0.0) {
        added_penalised -= subtracted_penalised;
        subtracted_penalised = 0.0;
    } else if (added_penalised > 0.0) & (added_depenalised == 0.0) {
        subtracted_penalised -= added_penalised;
        added_penalised = 0.0;
    }
    // if (subtracted_penalised < 0.0) | (added_penalised < 0.0) {
    //     println!("AFTER: ###############################################");
    //     println!("PENALISED:");
    //     println!("subtracted_penalised={}", subtracted_penalised);
    //     println!("added_penalised={}", added_penalised);
    //     println!("DEPENALISED:");
    //     println!("subtracted_depenalised={}", subtracted_depenalised);
    //     println!("added_depenalised={}", added_depenalised);
    // }
    // if subtracted_penalised < 0.0 {
    //     subtracted_penalised = 0.0;
    // }
    // if added_penalised < 0.0 {
    //     added_penalised = 0.0;
    // }
    
    for i in idx_depenalised.into_iter() {
        // if b_hat[i + 1] >= 0.0 {
        //     b_hat[i + 1] = b_hat[i + 1].signum() * (b_hat[i + 1].abs() + subtracted_penalised * (normed[i] / subtracted_depenalised));
        //     // b_hat[i + 1] += subtracted_penalised * (normed[i] / subtracted_depenalised);
        //     // b_hat[i + 1] += subtracted_penalised.abs() * (normed[i] / subtracted_depenalised);
        // } else {
        //     b_hat[i + 1] = b_hat[i + 1].signum() * (b_hat[i + 1].abs() + added_penalised * (normed[i] / added_depenalised));
        //     // b_hat[i + 1] -= added_penalised * (normed[i] / added_depenalised);
        //     // b_hat[i + 1] -= added_penalised.abs() * (normed[i] / added_depenalised);
        // }
        // subtracted_penalised = subtracted_penalised.abs();
        // added_penalised = added_penalised.abs();
        let q = (subtracted_penalised - added_penalised) * (normed[i] / (subtracted_depenalised + added_depenalised));
        // let q = total_penalised * (normed[i] / (subtracted_depenalised + added_depenalised));
        b_hat[i + 1] = b_hat[i + 1].signum() * (b_hat[i + 1].abs() + q);
    }
    // Insert the unpenalised intercept
    b_hat[0] = intercept;
    Ok(b_hat)
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_penalise() {
        let a: Array1<f64> =
            Array1::from_shape_vec(7, vec![5.0, -0.4, 0.0, 1.0, -0.1, 1.0, 0.0]).unwrap();
        assert_eq!(
            expand_and_contract(&a, &a, 1.00, 0.5).unwrap(),
            Array1::from_shape_vec(7, vec![5.0, 0.0, 0.0, 0.75, 0.0, 0.75, 0.0]).unwrap()
        );
        let b: Array1<f64> =
            Array1::from_shape_vec(7, vec![5.0, 0.4, 0.0, -1.0, 0.1, -1.0, 0.0]).unwrap();
        assert_eq!(
            expand_and_contract(&b, &b, 1.00, 0.5).unwrap(),
            Array1::from_shape_vec(7, vec![5.0, 0.0, 0.0, -0.75, 0.0, -0.75, 0.0]).unwrap()
        );
        let c: Array1<f64> =
            Array1::from_shape_vec(7, vec![5.0, 0.0, 0.0, 1.0, -0.4, 1.0, 0.0]).unwrap();
        assert_eq!(
            expand_and_contract(&c, &c, 1.00, 0.5).unwrap(),
            Array1::from_shape_vec(7, vec![5.0, 0.0, 0.0, 0.8, 0.0, 0.8, 0.0]).unwrap()
        );
        // let d: Array1<f64> =
        //     Array1::from_shape_vec(7, vec![5.0, 0.4, 1.0, -1.0, -0.4, -1.0, 1.0]).unwrap();
        // assert_eq!(
        //     expand_and_contract(&d, &d, 1.00, 0.5).unwrap(),
        //     Array1::from_shape_vec(7, vec![5.0, 0.0, 1.2, -1.2, 0.0, -1.2, 1.2]).unwrap()
        // );
        let a: Array1<f64> =
            Array1::from_shape_vec(7, vec![5.0, -0.5, 0.0, 1.0, -0.4, 1.0, 0.0]).unwrap();
        assert_eq!(
            expand_and_contract(&a, &a, 1.0, 0.5).unwrap(),
            Array1::from_shape_vec(7, vec![5.0, -0.58, 0.0, 0.84, 0.0, 0.84, 0.0]).unwrap()
        );
    }
}
