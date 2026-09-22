//! Property types combined by the expansion.

use std::sync::Arc;

/// A cluster property that the inclusion–exclusion step can combine: it only
/// needs to form linear combinations. Implementors are extensive quantities
/// of a cluster (arrays over temperature now, over time later).
pub trait Property: Clone + Send + Sync {
    /// A zero value with the same shape as `self`.
    fn zeros_like(&self) -> Self;

    /// `self += a · x`.
    ///
    /// # Panics
    /// If `x` does not have the same shape as `self`.
    fn axpy(&mut self, a: f64, x: &Self);

    /// Largest absolute component-wise difference (for tests and
    /// convergence diagnostics).
    ///
    /// # Panics
    /// If the shapes differ.
    fn max_abs_diff(&self, other: &Self) -> f64;
}

impl Property for Vec<f64> {
    fn zeros_like(&self) -> Self {
        vec![0.0; self.len()]
    }

    fn axpy(&mut self, a: f64, x: &Self) {
        assert_eq!(self.len(), x.len(), "property length mismatch");
        for (s, v) in self.iter_mut().zip(x) {
            *s += a * v;
        }
    }

    fn max_abs_diff(&self, other: &Self) -> f64 {
        assert_eq!(self.len(), other.len(), "property length mismatch");
        self.iter()
            .zip(other)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max)
    }
}

/// Finite-temperature thermodynamics of a cluster, each array over the same
/// temperature grid. All quantities are extensive (not per site), with
/// `k_B = 1`.
#[derive(Clone, Debug, PartialEq)]
pub struct Thermo {
    /// Temperature grid (shared, all entries `> 0`).
    pub temps: Arc<[f64]>,
    /// `ln Z`.
    pub ln_z: Vec<f64>,
    /// Energy `⟨H⟩`.
    pub energy: Vec<f64>,
    /// Entropy `S = ln Z + ⟨H⟩ / T`.
    pub entropy: Vec<f64>,
    /// Specific heat `C = (⟨H²⟩ − ⟨H⟩²) / T²`.
    pub specific_heat: Vec<f64>,
    /// Magnetisation `⟨S^z_tot⟩`, if requested.
    pub magnetization: Option<Vec<f64>>,
    /// Uniform susceptibility `(⟨(S^z_tot)²⟩ − ⟨S^z_tot⟩²) / T`, if requested.
    pub susceptibility: Option<Vec<f64>>,
}

impl Thermo {
    fn fields_mut(&mut self) -> impl Iterator<Item = &mut Vec<f64>> {
        [
            Some(&mut self.ln_z),
            Some(&mut self.energy),
            Some(&mut self.entropy),
            Some(&mut self.specific_heat),
            self.magnetization.as_mut(),
            self.susceptibility.as_mut(),
        ]
        .into_iter()
        .flatten()
    }

    fn fields(&self) -> impl Iterator<Item = &Vec<f64>> {
        [
            Some(&self.ln_z),
            Some(&self.energy),
            Some(&self.entropy),
            Some(&self.specific_heat),
            self.magnetization.as_ref(),
            self.susceptibility.as_ref(),
        ]
        .into_iter()
        .flatten()
    }

    fn assert_same_shape(&self, other: &Self) {
        assert!(
            Arc::ptr_eq(&self.temps, &other.temps) || self.temps == other.temps,
            "temperature grids differ"
        );
        assert_eq!(
            self.magnetization.is_some(),
            other.magnetization.is_some(),
            "magnetization present in one property but not the other"
        );
        assert_eq!(
            self.susceptibility.is_some(),
            other.susceptibility.is_some(),
            "susceptibility present in one property but not the other"
        );
    }
}

impl Property for Thermo {
    fn zeros_like(&self) -> Self {
        let z = || vec![0.0; self.temps.len()];
        Self {
            temps: self.temps.clone(),
            ln_z: z(),
            energy: z(),
            entropy: z(),
            specific_heat: z(),
            magnetization: self.magnetization.as_ref().map(|_| z()),
            susceptibility: self.susceptibility.as_ref().map(|_| z()),
        }
    }

    fn axpy(&mut self, a: f64, x: &Self) {
        self.assert_same_shape(x);
        for (s, v) in self.fields_mut().zip(x.fields()) {
            s.axpy(a, v);
        }
    }

    fn max_abs_diff(&self, other: &Self) -> f64 {
        self.assert_same_shape(other);
        self.fields()
            .zip(other.fields())
            .map(|(a, b)| a.max_abs_diff(b))
            .fold(0.0, f64::max)
    }
}
