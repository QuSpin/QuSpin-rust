//! `GenericBasis` — single owner of the [`SpaceInner`] dispatch.
//!
//! This is the type that knows how to construct the four-family
//! [`SpaceInner`] hierarchy from a `(n_sites, lhss, space_kind,
//! fermionic)` tuple, route symmetry-element insertions into the
//! correct family, and run BFS-build with family-aware seed decoding.
//!
//! [`SpinBasis`](crate::SpinBasis), [`BosonBasis`](crate::BosonBasis),
//! and [`FermionBasis`](crate::FermionBasis) all wrap a `GenericBasis`
//! and delegate the dispatch logic here, only adding their own
//! conventions on top (e.g. `FermionBasis` enforcing `fermionic =
//! true, lhss = 2` at construction; `SpinBasis::add_inv` choosing
//! between XOR bit-flip and value-inversion based on LHSS).

use crate::dispatch::{
    SpaceInner, SpaceInnerBit, SpaceInnerBitDefault, SpaceInnerDit, SpaceInnerDitDefault,
    SpaceInnerQuat, SpaceInnerQuatDefault, SpaceInnerTrit, SpaceInnerTritDefault,
};
#[cfg(feature = "large-int")]
use crate::dispatch::{
    SpaceInnerBitLargeInt, SpaceInnerDitLargeInt, SpaceInnerQuatLargeInt, SpaceInnerTritLargeInt,
};
use crate::space::{FullSpace, Subspace};
use crate::spin::SpaceKind;
use crate::sym::SymBasis;
use num_complex::Complex;
use quspin_bitbasis::{DynamicPermDitValues, PermDitMask, PermDitValues, StateTransitions};
use quspin_types::QuSpinError;

// ---------------------------------------------------------------------------
// GenericBasis
// ---------------------------------------------------------------------------

/// Single user-facing basis type that owns the [`SpaceInner`] dispatch
/// hierarchy. Every other basis wrapper in this crate is just a
/// trampoline through `GenericBasis`.
pub struct GenericBasis {
    pub inner: SpaceInner,
}

impl GenericBasis {
    /// Construct a new generic basis.
    ///
    /// `fermionic = true` is only meaningful for `lhss == 2`; passing
    /// `true` with a different `lhss` returns an error.
    ///
    /// # Errors
    /// - `lhss < 2`
    /// - `fermionic == true && lhss != 2`
    /// - [`SpaceKind::Full`] with more than 64 bits required
    /// - [`SpaceKind::Sub`] / [`SpaceKind::Symm`] with more than 8192 bits
    pub fn new(
        n_sites: usize,
        lhss: usize,
        space_kind: SpaceKind,
        fermionic: bool,
    ) -> Result<Self, QuSpinError> {
        if lhss < 2 {
            return Err(QuSpinError::ValueError(format!(
                "lhss must be >= 2, got {lhss}"
            )));
        }
        if fermionic && lhss != 2 {
            return Err(QuSpinError::ValueError(format!(
                "fermionic=true requires lhss=2, got lhss={lhss}"
            )));
        }
        let bits_per_dit = if lhss <= 2 {
            1
        } else {
            (usize::BITS - (lhss - 1).leading_zeros()) as usize
        };
        let n_bits = n_sites * bits_per_dit;
        let inner = build_space_inner(n_sites, lhss, space_kind, fermionic, n_bits)?;
        Ok(Self { inner })
    }

    /// The [`SpaceKind`] this basis was constructed with.
    #[inline]
    pub fn space_kind(&self) -> SpaceKind {
        self.inner.space_kind()
    }

    // -- Introspection (delegate to SpaceInner) ----------------------------

    /// Number of lattice sites.
    #[inline]
    pub fn n_sites(&self) -> usize {
        self.inner.n_sites()
    }

    /// Local Hilbert-space size (LHSS).
    #[inline]
    pub fn lhss(&self) -> usize {
        self.inner.lhss()
    }

    /// Number of basis states.
    #[inline]
    pub fn size(&self) -> usize {
        self.inner.size()
    }

    /// `true` once the basis has been built (or always for `Full*`).
    #[inline]
    pub fn is_built(&self) -> bool {
        self.inner.is_built()
    }

    /// One of `"full"`, `"subspace"`, or `"symmetric"`.
    #[inline]
    pub fn kind(&self) -> &'static str {
        self.inner.kind()
    }

    /// `true` for `Sym*` variants (symmetry-reduced subspaces).
    #[inline]
    pub fn is_symmetric(&self) -> bool {
        self.inner.is_symmetric()
    }

    /// Look up the index of a basis state given as a per-site
    /// occupation byte slice.
    #[inline]
    pub fn index_of_bytes(&self, bytes: &[u8]) -> Option<usize> {
        self.inner.index_of_bytes(bytes)
    }

    /// `i`-th basis state formatted as a bit / dit string.
    #[inline]
    pub fn state_at_str(&self, i: usize) -> String {
        self.inner.state_at_str(i)
    }

    /// `i`-th basis state formatted as a decimal integer string.
    #[inline]
    pub fn state_at_decimal_str(&self, i: usize) -> String {
        self.inner.state_at_decimal_str(i)
    }

    // -- Symmetry insertion ------------------------------------------------

    /// Add a lattice (site-permutation) symmetry element.
    pub fn add_lattice(
        &mut self,
        grp_char: Complex<f64>,
        perm: Vec<usize>,
    ) -> Result<(), QuSpinError> {
        validate_perm(&perm, self.inner.n_sites())?;
        self.assert_can_add_symmetry()?;
        self.inner.add_lattice(grp_char, &perm)
    }

    /// Add an inversion (XOR bit-flip) symmetry element. LHSS = 2 only.
    pub fn add_inv(&mut self, grp_char: Complex<f64>, locs: Vec<usize>) -> Result<(), QuSpinError> {
        if self.inner.lhss() != 2 {
            return Err(QuSpinError::ValueError(format!(
                "add_inv requires lhss=2, got lhss={}",
                self.inner.lhss(),
            )));
        }
        validate_locs(&locs, self.inner.n_sites())?;
        self.assert_can_add_symmetry()?;
        self.inner.add_inv(grp_char, &locs)
    }

    /// Add a local dit-permutation symmetry element.
    pub fn add_local(
        &mut self,
        grp_char: Complex<f64>,
        perm_vals: Vec<u8>,
        locs: Vec<usize>,
    ) -> Result<(), QuSpinError> {
        validate_perm_vals(&perm_vals, self.inner.lhss())?;
        validate_locs(&locs, self.inner.n_sites())?;
        self.assert_can_add_symmetry()?;
        self.inner.add_local(grp_char, perm_vals, locs)
    }

    // -- Build -------------------------------------------------------------

    /// Build the basis subspace from `seeds` under the connectivity
    /// described by `graph`.
    ///
    /// # Errors
    /// - Called on a [`SpaceKind::Full`] basis
    /// - Basis is already built
    /// - `graph.lhss() != self.lhss()`
    pub fn build<G: StateTransitions>(
        &mut self,
        graph: &G,
        seeds: &[Vec<u8>],
    ) -> Result<(), QuSpinError> {
        if self.space_kind() == SpaceKind::Full {
            return Err(QuSpinError::ValueError(
                "Full basis requires no build step".into(),
            ));
        }
        if self.inner.is_built() {
            return Err(QuSpinError::ValueError("basis is already built".into()));
        }
        if graph.lhss() != self.inner.lhss() {
            return Err(QuSpinError::ValueError(format!(
                "graph.lhss()={} does not match basis lhss={}",
                graph.lhss(),
                self.inner.lhss()
            )));
        }
        self.inner.build_seeds(graph, seeds)
    }

    // -- internal ---------------------------------------------------------

    fn assert_can_add_symmetry(&self) -> Result<(), QuSpinError> {
        if !self.inner.is_symmetric() {
            return Err(QuSpinError::ValueError(
                "symmetry elements require a symmetry-reduced (Symm) basis".into(),
            ));
        }
        if self.inner.is_built() {
            return Err(QuSpinError::ValueError(
                "cannot add symmetry elements after basis is built".into(),
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Display — delegates to the inner SpaceInner's existing impl.
// ---------------------------------------------------------------------------

impl std::fmt::Display for GenericBasis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.inner, f)
    }
}

// ---------------------------------------------------------------------------
// Argument validators (shared by add_lattice / add_inv / add_local)
// ---------------------------------------------------------------------------

fn validate_perm(perm: &[usize], n_sites: usize) -> Result<(), QuSpinError> {
    if perm.len() != n_sites {
        return Err(QuSpinError::ValueError(format!(
            "perm.len()={} but n_sites={n_sites}",
            perm.len()
        )));
    }
    let mut seen = vec![false; n_sites];
    for (i, &p) in perm.iter().enumerate() {
        if p >= n_sites {
            return Err(QuSpinError::ValueError(format!(
                "perm[{i}]={p} is out of range 0..{n_sites}"
            )));
        }
        if seen[p] {
            return Err(QuSpinError::ValueError(format!(
                "perm has duplicate target site {p}"
            )));
        }
        seen[p] = true;
    }
    Ok(())
}

fn validate_perm_vals(perm_vals: &[u8], lhss: usize) -> Result<(), QuSpinError> {
    if perm_vals.len() != lhss {
        return Err(QuSpinError::ValueError(format!(
            "perm_vals.len()={} but lhss={lhss}",
            perm_vals.len()
        )));
    }
    let mut seen = vec![false; lhss];
    for (i, &v) in perm_vals.iter().enumerate() {
        let v = v as usize;
        if v >= lhss {
            return Err(QuSpinError::ValueError(format!(
                "perm_vals[{i}]={v} is out of range 0..{lhss}"
            )));
        }
        if seen[v] {
            return Err(QuSpinError::ValueError(format!(
                "perm_vals has duplicate value {v}"
            )));
        }
        seen[v] = true;
    }
    Ok(())
}

fn validate_locs(locs: &[usize], n_sites: usize) -> Result<(), QuSpinError> {
    for (i, &loc) in locs.iter().enumerate() {
        if loc >= n_sites {
            return Err(QuSpinError::ValueError(format!(
                "locs[{i}]={loc} is out of range 0..{n_sites}"
            )));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Construction — pick the right family + width at build time
// ---------------------------------------------------------------------------

fn overflow_err(n_sites: usize, lhss: usize, n_bits: usize) -> QuSpinError {
    QuSpinError::ValueError(format!(
        "n_sites={n_sites} with lhss={lhss} requires {n_bits} bits, \
         exceeding the 8192-bit maximum"
    ))
}

fn build_space_inner(
    n_sites: usize,
    lhss: usize,
    space_kind: SpaceKind,
    fermionic: bool,
    n_bits: usize,
) -> Result<SpaceInner, QuSpinError> {
    match space_kind {
        SpaceKind::Full => build_full(n_sites, lhss, fermionic, n_bits),
        SpaceKind::Sub => build_sub(n_sites, lhss, fermionic, n_bits),
        SpaceKind::Symm => build_symm(n_sites, lhss, fermionic, n_bits),
    }
}

fn build_full(
    n_sites: usize,
    lhss: usize,
    fermionic: bool,
    n_bits: usize,
) -> Result<SpaceInner, QuSpinError> {
    if n_bits > 64 {
        return Err(QuSpinError::ValueError(format!(
            "Full basis requires n_bits <= 64, but n_sites={n_sites} with \
             lhss={lhss} needs {n_bits} bits"
        )));
    }
    // `Full*` variants exist in the *Default* sub-enum of every
    // family. We assign Full to whichever family matches the lhss.
    let inner = if n_bits <= 32 {
        let s = FullSpace::<u32>::new(lhss, n_sites, fermionic);
        match family_for(lhss) {
            Family::Bit => SpaceInner::Bit(SpaceInnerBit::Default(SpaceInnerBitDefault::Full32(s))),
            Family::Trit => {
                SpaceInner::Trit(SpaceInnerTrit::Default(SpaceInnerTritDefault::Full32(s)))
            }
            Family::Quat => {
                SpaceInner::Quat(SpaceInnerQuat::Default(SpaceInnerQuatDefault::Full32(s)))
            }
            Family::Dit => SpaceInner::Dit(SpaceInnerDit::Default(SpaceInnerDitDefault::Full32(s))),
        }
    } else {
        let s = FullSpace::<u64>::new(lhss, n_sites, fermionic);
        match family_for(lhss) {
            Family::Bit => SpaceInner::Bit(SpaceInnerBit::Default(SpaceInnerBitDefault::Full64(s))),
            Family::Trit => {
                SpaceInner::Trit(SpaceInnerTrit::Default(SpaceInnerTritDefault::Full64(s)))
            }
            Family::Quat => {
                SpaceInner::Quat(SpaceInnerQuat::Default(SpaceInnerQuatDefault::Full64(s)))
            }
            Family::Dit => SpaceInner::Dit(SpaceInnerDit::Default(SpaceInnerDitDefault::Full64(s))),
        }
    };
    Ok(inner)
}

#[derive(Clone, Copy)]
enum Family {
    Bit,
    Trit,
    Quat,
    Dit,
}

#[inline]
fn family_for(lhss: usize) -> Family {
    match lhss {
        2 => Family::Bit,
        3 => Family::Trit,
        4 => Family::Quat,
        _ => Family::Dit,
    }
}

fn build_sub(
    n_sites: usize,
    lhss: usize,
    fermionic: bool,
    n_bits: usize,
) -> Result<SpaceInner, QuSpinError> {
    macro_rules! sub_default {
        ($variant:ident, $B:ty, $family_ty:ident, $default_ty:ident, $outer:ident) => {{
            let s = Subspace::<$B>::new_empty(lhss, n_sites, fermionic);
            SpaceInner::$outer($family_ty::Default($default_ty::$variant(s)))
        }};
    }
    #[cfg(feature = "large-int")]
    macro_rules! sub_largeint {
        ($variant:ident, $B:ty, $family_ty:ident, $largeint_ty:ident, $outer:ident) => {{
            let s = Subspace::<$B>::new_empty(lhss, n_sites, fermionic);
            SpaceInner::$outer($family_ty::LargeInt($largeint_ty::$variant(s)))
        }};
    }
    let family = family_for(lhss);
    let inner = match (family, n_bits) {
        (Family::Bit, 0..=32) => {
            sub_default!(Sub32, u32, SpaceInnerBit, SpaceInnerBitDefault, Bit)
        }
        (Family::Bit, 33..=64) => {
            sub_default!(Sub64, u64, SpaceInnerBit, SpaceInnerBitDefault, Bit)
        }
        (Family::Bit, 65..=128) => {
            sub_default!(
                Sub128,
                ::ruint::Uint<128, 2>,
                SpaceInnerBit,
                SpaceInnerBitDefault,
                Bit
            )
        }
        (Family::Bit, 129..=256) => {
            sub_default!(
                Sub256,
                ::ruint::Uint<256, 4>,
                SpaceInnerBit,
                SpaceInnerBitDefault,
                Bit
            )
        }
        (Family::Trit, 0..=32) => {
            sub_default!(Sub32, u32, SpaceInnerTrit, SpaceInnerTritDefault, Trit)
        }
        (Family::Trit, 33..=64) => {
            sub_default!(Sub64, u64, SpaceInnerTrit, SpaceInnerTritDefault, Trit)
        }
        (Family::Trit, 65..=128) => sub_default!(
            Sub128,
            ::ruint::Uint<128, 2>,
            SpaceInnerTrit,
            SpaceInnerTritDefault,
            Trit
        ),
        (Family::Trit, 129..=256) => sub_default!(
            Sub256,
            ::ruint::Uint<256, 4>,
            SpaceInnerTrit,
            SpaceInnerTritDefault,
            Trit
        ),
        (Family::Quat, 0..=32) => {
            sub_default!(Sub32, u32, SpaceInnerQuat, SpaceInnerQuatDefault, Quat)
        }
        (Family::Quat, 33..=64) => {
            sub_default!(Sub64, u64, SpaceInnerQuat, SpaceInnerQuatDefault, Quat)
        }
        (Family::Quat, 65..=128) => sub_default!(
            Sub128,
            ::ruint::Uint<128, 2>,
            SpaceInnerQuat,
            SpaceInnerQuatDefault,
            Quat
        ),
        (Family::Quat, 129..=256) => sub_default!(
            Sub256,
            ::ruint::Uint<256, 4>,
            SpaceInnerQuat,
            SpaceInnerQuatDefault,
            Quat
        ),
        (Family::Dit, 0..=32) => {
            sub_default!(Sub32, u32, SpaceInnerDit, SpaceInnerDitDefault, Dit)
        }
        (Family::Dit, 33..=64) => {
            sub_default!(Sub64, u64, SpaceInnerDit, SpaceInnerDitDefault, Dit)
        }
        (Family::Dit, 65..=128) => sub_default!(
            Sub128,
            ::ruint::Uint<128, 2>,
            SpaceInnerDit,
            SpaceInnerDitDefault,
            Dit
        ),
        (Family::Dit, 129..=256) => sub_default!(
            Sub256,
            ::ruint::Uint<256, 4>,
            SpaceInnerDit,
            SpaceInnerDitDefault,
            Dit
        ),
        #[cfg(feature = "large-int")]
        (Family::Bit, 257..=512) => sub_largeint!(
            Sub512,
            ::ruint::Uint<512, 8>,
            SpaceInnerBit,
            SpaceInnerBitLargeInt,
            Bit
        ),
        #[cfg(feature = "large-int")]
        (Family::Bit, 513..=1024) => sub_largeint!(
            Sub1024,
            ::ruint::Uint<1024, 16>,
            SpaceInnerBit,
            SpaceInnerBitLargeInt,
            Bit
        ),
        #[cfg(feature = "large-int")]
        (Family::Bit, 1025..=2048) => sub_largeint!(
            Sub2048,
            ::ruint::Uint<2048, 32>,
            SpaceInnerBit,
            SpaceInnerBitLargeInt,
            Bit
        ),
        #[cfg(feature = "large-int")]
        (Family::Bit, 2049..=4096) => sub_largeint!(
            Sub4096,
            ::ruint::Uint<4096, 64>,
            SpaceInnerBit,
            SpaceInnerBitLargeInt,
            Bit
        ),
        #[cfg(feature = "large-int")]
        (Family::Bit, 4097..=8192) => sub_largeint!(
            Sub8192,
            ::ruint::Uint<8192, 128>,
            SpaceInnerBit,
            SpaceInnerBitLargeInt,
            Bit
        ),
        #[cfg(feature = "large-int")]
        (Family::Trit, 257..=512) => sub_largeint!(
            Sub512,
            ::ruint::Uint<512, 8>,
            SpaceInnerTrit,
            SpaceInnerTritLargeInt,
            Trit
        ),
        #[cfg(feature = "large-int")]
        (Family::Trit, 513..=1024) => sub_largeint!(
            Sub1024,
            ::ruint::Uint<1024, 16>,
            SpaceInnerTrit,
            SpaceInnerTritLargeInt,
            Trit
        ),
        #[cfg(feature = "large-int")]
        (Family::Trit, 1025..=2048) => sub_largeint!(
            Sub2048,
            ::ruint::Uint<2048, 32>,
            SpaceInnerTrit,
            SpaceInnerTritLargeInt,
            Trit
        ),
        #[cfg(feature = "large-int")]
        (Family::Trit, 2049..=4096) => sub_largeint!(
            Sub4096,
            ::ruint::Uint<4096, 64>,
            SpaceInnerTrit,
            SpaceInnerTritLargeInt,
            Trit
        ),
        #[cfg(feature = "large-int")]
        (Family::Trit, 4097..=8192) => sub_largeint!(
            Sub8192,
            ::ruint::Uint<8192, 128>,
            SpaceInnerTrit,
            SpaceInnerTritLargeInt,
            Trit
        ),
        #[cfg(feature = "large-int")]
        (Family::Quat, 257..=512) => sub_largeint!(
            Sub512,
            ::ruint::Uint<512, 8>,
            SpaceInnerQuat,
            SpaceInnerQuatLargeInt,
            Quat
        ),
        #[cfg(feature = "large-int")]
        (Family::Quat, 513..=1024) => sub_largeint!(
            Sub1024,
            ::ruint::Uint<1024, 16>,
            SpaceInnerQuat,
            SpaceInnerQuatLargeInt,
            Quat
        ),
        #[cfg(feature = "large-int")]
        (Family::Quat, 1025..=2048) => sub_largeint!(
            Sub2048,
            ::ruint::Uint<2048, 32>,
            SpaceInnerQuat,
            SpaceInnerQuatLargeInt,
            Quat
        ),
        #[cfg(feature = "large-int")]
        (Family::Quat, 2049..=4096) => sub_largeint!(
            Sub4096,
            ::ruint::Uint<4096, 64>,
            SpaceInnerQuat,
            SpaceInnerQuatLargeInt,
            Quat
        ),
        #[cfg(feature = "large-int")]
        (Family::Quat, 4097..=8192) => sub_largeint!(
            Sub8192,
            ::ruint::Uint<8192, 128>,
            SpaceInnerQuat,
            SpaceInnerQuatLargeInt,
            Quat
        ),
        #[cfg(feature = "large-int")]
        (Family::Dit, 257..=512) => sub_largeint!(
            Sub512,
            ::ruint::Uint<512, 8>,
            SpaceInnerDit,
            SpaceInnerDitLargeInt,
            Dit
        ),
        #[cfg(feature = "large-int")]
        (Family::Dit, 513..=1024) => sub_largeint!(
            Sub1024,
            ::ruint::Uint<1024, 16>,
            SpaceInnerDit,
            SpaceInnerDitLargeInt,
            Dit
        ),
        #[cfg(feature = "large-int")]
        (Family::Dit, 1025..=2048) => sub_largeint!(
            Sub2048,
            ::ruint::Uint<2048, 32>,
            SpaceInnerDit,
            SpaceInnerDitLargeInt,
            Dit
        ),
        #[cfg(feature = "large-int")]
        (Family::Dit, 2049..=4096) => sub_largeint!(
            Sub4096,
            ::ruint::Uint<4096, 64>,
            SpaceInnerDit,
            SpaceInnerDitLargeInt,
            Dit
        ),
        #[cfg(feature = "large-int")]
        (Family::Dit, 4097..=8192) => sub_largeint!(
            Sub8192,
            ::ruint::Uint<8192, 128>,
            SpaceInnerDit,
            SpaceInnerDitLargeInt,
            Dit
        ),
        _ => return Err(overflow_err(n_sites, lhss, n_bits)),
    };
    Ok(inner)
}

fn build_symm(
    n_sites: usize,
    lhss: usize,
    fermionic: bool,
    n_bits: usize,
) -> Result<SpaceInner, QuSpinError> {
    // Each family knows its L type, so the construction differs per
    // family. We use a small helper macro per family to avoid spelling
    // out every (width × variant) combination by hand.
    match family_for(lhss) {
        Family::Bit => build_symm_bit(n_sites, lhss, fermionic, n_bits),
        Family::Trit => build_symm_trit(n_sites, lhss, fermionic, n_bits),
        Family::Quat => build_symm_quat(n_sites, lhss, fermionic, n_bits),
        Family::Dit => build_symm_dit(n_sites, lhss, fermionic, n_bits),
    }
}

macro_rules! mk_symm {
    (
        $n_sites:expr, $lhss:expr, $fermionic:expr, $n_bits:expr,
        $L:ty,
        $family_variant:ident,
        $family_ty:ident,
        $default_ty:ident,
        $largeint_ty:ident
        $(,)?
    ) => {{
        let n_sites = $n_sites;
        let lhss = $lhss;
        let fermionic = $fermionic;
        let n_bits = $n_bits;
        match n_bits {
            0..=32 => {
                let s = SymBasis::<u32, $L, _>::new_empty(lhss, n_sites, fermionic);
                Ok(SpaceInner::$family_variant($family_ty::Default(
                    $default_ty::Sym32(s),
                )))
            }
            33..=64 => {
                let s = SymBasis::<u64, $L, _>::new_empty(lhss, n_sites, fermionic);
                Ok(SpaceInner::$family_variant($family_ty::Default(
                    $default_ty::Sym64(s),
                )))
            }
            65..=128 => {
                let s =
                    SymBasis::<::ruint::Uint<128, 2>, $L, _>::new_empty(lhss, n_sites, fermionic);
                Ok(SpaceInner::$family_variant($family_ty::Default(
                    $default_ty::Sym128(s),
                )))
            }
            129..=256 => {
                let s =
                    SymBasis::<::ruint::Uint<256, 4>, $L, _>::new_empty(lhss, n_sites, fermionic);
                Ok(SpaceInner::$family_variant($family_ty::Default(
                    $default_ty::Sym256(s),
                )))
            }
            #[cfg(feature = "large-int")]
            257..=512 => {
                let s =
                    SymBasis::<::ruint::Uint<512, 8>, $L, _>::new_empty(lhss, n_sites, fermionic);
                Ok(SpaceInner::$family_variant($family_ty::LargeInt(
                    $largeint_ty::Sym512(s),
                )))
            }
            #[cfg(feature = "large-int")]
            513..=1024 => {
                let s =
                    SymBasis::<::ruint::Uint<1024, 16>, $L, _>::new_empty(lhss, n_sites, fermionic);
                Ok(SpaceInner::$family_variant($family_ty::LargeInt(
                    $largeint_ty::Sym1024(s),
                )))
            }
            #[cfg(feature = "large-int")]
            1025..=2048 => {
                let s =
                    SymBasis::<::ruint::Uint<2048, 32>, $L, _>::new_empty(lhss, n_sites, fermionic);
                Ok(SpaceInner::$family_variant($family_ty::LargeInt(
                    $largeint_ty::Sym2048(s),
                )))
            }
            #[cfg(feature = "large-int")]
            2049..=4096 => {
                let s =
                    SymBasis::<::ruint::Uint<4096, 64>, $L, _>::new_empty(lhss, n_sites, fermionic);
                Ok(SpaceInner::$family_variant($family_ty::LargeInt(
                    $largeint_ty::Sym4096(s),
                )))
            }
            #[cfg(feature = "large-int")]
            4097..=8192 => {
                let s = SymBasis::<::ruint::Uint<8192, 128>, $L, _>::new_empty(
                    lhss, n_sites, fermionic,
                );
                Ok(SpaceInner::$family_variant($family_ty::LargeInt(
                    $largeint_ty::Sym8192(s),
                )))
            }
            _ => Err(overflow_err(n_sites, lhss, n_bits)),
        }
    }};
}

fn build_symm_bit(
    n_sites: usize,
    lhss: usize,
    fermionic: bool,
    n_bits: usize,
) -> Result<SpaceInner, QuSpinError> {
    type L<B> = PermDitMask<B>;
    // The macro can't carry a generic-over-B local-op type; Bit's
    // `PermDitMask<B>` needs the per-width substitution. Inline the
    // dispatch directly instead of using `mk_symm!`.
    match n_bits {
        0..=32 => {
            let s = SymBasis::<u32, L<u32>, _>::new_empty(lhss, n_sites, fermionic);
            Ok(SpaceInner::Bit(SpaceInnerBit::Default(
                SpaceInnerBitDefault::Sym32(s),
            )))
        }
        33..=64 => {
            let s = SymBasis::<u64, L<u64>, _>::new_empty(lhss, n_sites, fermionic);
            Ok(SpaceInner::Bit(SpaceInnerBit::Default(
                SpaceInnerBitDefault::Sym64(s),
            )))
        }
        65..=128 => {
            type B = ::ruint::Uint<128, 2>;
            let s = SymBasis::<B, L<B>, _>::new_empty(lhss, n_sites, fermionic);
            Ok(SpaceInner::Bit(SpaceInnerBit::Default(
                SpaceInnerBitDefault::Sym128(s),
            )))
        }
        129..=256 => {
            type B = ::ruint::Uint<256, 4>;
            let s = SymBasis::<B, L<B>, _>::new_empty(lhss, n_sites, fermionic);
            Ok(SpaceInner::Bit(SpaceInnerBit::Default(
                SpaceInnerBitDefault::Sym256(s),
            )))
        }
        #[cfg(feature = "large-int")]
        257..=512 => {
            type B = ::ruint::Uint<512, 8>;
            let s = SymBasis::<B, L<B>, _>::new_empty(lhss, n_sites, fermionic);
            Ok(SpaceInner::Bit(SpaceInnerBit::LargeInt(
                SpaceInnerBitLargeInt::Sym512(s),
            )))
        }
        #[cfg(feature = "large-int")]
        513..=1024 => {
            type B = ::ruint::Uint<1024, 16>;
            let s = SymBasis::<B, L<B>, _>::new_empty(lhss, n_sites, fermionic);
            Ok(SpaceInner::Bit(SpaceInnerBit::LargeInt(
                SpaceInnerBitLargeInt::Sym1024(s),
            )))
        }
        #[cfg(feature = "large-int")]
        1025..=2048 => {
            type B = ::ruint::Uint<2048, 32>;
            let s = SymBasis::<B, L<B>, _>::new_empty(lhss, n_sites, fermionic);
            Ok(SpaceInner::Bit(SpaceInnerBit::LargeInt(
                SpaceInnerBitLargeInt::Sym2048(s),
            )))
        }
        #[cfg(feature = "large-int")]
        2049..=4096 => {
            type B = ::ruint::Uint<4096, 64>;
            let s = SymBasis::<B, L<B>, _>::new_empty(lhss, n_sites, fermionic);
            Ok(SpaceInner::Bit(SpaceInnerBit::LargeInt(
                SpaceInnerBitLargeInt::Sym4096(s),
            )))
        }
        #[cfg(feature = "large-int")]
        4097..=8192 => {
            type B = ::ruint::Uint<8192, 128>;
            let s = SymBasis::<B, L<B>, _>::new_empty(lhss, n_sites, fermionic);
            Ok(SpaceInner::Bit(SpaceInnerBit::LargeInt(
                SpaceInnerBitLargeInt::Sym8192(s),
            )))
        }
        _ => Err(overflow_err(n_sites, lhss, n_bits)),
    }
}

fn build_symm_trit(
    n_sites: usize,
    lhss: usize,
    fermionic: bool,
    n_bits: usize,
) -> Result<SpaceInner, QuSpinError> {
    mk_symm!(
        n_sites,
        lhss,
        fermionic,
        n_bits,
        PermDitValues<3>,
        Trit,
        SpaceInnerTrit,
        SpaceInnerTritDefault,
        SpaceInnerTritLargeInt,
    )
}

fn build_symm_quat(
    n_sites: usize,
    lhss: usize,
    fermionic: bool,
    n_bits: usize,
) -> Result<SpaceInner, QuSpinError> {
    mk_symm!(
        n_sites,
        lhss,
        fermionic,
        n_bits,
        PermDitValues<4>,
        Quat,
        SpaceInnerQuat,
        SpaceInnerQuatDefault,
        SpaceInnerQuatLargeInt,
    )
}

fn build_symm_dit(
    n_sites: usize,
    lhss: usize,
    fermionic: bool,
    n_bits: usize,
) -> Result<SpaceInner, QuSpinError> {
    mk_symm!(
        n_sites,
        lhss,
        fermionic,
        n_bits,
        DynamicPermDitValues,
        Dit,
        SpaceInnerDit,
        SpaceInnerDitDefault,
        SpaceInnerDitLargeInt,
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    #[test]
    fn generic_basis_full_lhss3() {
        let basis = GenericBasis::new(2, 3, SpaceKind::Full, false).unwrap();
        assert_eq!(basis.inner.size(), 9);
        assert!(basis.inner.is_built());
    }

    #[test]
    fn generic_basis_lhss1_errors() {
        assert!(GenericBasis::new(4, 1, SpaceKind::Sub, false).is_err());
    }

    #[test]
    fn generic_basis_fermionic_with_lhss3_errors() {
        assert!(GenericBasis::new(4, 3, SpaceKind::Sub, true).is_err());
    }

    #[test]
    fn generic_basis_add_lattice_on_non_symm_errors() {
        let mut basis = GenericBasis::new(4, 2, SpaceKind::Sub, false).unwrap();
        let result = basis.add_lattice(Complex::new(1.0, 0.0), vec![1, 2, 3, 0]);
        assert!(result.is_err());
    }
}
