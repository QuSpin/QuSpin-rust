//! Bit-width type aliases used across the family-specific dispatch
//! enums.
//!
//! `B128`/`B256` are always available; `B512`..`B8192` are gated
//! behind the `large-int` feature. Every dispatch sub-enum that
//! includes the wide widths is itself gated behind the same feature
//! so that no `#[cfg]` arms ever appear inside a match expression.

pub(crate) type B128 = ruint::Uint<128, 2>;
pub(crate) type B256 = ruint::Uint<256, 4>;

#[cfg(feature = "large-int")]
pub(crate) type B512 = ruint::Uint<512, 8>;
#[cfg(feature = "large-int")]
pub(crate) type B1024 = ruint::Uint<1024, 16>;
#[cfg(feature = "large-int")]
pub(crate) type B2048 = ruint::Uint<2048, 32>;
#[cfg(feature = "large-int")]
pub(crate) type B4096 = ruint::Uint<4096, 64>;
#[cfg(feature = "large-int")]
pub(crate) type B8192 = ruint::Uint<8192, 128>;
