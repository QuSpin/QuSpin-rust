use std::env;
use std::fs;
use std::path::{Path, PathBuf};

/// Rename whole-identifier occurrences of `from` to `to` in `src`, except
/// when the identifier is immediately followed by '.' (which only happens
/// inside `#include "fast_copy.h"`-style strings, which must stay intact).
fn rename_token(src: &str, from: &str, to: &str) -> String {
    let bytes = src.as_bytes();
    let mut out = String::with_capacity(src.len());
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i] as char;
        if c.is_ascii_alphanumeric() || c == '_' {
            // Collect the full identifier token.
            let start = i;
            while i < bytes.len() {
                let c = bytes[i] as char;
                if c.is_ascii_alphanumeric() || c == '_' {
                    i += 1;
                } else {
                    break;
                }
            }
            let token = &src[start..i];
            let next_is_dot = bytes.get(i).map(|b| *b as char) == Some('.');
            if token == from && !next_is_dot {
                out.push_str(to);
            } else {
                out.push_str(token);
            }
        } else {
            out.push(c);
            i += 1;
        }
    }
    out
}

/// Rename every identifier starting with `helper_` by appending `_{suffix}`,
/// except when immediately followed by '.' (inside include strings).
fn rename_helpers(src: &str, suffix: &str) -> String {
    let bytes = src.as_bytes();
    let mut out = String::with_capacity(src.len());
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i] as char;
        if c.is_ascii_alphanumeric() || c == '_' {
            let start = i;
            while i < bytes.len() {
                let c = bytes[i] as char;
                if c.is_ascii_alphanumeric() || c == '_' {
                    i += 1;
                } else {
                    break;
                }
            }
            let token = &src[start..i];
            let next_is_dot = bytes.get(i).map(|b| *b as char) == Some('.');
            if token.starts_with("helper_") && !next_is_dot {
                out.push_str(token);
                out.push('_');
                out.push_str(suffix);
            } else {
                out.push_str(token);
            }
        } else {
            out.push(c);
            i += 1;
        }
    }
    out
}

/// The upstream fht_*.c files define non-static functions:
///   fht_float, fht_double, fast_copy, and many helper_* helpers.
/// These collide across SIMD variants when linked into the same binary
/// for runtime dispatch, so we rename them all to be unique per variant
/// before compiling, and append the small *_oop wrapper functions
/// (normally provided by fht_impl.h, which we don't otherwise need).
fn generate_variant_source(csrc_dir: &Path, impl_file: &str, suffix: &str) -> String {
    let mut src = fs::read_to_string(csrc_dir.join(impl_file)).expect("read impl file");
    src.push('\n');
    src.push_str(&fs::read_to_string(csrc_dir.join("fast_copy.c")).expect("read fast_copy.c"));
    src.push('\n');

    for name in [
        "fht_float_oop",
        "fht_double_oop",
        "fht_float",
        "fht_double",
        "fast_copy",
    ] {
        let renamed = format!("{name}_{suffix}");
        src = rename_token(&src, name, &renamed);
    }
    src = rename_helpers(&src, suffix);

    src.push_str(&format!(
        r#"

int fht_float_oop_{suffix}(float *in, float *out, int log_n) {{
    fast_copy_{suffix}(out, in, sizeof(float) << log_n);
    return fht_float_{suffix}(out, log_n);
}}

int fht_double_oop_{suffix}(double *in, double *out, int log_n) {{
    fast_copy_{suffix}(out, in, sizeof(double) << log_n);
    return fht_double_{suffix}(out, log_n);
}}
"#
    ));

    src
}

fn build_variant(csrc_dir: &Path, out_dir: &Path, impl_file: &str, suffix: &str, flags: &[&str]) {
    let generated = generate_variant_source(csrc_dir, impl_file, suffix);
    let gen_path = out_dir.join(format!("fht_{suffix}.c"));
    fs::write(&gen_path, generated).expect("write generated source");

    let mut build = cc::Build::new();
    build.file(&gen_path).include(csrc_dir).opt_level(3);
    for f in flags {
        build.flag_if_supported(f);
    }
    build.compile(&format!("fht_{suffix}"));
}

fn main() {
    // Declare the custom cfg so `cargo build` doesn't warn about it on
    // newer Rust versions (unexpected_cfgs lint).
    println!("cargo:rustc-check-cfg=cfg(fht_have_simd_variants)");

    let csrc_dir = PathBuf::from("csrc");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Scalar variant: always built, portable to any target (incl. non-x86).
    // -ffast-math lets the auto-vectorizer use NEON/SSE where available.
    build_variant(
        &csrc_dir,
        &out_dir,
        "fht_scalar.c",
        "scalar",
        &["-ffast-math"],
    );

    if cfg!(any(target_arch = "x86", target_arch = "x86_64")) {
        // fht_sse.c / fht_avx.c use GNU extended inline asm (`__asm__ volatile`),
        // which plain MSVC (cl.exe) cannot compile. GCC, Clang, and clang-cl
        // (and MinGW) all support it. Skip these variants under MSVC so the
        // build doesn't fail; the scalar fallback is still fully functional.
        let is_msvc = cc::Build::new().get_compiler().is_like_msvc();
        if !is_msvc {
            build_variant(&csrc_dir, &out_dir, "fht_sse.c", "sse", &["-msse2"]);
            build_variant(
                &csrc_dir,
                &out_dir,
                "fht_avx.c",
                "avx2",
                &["-mavx2", "-mfma"],
            );
            println!("cargo:rustc-cfg=fht_have_simd_variants");
        } else {
            println!(
                "cargo:warning=MSVC detected: fht SSE/AVX2 variants use GNU inline asm \
                 and were skipped. Falling back to the scalar implementation. \
                 Use MinGW or clang-cl for SIMD variants on Windows."
            );
        }
    }

    println!("cargo:rerun-if-changed=csrc");
}
