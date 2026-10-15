# paper-p2730

This repository holds the certificate, the independent verifier and the generating code for the paper:

> Sergey Nefedov, *A certified bracket of width 1.5·10⁻⁸ for the L⁴ norm of the cyclic Hilbert transform on ℤ/31ℤ* (2026). PDF: [`paper/Nefedov_P2730_certified_bracket.pdf`](paper/Nefedov_P2730_certified_bracket.pdf).

C₃₁ is the sharp L⁴ operator norm of the sign Fourier multiplier, Ĥf(k) = −i·sgn(k)·f̂(k), acting on real zero-mean functions on ℤ/31ℤ. It is the subject of [TheoremDB Problem P2730](https://theoremdb.org/statements/P2730/). The certified result is:

```
1.5693754353475169 < C31 < 31387509/20000000 = 1.56937545
```

The previously recorded bracket was 1.5693 < C₃₁ < 1.6453.

## Verify

The verifier needs only Python 3 and its standard library:

```
python3 -I verify.py certificate.json
```

It runs in about 5 minutes on one core. Every inequality is checked in exact rational arithmetic, and the script prints `VERIFIED` only if all of them pass. The expected output is in [`verify_output.txt`](verify_output.txt).

## What the certificate proves

For every real f with Σf = 0, the certificate supplies an identity

```
U⁴·Σf⁴ − Σ(Hf)⁴ = zᵀQz + (Σf)·C(f) + Σ_α e_α f^α
```

with these ingredients:

- Q is a translation-invariant Gram matrix on the 496 quadratic monomials. It satisfies Q − 2⁻²⁹I = LLᵀ + R exactly, with ‖R‖ < 8.6·10⁻¹², so Q ⪰ 1.85·10⁻⁹ I.
- C is a cubic multiplier. The term (Σf)·C(f) vanishes on the zero-mean subspace.
- The residual coefficients satisfy Σ|e_α| < 2.05·10⁻¹¹, checked over all 46,376 monomials. The kernel h(j) is enclosed from scratch: π by Machin's formula, sin by a Taylor series with explicit remainder.

Together these give U⁴G − F ≥ 1.83·10⁻⁹·G on the zero-mean subspace. The lower bound comes from a rational zero-mean witness.

## Repository layout

| Path | Contents |
|---|---|
| `verify.py` | Independent exact checker (standard library only) |
| `certificate.json` | Integer certificate: K and the cubic coefficients (scale 2⁻⁵⁰), L (scale 2⁻⁴⁵), the witness q, U, γ₀ |
| `verify_output.txt` | Checker output |
| `extremizer.txt` | Numerical C₃₁ to 60 digits and the critical point to 50 digits (not certified) |
| `paper/` | LaTeX source and PDF |
| `generation/` | Numerical SDP (CVXPY + Clarabel), rationalization, 80-digit Newton solve. Not needed for verification. |

### Regenerating the certificate (optional)

```
cd generation
pip install numpy scipy cvxpy clarabel mpmath
python3 sdp2.py 1e-8                       # SDP solution -> sol_1e-08.pkl (included)
python3 gen.py sol_1e-08.pkl 31387509/20000000 29 ../certificate_new.json
python3 newton.py                          # 80-digit extremizer -> fstar_mp.pkl (included)
```

The published certificate uses as witness the 80-digit point from `fstar_mp.pkl`, scaled by 2⁸⁰ (paper, Section 5). Versions used: Python 3.13.16, NumPy 2.5.3, SciPy 1.18.1, CVXPY 1.9.3, Clarabel 0.11.1, mpmath 1.4.1.

## License

The code is released under the MIT License. The data (`certificate.json`, `extremizer.txt`, `*.pkl`, `*.npy`) are released under CC BY 4.0. See [`LICENSE`](LICENSE). To cite this work, see [`CITATION.cff`](CITATION.cff).
