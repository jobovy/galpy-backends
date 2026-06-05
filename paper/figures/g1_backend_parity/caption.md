**Figure G1 — Backend value parity and autodiff consistency (CPU, float64).**

*(a)* Maximum absolute difference between the numpy reference and each backend (jax, torch) for the Plummer and Isochrone potential value $\Phi$, the radial and vertical forces $F_R,\,F_z$, and the second derivatives $\partial^2_R\Phi,\,\partial^2_z\Phi$, evaluated over a $60\times60$ grid in $(R,z)\in[0.3,3]\times[-1,1]$. In double precision, **jax reproduces numpy bit-for-bit** — exactly zero difference for every quantity (filled markers on the "$=0$" line) — because XLA's float64 primitives match numpy's. **torch agrees to within one unit in the last place** ($\lesssim 10^{-15}$; open markers near the float64-$\epsilon$ line): quantities built only from correctly-rounded primitives (e.g. `sqrt`) match exactly, while those using a non-integer power ($\mathrm{denom}^{-1.5}$, $\mathrm{denom}^{-2.5}$, which `pow` does not round correctly) differ by a fraction of a ULP at some grid points.

*(b)* Autodiff consistency: both `jax.grad` and `torch.autograd` reproduce galpy's analytic derivatives — $\partial\Phi/\partial R = -F_R$ and $\partial(-F_R)/\partial R = \partial^2_R\Phi$ — to $\sim 10^{-15}$.

**Take-away:** the multi-backend compute layer is numerically faithful (bit-identical under jax, machine-$\epsilon$ under torch) and differentiable to round-off.

*Note on the grid:* a single $z=\text{const}$ slice coincidentally shows zero difference for most quantities (catching only $F_R$); the full $(R,z)$ mesh is used so the per-element ULP differences surface honestly.
