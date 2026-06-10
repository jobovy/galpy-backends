**Largest Lyapunov exponents from the variational equations:
`Orbit.lyapunov` recovering the literature Hénon–Heiles values and the
regular-orbit $\ln(t)/t$ decay.**
Running Benettin et al. (1980) estimates $\lambda(t)$: the deviation vector is
propagated with the C variational equations (`integrate_dxdv`) and
renormalized every 10 output intervals; `dop853_c` at default tolerances.

**(a)** Hénon–Heiles at $E=1/8$ ($\mathrm{amp}=1$, time in Hénon–Heiles
units), $t$ up to $3\times10^4$: the documented *chaotic* orbit F and
*regular* orbit E of Skokos et al. (2002, Sect. 3.2), both at the same
energy. The chaotic estimate converges into the literature band
$\lambda\simeq0.04$–$0.08$ for the chaotic sea at this energy (Benettin,
Galgani & Strelcyn 1976, Fig. 4; Skokos 2010, Fig. 2):
$\lambda(1.5\times10^4)=0.0425$, $\lambda(3\times10^4)=0.0456$. The regular
orbit at the *same* energy decays along the $\ln(t)/t$ guide to
$2.7\times10^{-4}$ — a factor $\sim170$ separation, the classic
chaos-detection contrast. **(b)** A regular Sun-like 3D orbit in
`MWPotential2014` ($R=1$, $v_T=1$, $z=0.0125$, $v_z=0.036$), $t$ up to
$3\times10^3$ internal time units ($\approx107\,$Gyr): $\lambda(t)$ decays as
$\ln(t)/t$ throughout, reaching $2.8\times10^{-3}$ ($\approx0.08\,$Gyr$^{-1}$
for $R_0=8\,$kpc, $v_0=220\,$km s$^{-1}$) — for a regular orbit the running
estimate is an upper limit that tightens indefinitely, so the curve tracking
the guide (rather than flattening) *is* the regularity diagnostic.

Caveat / provenance: measured against the `feat/pvar-lyapunov` galpy build
(commit `7ca49a1b`) **plus a local one-line fix** of
`integrateFullOrbit_dxdv`, whose returned base orbit had its $z$-column
overwritten by $v_T$ (an aliased-view bug); `Orbit.lyapunov` restarts each
renormalization segment from that returned orbit, so the 3D panels require
the fix (the unfixed code yields spurious plateaus $\lambda\sim0.05$ even for
integrable potentials). The planar panel (a) does not pass through the
affected code path and is unchanged by the fix. With the fix, segmented and
single-shot deviation propagation agree to $7\times10^{-13}$ and the estimate
is independent of the renormalization interval.
