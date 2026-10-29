"""Per-site classification of the island audit; per-test tiers."""
import json, sys, collections

# function-level categories (file:function); line overrides in SITE
FUNC = {
 # C extensions (argument packing; C bridges with C-STM / custom-VJP derivatives)
 "orbit/integratePlanarOrbit.py:_finalize_pot_args": "C", "orbit/integratePlanarOrbit.py:_parse_scf_pot": "C",
 "potential/MultipoleExpansionPotential.py:_serialize_for_c": "C",
 "actionAngle/actionAngleStaeckel_c.py:*": "C", "actionAngle/actionAngleStaeckel.py:_staeckel_c_backend_refu0": "C",
 "actionAngle/actionAngleStaeckel.py:_staeckel_c_freq_circ_fix": "C", "actionAngle/actionAngleStaeckel.py:<genexpr>": "C",
 "orbit/Orbits.py:_setupaA": "C", "orbit/Orbits.py:_setup_actionsFreqsAngles": "C", "orbit/Orbits.py:_integrate_impl": "C",
 "df/streamspraydf.py:_integrate_progenitor": "C", "df/streamspraydf.py:_integrate_center": "C",
 # host control and bookkeeping
 "orbit/Orbits.py:__init__": "CTL", "orbit/Orbits.py:_generate_auto_time_array": "CTL", "orbit/Orbits.py:_integrate_backend_continued": "CTL",
 "orbit/Orbits.py:_call_internal": "CTL", "orbit/Orbits.py:_call_internal_backend_interp": "CTL", "orbit/Orbits.py:_cache_backend_interp": "CTL",
 "orbit/Orbits.py:_unbound_indx_and_aAkwargs": "CTL", "orbit/Orbits.py:_calculate_dynamical_time": "CTL", "orbit/Orbits.py:SOS": "CTL",
 "orbit/Orbits.py:bruteSOS": "CTL",
 "actionAngle/actionAngleSpherical.py:_relative_xtol": "CTL", "actionAngle/actionAngleSpherical.py:_problem_backend": "CTL",
 "actionAngle/actionAngleVertical.py:_calc_xmax_backend": "CTL", "actionAngle/actionAngleIsochroneInverse.py:_xvFreqs": "CTL",
 "actionAngle/actionAngleStaeckel.py:_staeckel_uminumax": "CTL", "actionAngle/actionAngleStaeckel.py:_staeckel_vmin": "CTL",
 "actionAngle/actionAngleIsochroneApprox.py:estimateBIsochrone": "CTL", "actionAngle/actionAngleVerticalInverse.py:_mm_J_of_E_dispatch": "CTL",
 "actionAngle/actionAngleVerticalInverse.py:_init_mm_backend": "CTL", "df/osipkovmerrittdf.py:_panel_root": "CTL",
 "df/sphericaldf.py:_make_pvr_interpolator_backend": "CTL", "df/sphericaldf.py:_make_cmf_interpolator": "CTL",
 "df/sphericaldf.py:_make_pvr_interpolator": "CTL",
 "potential/SCFPotential.py:_init_static": "CTL", "potential/SCFPotential.py:_init_timedep": "CTL",
 "potential/ExpTruncNFWPotential.py:__init__": "CTL", "potential/EinastoPotential.py:__init__": "CTL",
 "potential/EllipsoidalPotential.py:*": "CTL", "potential/RotateAndTiltWrapperPotential.py:*": "CTL",
 "potential/FerrersPotential.py:_derivs_xyz": "CTL", "potential/RazorThinExponentialDiskPotential.py:_force_panels": "CTL",
 "potential/NonInertialFrameForce.py:_force": "CTL",
 "util/symplecticode.py:leapfrog": "CTL", "util/conversion.py:_parse_grid_quantity": "CTL",
 "df/streamdf.py:_determine_nTrackIterations": "CTL", "df/streamspraydf.py:_auto_track_time_range": "CTL",
 "df/streamspraydf.py:_sample_tail": "CTL", "df/streamspraydf.py:pericenter_stripping_pdf": "CTL",
 "df/streamspraydf.py:streamTrack": "CTL", "df/streamTrack.py:_prog_at": "CTL", "df/streamTrack.py:_fit_track_from_particles": "CTL",
 "df/streamTrack.py:_smooth_series_backend": "CTL", "df/streamTrack.py:__init__": "CTL",
 "potential/Potential.py:rl": "CTL", "potential/Potential.py:rE": "CTL", "potential/Potential.py:zvc": "CTL",
 "potential/Potential.py:zvc_range": "CTL", "potential/TwoPowerSphericalPotential.py:rvir": "CTL",
 # output: astropy, matplotlib, numexpr, NEMO
 "util/conversion.py:wrapped": "OUT", "util/conversion.py:<genexpr>": "OUT", "orbit/Orbits.py:SkyCoord": "OUT",
 "orbit/Orbits.py:plot": "OUT", "orbit/Orbits.py:plot3d": "OUT", "orbit/Orbits.py:plotSOS": "OUT", "orbit/Orbits.py:_base_plotSOS": "OUT",
 "orbit/Orbits.py:_parse_plot_quantity": "OUT", "orbit/Orbits.py:_obs_asnumpy": "OUT", "util/plot.py:plot": "OUT",
 "potential/Potential.py:plotPotentials": "OUT", "potential/Potential.py:plotDensities": "OUT", "potential/Potential.py:plotSurfaceDensities": "OUT",
 "df/streamTrack.py:plot": "OUT", "df/streamdf.py:plotCompareTrackAAModel": "OUT", "df/streamdf.py:_parse_progenitor_dim": "OUT",
 "df/streamspraydf.py:sample": "OUT", "df/sphericaldf.py:vmomentdensity": "OUT", "df/diskdf.py:vmomentsurfacemass": "OUT",
 "df/quasiisothermaldf.py:vmomentdensity": "OUT", "df/quasiisothermaldf.py:jmomentdensity": "OUT",
 "actionAngle/actionAngleIsochroneApprox.py:plot": "OUT", "actionAngle/actionAngleIsochroneApprox.py:<genexpr>": "OUT",
 "potential/TwoPowerSphericalPotential.py:_nemo_accpars": "OUT", "util/multi.py:run_tasks": "PYINT",
 # numpy functions supplied by the test, called with backend nodes
 "potential/SCFPotential.py:integrand": "USER", "potential/SCFPotential.py:integrand_batched": "USER",
 "potential/SCFPotential.py:_dens_accepts_arrays": "USER", "potential/SCFPotential.py:scf_compute_coeffs_axi": "USER",
 "potential/SCFPotential.py:scf_compute_coeffs_spherical": "USER", "potential/SCFPotential.py:_timedep_dens_setup": "USER",
 "potential/SCFPotential.py:f": "USER", "potential/SCFPotential.py:f_batched": "USER", "potential/SCFPotential.py:<lambda>": "USER",
 "potential/SCFPotential.py:from_density": "USER", "potential/SCFPotential.py:integrand_batched_reduce": "USER",
 "potential/KuijkenDubinskiDiskExpansionPotential.py:*": "USER", "potential/AnySphericalPotential.py:_backend_dens": "USER",
 "df/quasiisothermaldf.py:__call__": "USER",
 # construction-time constant tables (nothing differentiated); evaluation on the backend
 "potential/MultipoleExpansionPotential.py:from_density": "TAB", "potential/MultipoleExpansionPotential.py:_scf_density_multipoles": "TAB",
 "potential/MultipoleExpansionPotential.py:_multipoles_at": "TAB", "potential/SCFPotential.py:_scf_coeffs_from_multipole": "TAB",
 "potential/SCFPotential.py:_init_timedep_backend": "TAB", "potential/SCFPotential.py:from_nbody": "TAB",
 "potential/interpSphericalPotential.py:__init__": "TAB", "potential/AdiabaticContractionWrapperPotential.py:__init__": "TAB",
 "potential/ChandrasekharDynamicalFrictionForce.py:__init__": "TAB", "df/jeans.py:_on": "TAB",
 "potential/interpRZPotential.py:__init__": "TAB", "potential/interpRZPotential.py:_grid_eval": "TAB", "potential/interpRZPotential.py:_grid": "TAB",
 "potential/EinastoPotential.py:_calculate_dn": "TAB", "potential/Potential.py:_rlfunc_host": "TAB",
 "df/quasiisothermaldf.py:_setup_rg_table": "TAB", "df/quasiisothermaldf.py:_setup_rg_table_backend": "TAB",
 "df/sphericaldf.py:_ensure_fE_interp": "TAB", "df/sphericaldf.py:_setup_rphi_interpolator": "TAB", "df/sphericaldf.py:dMdE": "TAB",
 "df/sphericaldf.py:_dMdE": "TAB", "df/osipkovmerrittdf.py:_ensure_fQ_interp": "TAB",
 "df/constantbetadf.py:_evalpot_asnumpy": "TAB", "df/constantbetadf.py:_fEintegrand_raw": "TAB",
 "actionAngle/actionAngleStaeckelGrid.py:*": "TAB", "actionAngle/actionAngleAdiabaticGrid.py:*": "TAB",
 "actionAngle/actionAngleVerticalInverse.py:__init__": "TAB", "actionAngle/actionAngleVerticalInverse.py:wrapper": "TAB",
 "actionAngle/actionAngleVerticalInverse.py:<lambda>": "TAB",
 # sampling with numpy's global RNG (a backend key runs the backend sampler)
 "df/sphericaldf.py:_sample_v": "RNG", "df/sphericaldf.py:_sample_r": "RNG", "df/sphericaldf.py:_numpy_spline": "RNG",
 "df/quasiisothermaldf.py:sampleV": "RNG", "df/quasiisothermaldf.py:sampleV_interpolate": "RNG",
 # streamdf.sample(key=None): numpy-RNG draws propagated on the stream's backend, output cast to
 # numpy like the other samplers' key=None output (retired with Jo, 2026-10-18)
 "df/streamdf.py:sample": "RNG",
 "df/quasiisothermaldf.py:_sampleV_preoptimized": "RNG", "df/diskdf.py:wrapper": "RNG", "df/diskdf.py:_ELtowRRapRperi": "RNG",
 # the legacy pure-Python integrators (numpy by design; warned)
 "util/leung_dop853.py:*": "PYINT", "orbit/integrateFullOrbit.py:*": "PYINT", "orbit/integratePlanarOrbit.py:integrate_for_map": "PYINT",
 "orbit/integratePlanarOrbit.py:_planarEOM_dxdv": "PYINT", "orbit/integratePlanarOrbit.py:_planarRectForce": "PYINT",
 "orbit/integrateLinearOrbit.py:*": "PYINT",
 # documented exceptions
 "df/evolveddiskdf.py:*": "EXC", "potential/AnyAxisymmetricRazorThinDiskPotential.py:*": "EXC",
 # a numpy-built stream (module fixtures built before the force fixture): its
 # lazily-computed state stays numpy, like the object (#1726)
 "df/streamdf.py:_determine_stream_spreadLB": "EXC", "df/streamdf.py:_interpolate_stream_track": "EXC",
 "df/streamdf.py:_lb_track": "EXC", "df/streamdf.py:length": "EXC",
 # numpy-by-design callers: the Python integrators' force queries (MovingObject's
 # numpy pin), StaeckelGrid's tables and the planar Python integrator (interpRZ's
 # numpy branch, out-of-grid points)
 "potential/MovingObjectPotential.py:wrapper": "PYINT",
 "potential/interpRZPotential.py:_evaluate": "TAB", "potential/interpRZPotential.py:_Rforce": "PYINT",
 "df/streamdf.py:_parse_track_dim": "OUT",
 # genuine islands: numpy computation that could run on the backend
 "df/diskdf.py:_ssp": "ISL",
 "actionAngle/actionAngleIsochroneApprox.py:_h": "ISL",
 "df/streamdf.py:*": "ISL", "df/streamgapdf.py:*": "ISL", "df/streamTrack.py:__call__": "ISL",
 "df/quasiisothermaldf.py:estimate_hr": "ISL", "df/quasiisothermaldf.py:estimate_hz": "ISL", "df/quasiisothermaldf.py:estimate_hsr": "ISL",
 "df/quasiisothermaldf.py:estimate_hsz": "ISL", "df/quasiisothermaldf.py:surfacemass_z": "ISL",
 "orbit/Orbits.py:E": "ISL", "orbit/Orbits.py:rguiding": "ISL", "orbit/Orbits.py:rE": "ISL", "orbit/Orbits.py:LcE": "ISL",
 "orbit/Orbits.py:Jacobi": "ISL", "orbit/Orbits.py:_fit_orbit_mlogl": "ISL", "util/coords.py:align_to_orbit": "ISL",
 "potential/Potential.py:lindbladR": "ISL", "potential/Potential.py:rhalf": "ISL",
 "potential/Potential.py:_lindbladR_backend": "CTL", "df/quasiisothermaldf.py:_rl_host": "ISL",
 "actionAngle/actionAngleIsochroneApprox.py:_parse_args": "C",
 "potential/planarPotential.py:*": "ISL", "potential/SpiralArmsPotential.py:*": "ISL", "potential/Irrgang13.py:*": "ISL",
 "potential/MultipoleExpansionPotential.py:_compute_spher_forces_at_point": "ISL",
 "potential/MultipoleExpansionPotential.py:_compute_spher_2nd_derivs_at_point": "ISL",
 "potential/MultipoleExpansionPotential.py:_eval_R_lm": "ISL", "potential/MultipoleExpansionPotential.py:_eval_dR_lm": "ISL",
 "potential/MultipoleExpansionPotential.py:_eval_radial_lm_timedep": "ISL", "util/special.py:compute_legendre": "ISL",
 "potential/ChandrasekharDynamicalFrictionForce.py:lnLambda": "ISL",
}
# scipy fallbacks inside otherwise-backend root finders
SITE = {
 "orbit/Orbits.py:10487:<lambda>": "CTL", "orbit/Orbits.py:10488:<lambda>": "CTL",
 "potential/Potential.py:5367:rhalf": "CTL",
 "potential/Potential.py:3709:rl": "ISL", "potential/Potential.py:3719:rl": "ISL",
 "potential/Potential.py:3916:rE": "ISL", "potential/Potential.py:3928:rE": "ISL",
 "potential/Potential.py:5156:zvc": "ISL", "potential/Potential.py:5157:zvc": "ISL", "potential/Potential.py:5173:zvc": "ISL",
 "potential/Potential.py:5267:zvc_range": "ISL", "potential/Potential.py:5284:zvc_range": "ISL",
 "potential/TwoPowerSphericalPotential.py:1577:rvir": "ISL", "potential/TwoPowerSphericalPotential.py:1578:rvir": "ISL",
}

def category(site):
    s = site[len("galpy/"):]
    if s in SITE: return SITE[s]
    f, line, func = s.rsplit(":", 2)
    return FUNC.get(f"{f}:{func}") or FUNC.get(f"{f}:*") or "UNCLASSIFIED"

ORDER = ["C", "CTL", "OUT", "USER", "TAB", "RNG", "PYINT", "EXC", "ISL"]
bk = sys.argv[1]
m = json.load(open(f"/var/tmp/claude-probes/audit/merged_{bk}.json"))
ran = [t for t, o in m["all"].items() if o in ("passed", "failed")]
tcat = collections.defaultdict(set); unc = collections.Counter()
for s, ts in m["site_tests"].items():
    c = category(s)
    if c == "UNCLASSIFIED": unc[s] = len(ts)
    for t in ts: tcat[t].add(c)
print(f"{bk}: ran {len(ran)}; unclassified sites {len(unc)}", unc.most_common(12))
n = len(ran)
def share(pred):
    k = sum(1 for t in ran if pred(tcat.get(t, set())))
    return k, 100.0 * k / n
tiers = [
 ("no numpy contact at all", lambda c: not c),
 ("+ C extensions", lambda c: c <= {"C"}),
 ("+ host control/bookkeeping", lambda c: c <= {"C", "CTL"}),
 ("+ output (units/plots)", lambda c: c <= {"C", "CTL", "OUT"}),
 ("+ test-supplied numpy functions", lambda c: c <= {"C", "CTL", "OUT", "USER"}),
 ("+ constant setup tables", lambda c: c <= {"C", "CTL", "OUT", "USER", "TAB"}),
 ("+ numpy-RNG sampling", lambda c: c <= {"C", "CTL", "OUT", "USER", "TAB", "RNG"}),
 ("+ Python integrators", lambda c: c <= {"C", "CTL", "OUT", "USER", "TAB", "RNG", "PYINT"}),
 ("+ documented exceptions", lambda c: "ISL" not in c),
]
for name, p in tiers:
    k, f = share(p); print(f"  {name:36s} {k:5d}  {f:6.2f}%")
isl = collections.Counter()
for s, ts in m["site_tests"].items():
    if category(s) == "ISL":
        f, line, func = s[len("galpy/"):].rsplit(":", 2)
        isl[f"{f}:{func}"] = max(isl[f"{f}:{func}"], 0)
isl_tests = collections.defaultdict(set)
for s, ts in m["site_tests"].items():
    if category(s) == "ISL":
        f, line, func = s[len("galpy/"):].rsplit(":", 2); isl_tests[f"{f}:{func}"] |= set(ts)
print("  islands by function (tests):")
for k, v in sorted(isl_tests.items(), key=lambda kv: -len(kv[1])): print(f"    {len(v):4d} {k}")
only = collections.Counter()
for t in ran:
    c = tcat.get(t, set())
    if "ISL" in c: only[t.split("::")[0]] += 1
print("  island tests by file:", only.most_common())
json.dump({t: sorted(tcat.get(t, set())) for t in ran}, open(f"/var/tmp/claude-probes/audit/tcat_{bk}.json", "w"))
