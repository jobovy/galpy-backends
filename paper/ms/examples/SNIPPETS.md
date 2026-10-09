# Paper code snippets: corrected and reformatted

Each snippet of `sections/*.tex` (`\begin{code}` blocks and the one `\begin{codefig}`), its CURRENT text, and the REPLACEMENT text to paste between the `\begin{...}`/`\end{...}` lines. The replacement is exactly the snippet that `examples/<name>.py` runs (between the `# --- paper snippet begins/ends ---` markers), so it is tested by `tests_paper_examples.py`. Every replacement line is at most 56 characters, including indentation and comments.

Status: **simplify** = runs as is, but a workaround in it is no longer needed after a galpy fix; **WRONG** = the snippet failed or contradicted its claim and was fixed; **reformat** = layout only (lines > 56 characters, spacing, comments); **unchanged** = already fine.

| example | location | env | status |
|---|---|---|---|
| `design_01_rforce_jax` | sections/design.tex:40 | code | unchanged |
| `design_02_use_context` | sections/design.tex:51 | code | unchanged |
| `design_03_cuda_orbit` | sections/design.tex:59 | code | unchanged |
| `design_04_namespace_evaluate` | sections/design.tex:97 | code | unchanged |
| `design_05_sample_key` | sections/design.tex:195 | code | unchanged |
| `design_06_jit_grad_loss` | sections/design.tex:211 | code | unchanged |
| `potentials_01_jax_param_grad` | sections/potentials.tex:49 | code | unchanged |
| `potentials_02_torch_param_grad` | sections/potentials.tex:57 | code | unchanged |
| `potentials_03_new_potential_codefig` | sections/potentials.tex:132 | codefig | unchanged |
| `orbits_01_diffrax_param_grad` | sections/orbits.tex:22 | code | unchanged |
| `orbits_02_torchode_param_grad` | sections/orbits.tex:31 | code | unchanged |
| `orbits_03_cstm_jacobian` | sections/orbits.tex:90 | code | unchanged |
| `orbits_04_cstm_vmap` | sections/orbits.tex:99 | code | unchanged |
| `orbits_05_dxdv_lyapunov` | sections/orbits.tex:159 | code | unchanged |
| `actionangle_01_staeckel_param_grad` | sections/actionangle.tex:60 | code | unchanged |
| `actionangle_02_staeckel_c_jit_grad` | sections/actionangle.tex:96 | code | unchanged |
| `actionangle_03_estimate_delta` | sections/actionangle.tex:111 | code | unchanged |
| `actionangle_04_vertical_inverse_jacfwd` | sections/actionangle.tex:164 | code | unchanged |
| `dfs_01_constantbeta_torch_sample_grad` | sections/dfs.tex:106 | code | simplify |
| `dfs_02_king_W0_grad` | sections/dfs.tex:120 | code | simplify |
| `dfs_03_eddington_tracer_grad` | sections/dfs.tex:131 | code | simplify |
| `dfs_04_qdf_sampleV` | sections/dfs.tex:163 | code | simplify |
| `streams_01_streamdf` | sections/streams.tex:38 | code | unchanged |
| `streams_02_spray_jvp` | sections/streams.tex:107 | code | simplify |
| `streams_03_spray_track_jacfwd` | sections/streams.tex:123 | code | simplify |
| `examples_01_hvs_cstm` | sections/examples.tex:34 | code | unchanged |
| `examples_02_hvs_q_fit` | sections/examples.tex:45 | code | unchanged |
| `examples_03_neural_inverse` | sections/examples.tex:101 | code | unchanged |

Totals: 22 unchanged, 6 simplify

## Text around the snippets

- dfs.tex (Spherical DFs): "A DF built from a density that is not the one generating the potential is differentiable with respect to the parameters of the density" can again say "of both": the derivative w.r.t. the potential's parameter runs without a backend context (galpy #1646) and at a float radius (galpy #1656).
- streams.tex (Particle-spray models), after the `track` snippet: "but this derivative holds the discrete choices of the track fit (its binning in stripping time and the knots of its spline) fixed, and a finite difference of the fitted track does not converge onto it. For a likelihood, smooth statistics of the particles themselves are the better target." The first half holds; the second does not: finite differences of the same fit with steps small enough that no discrete choice changes (h = 1e-6 and 3e-7 for this stream) agree with each other and with `jacfwd` (see `streams_03`). Suggested: "where the derivative holds the discrete choices of the track fit (each particle's closest point on the progenitor's orbit and the amount of smoothing) fixed; it agrees with finite differences of the fitted track for steps small enough that none of these choices changes." (and drop the likelihood sentence, or keep it as advice about smoothness for optimizers).


## `design_01_rforce_jax` — sections/design.tex, line 40 (`code`)

**unchanged**: Matches the .tex.

Text (no change):

```python
from galpy.potential import (MWPotential2014,
                             evaluateRforces)
R= jnp.linspace(0.5,2.,1001)
FR= evaluateRforces(MWPotential2014,R,0.) # jax array
```

## `design_02_use_context` — sections/design.tex, line 51 (`code`)

**unchanged**: Matches the .tex.

Text (no change):

```python
import galpy.backend
with galpy.backend.use('jax'):
    Phi= evaluatePotentials(MWPotential2014,1.,0.) # jax
```

## `design_03_cuda_orbit` — sections/design.tex, line 59 (`code`)

**unchanged**: Matches the .tex.

Text (no change):

```python
o= Orbit(torch.tensor([1.,0.1,1.1,0.,0.05,0.],
                      device='cuda'))
o.integrate(ts,MWPotential2014,method='torchode')
o.R(ts) # a CUDA tensor
```

## `design_04_namespace_evaluate` — sections/design.tex, line 97 (`code`)

**unchanged**: Matches the .tex.

Text (no change):

```python
def _evaluate(self,R,z,phi=0.,t=0.):
    xp= get_namespace(R,z,phi,t)
    return -self._amp/xp.sqrt(R**2+z**2+self._b2)
```

## `design_05_sample_key` — sections/design.tex, line 195 (`code`)

**unchanged**: Matches the .tex.

Text (no change):

```python
from galpy.backend import random as grandom
hp= HernquistPotential(amp=2.,a=1.3)
df= isotropicHernquistdf(pot=hp)
sam= df.sample(n=10**6)          # numpy, as before
key= grandom.key(42,backend='jax')
sam= df.sample(n=10**6,key=key)  # jax, differentiable
```

## `design_06_jit_grad_loss` — sections/design.tex, line 211 (`code`)

**unchanged**: Matches the .tex.

Text (no change):

```python
def loss(q):
    pot= LogarithmicHaloPotential(normalize=1.,q=q)
    o= Orbit(jnp.asarray(ic))
    o.integrate(ts,pot,method='diffrax')
    return jnp.sum((o.x(ts)-x_obs)**2)
grad_loss= jax.jit(jax.grad(loss))
```

## `potentials_01_jax_param_grad` — sections/potentials.tex, line 49 (`code`)

**unchanged**: Matches the .tex.

Text (no change):

```python
from galpy.potential import MiyamotoNagaiPotential
def Rforce_of_a(a):
    mp= MiyamotoNagaiPotential(amp=1.,a=a,b=0.3)
    return mp.Rforce(1.2,0.1)
dFR_da= jax.grad(Rforce_of_a)(0.5)
```

## `potentials_02_torch_param_grad` — sections/potentials.tex, line 57 (`code`)

**unchanged**: Matches the .tex.

Text (no change):

```python
a= torch.tensor(0.5,requires_grad=True)
mp= MiyamotoNagaiPotential(amp=1.,a=a,b=0.3)
mp.Rforce(torch.tensor(1.2),
          torch.tensor(0.1)).backward()
a.grad # dFR/da
```

## `potentials_03_new_potential_codefig` — sections/potentials.tex, line 132 (`codefig`)

**unchanged**: Matches the .tex.

Text (no change):

```python
from galpy.potential import Potential
from galpy.backend import get_namespace
class MNLikePotential(Potential):
    # a Miyamoto-Nagai disk written for every backend
    def __init__(self,amp=1.,a=1.,b=0.1,
                 normalize=False):
        Potential.__init__(self,amp=amp)
        self._a= a
        self._b2= b**2
        if normalize: self.normalize(normalize)
    def _evaluate(self,R,z,phi=0.,t=0.):
        xp= get_namespace(R,z)
        s= self._a+xp.sqrt(z**2+self._b2)
        return -1./xp.sqrt(R**2+s**2)
    def _Rforce(self,R,z,phi=0.,t=0.):
        xp= get_namespace(R,z)
        s= self._a+xp.sqrt(z**2+self._b2)
        return -R*(R**2+s**2)**-1.5
    def _zforce(self,R,z,phi=0.,t=0.):
        xp= get_namespace(R,z)
        sqb= xp.sqrt(z**2+self._b2)
        s= self._a+sqb
        return -z*s/sqb*(R**2+s**2)**-1.5
# differentiate an orbit in it w.r.t. the scale length a
def final_z(a):
    o= Orbit(jnp.asarray([1.,0.1,1.1,0.1,0.05,0.]))
    mp= MNLikePotential(a=a,b=0.3,normalize=1.)
    o.integrate(ts,mp,method='diffrax')
    return o.z(ts[-1])
jax.grad(final_z)(0.5)
```

## `orbits_01_diffrax_param_grad` — sections/orbits.tex, line 22 (`code`)

**unchanged**: Matches the .tex.

Text (no change):

```python
def final_R(b):
    o= Orbit(jnp.asarray(ic))
    pp= PlummerPotential(amp=1.,b=b)
    o.integrate(ts,pp,method='diffrax')
    return o.R(ts[-1])
dR_db= jax.grad(final_R)(0.6)
```

## `orbits_02_torchode_param_grad` — sections/orbits.tex, line 31 (`code`)

**unchanged**: Matches the .tex.

Text (no change):

```python
b= torch.tensor(0.6,requires_grad=True)
o= Orbit(torch.as_tensor(ic))
pp= PlummerPotential(amp=1.,b=b)
o.integrate(torch.as_tensor(ts),pp,method='torchode')
o.R(ts[-1]).backward() # b.grad = dR/db
```

## `orbits_03_cstm_jacobian` — sections/orbits.tex, line 90 (`code`)

**unchanged**: Matches the .tex.

Text (no change):

```python
def end_point(ic):
    o= Orbit(ic)
    o.integrate(ts,MWPotential2014,
                method='dop853_c') # C + variational
    return jnp.stack([o.R(ts[-1]),o.z(ts[-1])])
M= jax.jacrev(end_point)(jnp.asarray(ic)) # STM rows
```

## `orbits_04_cstm_vmap` — sections/orbits.tex, line 99 (`code`)

**unchanged**: Matches the .tex.

Text (no change):

```python
Ms= jax.vmap(jax.jacrev(end_point))(ics) # ics: (N,6)
```

## `orbits_05_dxdv_lyapunov` — sections/orbits.tex, line 159 (`code`)

**unchanged**: Matches the .tex.

Text (no change):

```python
o= Orbit([0.8,0.3,0.75,0.,0.2,0.])
o.integrate_dxdv([1.,0.,0.,0.,0.,0.],ts,
                 MWPotential2014,method='dop853_c')
o.getOrbit_dxdv() # the deviation along the orbit
o.lyapunov(ts,MWPotential2014,method='dop853_c',
           renorm_every=10)
```

## `actionangle_01_staeckel_param_grad` — sections/actionangle.tex, line 60 (`code`)

**unchanged**: Matches the .tex.

Text (no change):

```python
def jr_of_a(a):
    mp= MiyamotoNagaiPotential(a=a,b=0.3)
    aAS= actionAngleStaeckel(pot=mp,delta=0.4,c=False)
    return aAS(R,vR,vT,z,vz)[0].sum()
djr_da= jax.grad(jr_of_a)(0.5)
```

## `actionangle_02_staeckel_c_jit_grad` — sections/actionangle.tex, line 96 (`code`)

**unchanged**: Matches the .tex.

Text (no change):

```python
aAS= actionAngleStaeckel(pot=MWPotential2014,
                         delta=0.45,c=True)
def total_jr(R):
    return aAS(R,vR,vT,z,vz)[0].sum()
dJ= jax.jit(jax.grad(total_jr))
djr_dR= dJ(jnp.asarray([1.1,0.9]))
```

## `actionangle_03_estimate_delta` — sections/actionangle.tex, line 111 (`code`)

**unchanged**: Matches the .tex.

Text (no change):

```python
delta= estimateDeltaStaeckel(MWPotential2014,
                             o.R(ts),o.z(ts))
aAS= actionAngleStaeckel(pot=MWPotential2014,
                         delta=delta,c=False)
jr,lz,jz= aAS(o.R(ts),o.vR(ts),o.vT(ts),
              o.z(ts),o.vz(ts))
```

## `actionangle_04_vertical_inverse_jacfwd` — sections/actionangle.tex, line 164 (`code`)

**unchanged**: Matches the .tex.

Text (no change):

```python
def torus(sigma):
    disk= IsothermalDiskPotential(amp=1.,sigma=sigma)
    aAVI= actionAngleVerticalInverse(
        pot=disk,Es=Es,nta=128,setup_interp=True)
    return aAVI.xvFreqs(J,angles)[:2] # (z,v_z)
z,vz= torus(0.5)
dz_dsigma,dvz_dsigma= jax.jacfwd(torus)(0.5)
```

## `dfs_01_constantbeta_torch_sample_grad` — sections/dfs.tex, line 106 (`code`)

**simplify**: The `galpy.backend.use('torch')` block is no longer needed: since galpy PR #1656 a differentiated potential parameter is the data for the constant-beta DF's $f(E)$ inversion and its sampler. Dropped and dedented; the derivative is unchanged (r_i/a to 1e-13).

Current:

```python
a= torch.tensor(1.,dtype=torch.float64,
                requires_grad=True)
with galpy.backend.use('torch'):
    hp= HernquistPotential(amp=2.,a=a)
    df= constantbetadf(pot=hp,beta=-0.2)
    key= galpy.backend.random.key(7,backend='torch')
    sam= df.sample(n=1000,key=key)
dr_da= torch.autograd.grad(sam.r().sum(),a)
```

Replacement:

```python
a= torch.tensor(1.,dtype=torch.float64,
                requires_grad=True)
hp= HernquistPotential(amp=2.,a=a)
df= constantbetadf(pot=hp,beta=-0.2)
key= galpy.backend.random.key(7,backend='torch')
sam= df.sample(n=1000,key=key)
dr_da= torch.autograd.grad(sam.r().sum(),a)
```

## `dfs_02_king_W0_grad` — sections/dfs.tex, line 120 (`code`)

**simplify**: The `galpy.backend.use('jax')` block is no longer needed (galpy PR #1646: the King constructor follows the traced W0), and `sigmar` takes a Python-float radius under a traced parameter (galpy PR #1656), so `jnp.asarray(0.5)` becomes `0.5`. The jitted derivative is unchanged (matches a five-point finite difference to 2e-7, and the old snippet to 1e-12).

Current:

```python
def sigmar(W0):
    with galpy.backend.use('jax'):
        kdf= kingdf(W0=W0,M=2.3,rt=1.4)
        return kdf.sigmar(jnp.asarray(0.5))
dsig= jax.jit(jax.grad(sigmar)) # ODE by diffrax
dsig(3.)
```

Replacement:

```python
def sigmar(W0):
    kdf= kingdf(W0=W0,M=2.3,rt=1.4)
    return kdf.sigmar(0.5)
dsig= jax.jit(jax.grad(sigmar)) # ODE by diffrax
dsig(3.)
```

## `dfs_03_eddington_tracer_grad` — sections/dfs.tex, line 131 (`code`)

**simplify**: `df.sigmar(jnp.asarray(1.))` becomes `df.sigmar(1.)`: since galpy PR #1656 the moments take a Python-float radius under a traced parameter. The derivative w.r.t. the potential's parameter also runs without a backend context and at a float radius (checked in the example), so "differentiable with respect to the parameters of the density" could again say "of both".

Current:

```python
def dispersion(a_tracer):
    hp= HernquistPotential(amp=0.1,a=a_tracer)
    df= eddingtondf(pot=NFWPotential(amp=1.,a=2.),
                    denspot=hp)
    return df.sigmar(jnp.asarray(1.))
jax.grad(dispersion)(0.7)
```

Replacement:

```python
def dispersion(a_tracer):
    hp= HernquistPotential(amp=0.1,a=a_tracer)
    df= eddingtondf(pot=NFWPotential(amp=1.,a=2.),
                    denspot=hp)
    return df.sigmar(1.)
jax.grad(dispersion)(0.7)
```

## `dfs_04_qdf_sampleV` — sections/dfs.tex, line 163 (`code`)

**simplify**: The `galpy.backend.use('jax')` block is no longer needed (galpy PR #1646: the draw follows the key); the same draw, and its derivative w.r.t. the DF's parameters matches a finite difference for MWPotential2014 and MWPotential.

Current:

```python
qdf= quasiisothermaldf(1./3.,0.2,0.1,1.,1.,
                       pot=MWPotential2014,aA=aAS)
with galpy.backend.use('jax'):
    vs= qdf.sampleV(0.9,0.05,n=10**4,
                    key=grandom.key(0,backend='jax'))
sigma_R= jnp.std(vs[:,0])
```

Replacement:

```python
qdf= quasiisothermaldf(1./3.,0.2,0.1,1.,1.,
                       pot=MWPotential2014,aA=aAS)
vs= qdf.sampleV(0.9,0.05,n=10**4,
                key=grandom.key(0,backend='jax'))
sigma_R= jnp.std(vs[:,0])
```

## `streams_01_streamdf` — sections/streams.tex, line 38 (`code`)

**unchanged**: Matches the .tex.

Text (no change):

```python
aAI= actionAngleIsochroneApprox(pot=lp,b=0.8,
        integrate_method='diffrax')
sdf= streamdf(sigv/220.,pot=lp,aA=aAI,
              progenitor=Orbit(jnp.asarray(prog)),
              tdisrupt=4.5/time_in_Gyr(220.,8.),
              leading=True,nTrackChunks=11)
sdf.sample(n=10**4,key=grandom.key(3,backend='jax'))
```

## `streams_02_spray_jvp` — sections/streams.tex, line 107 (`code`)

**simplify**: The progenitor can be set up from the list, `progenitor=Orbit(ic)` (galpy PR #1646: the numpy and backend samplers agree to ~1e-8), saving a line. `integrate_kwargs={'adjoint':'direct'}` stays: recent jax lets galpy select the direct adjoint under jvp itself (checked in the example), but jax 0.6, the last jax for Python 3.10, does not.

Current:

```python
key= galpy.backend.random.key(1,backend='jax')
def leading_arm(q):
    lp= LogarithmicHaloPotential(normalize=1.,q=q)
    prog= Orbit(jnp.asarray(ic))
    spdf= fardal15spraydf(mass,progenitor=prog,
        pot=lp,tdisrupt=td,
        integrate_kwargs={'adjoint':'direct'})
    return spdf.sample(n=1500,return_orbit=False,
                       tail='leading',key=key)
xv,dxv= jax.jvp(leading_arm,(q0,),(1.,))
```

Replacement:

```python
key= galpy.backend.random.key(1,backend='jax')
def leading_arm(q):
    lp= LogarithmicHaloPotential(normalize=1.,q=q)
    spdf= fardal15spraydf(mass,progenitor=Orbit(ic),
        pot=lp,tdisrupt=td,
        integrate_kwargs={'adjoint':'direct'})
    return spdf.sample(n=1500,return_orbit=False,
                       tail='leading',key=key)
xv,dxv= jax.jvp(leading_arm,(q0,),(1.,))
```

## `streams_03_spray_track_jacfwd` — sections/streams.tex, line 123 (`code`)

**simplify**: Same progenitor simplification as the previous snippet. The text after the snippet ("a finite difference of the fitted track does not converge onto it") no longer holds: finite differences of the same, traced fit (`jax.jit(track)`) with steps h = 1e-6 and 3e-7 agree with each other and with `jacfwd` (the fit's discrete choices change only for larger steps); the example now checks this instead of the non-convergence. (A finite difference of eager calls at float q does not converge because those take the NumPy fit, a different estimator.)

Current:

```python
def track(q):
    lp= LogarithmicHaloPotential(normalize=1.,q=q)
    prog= Orbit(jnp.asarray(ic))
    spdf= fardal15spraydf(mass,progenitor=prog,
        pot=lp,tdisrupt=td,
        integrate_kwargs={'adjoint':'direct'})
    tr= spdf.streamTrack(n=1500,key=key)
    return tr.x(tps)
track(0.9); jax.jacfwd(track)(0.9)
```

Replacement:

```python
def track(q):
    lp= LogarithmicHaloPotential(normalize=1.,q=q)
    spdf= fardal15spraydf(mass,progenitor=Orbit(ic),
        pot=lp,tdisrupt=td,
        integrate_kwargs={'adjoint':'direct'})
    tr= spdf.streamTrack(n=1500,key=key)
    return tr.x(tps)
track(0.9); jax.jacfwd(track)(0.9)
```

## `examples_01_hvs_cstm` — sections/examples.tex, line 34 (`code`)

**unchanged**: Matches the .tex.

Text (no change):

```python
def closest_approach_ic(vxvv):
    # vxvv: (ra,dec,d,pmra,pmdec,vlos)
    o= Orbit(vxvv,radec=True,ro=8.,vo=220.)
    o.integrate(-ts,pot,method='dop853_c') # C + STM
    return jnp.min(o.r(-ts))
J= jax.jacrev(closest_approach_ic)(
    jnp.asarray(vxvv_obs))
```

## `examples_02_hvs_q_fit` — sections/examples.tex, line 45 (`code`)

**unchanged**: Matches the .tex.

Text (no change):

```python
def closest_approach(q):
    pot= LogarithmicHaloPotential(normalize=1.,
                                  core=0.2,q=q)
    o= Orbit(jnp.asarray(vxvv),radec=True,ro=8.,vo=220.)
    o.integrate(-ts,pot,method='diffrax')
    return jnp.min(o.r(-ts))
objective= jax.jit(jax.value_and_grad(closest_approach))
q= adam_descent(objective,q0=1.1) # a few lines of optax
```

## `examples_03_neural_inverse` — sections/examples.tex, line 101 (`code`)

**unchanged**: Matches the .tex.

Text (no change):

```python
def loss(params,J,theta):
    z,vz= net.apply(params,J,theta) # the network
    Jp,_,thetap= aAV.actionsFreqsAngles(z,vz) # galpy
    return (jnp.mean((Jp-J)**2/J**2)
            +jnp.mean(wrap(thetap-theta)**2))
grads= jax.jit(jax.grad(loss))(params,J,theta)
```
