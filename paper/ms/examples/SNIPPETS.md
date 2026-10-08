# Paper code snippets: corrected and reformatted

Each snippet of `sections/*.tex` (`\begin{code}` blocks and the one `\begin{codefig}`), its CURRENT text, and the REPLACEMENT text to paste between the `\begin{...}`/`\end{...}` lines. The replacement is exactly the snippet that `examples/<name>.py` runs (between the `# --- paper snippet begins/ends ---` markers), so it is tested by `tests_paper_examples.py`. Every replacement line is at most 56 characters, including indentation and comments.

Status: **WRONG** = the snippet failed or contradicted its claim and was fixed; **reformat** = layout only (lines > 56 characters, spacing, comments); **unchanged** = already fine.

| example | location | env | status |
|---|---|---|---|
| `design_01_rforce_jax` | sections/design.tex:39 | code | reformat |
| `design_02_use_context` | sections/design.tex:49 | code | WRONG |
| `design_03_cuda_orbit` | sections/design.tex:56 | code | WRONG |
| `design_04_namespace_evaluate` | sections/design.tex:92 | code | unchanged |
| `design_05_sample_key` | sections/design.tex:189 | code | reformat |
| `design_06_jit_grad_loss` | sections/design.tex:204 | code | unchanged |
| `potentials_01_jax_param_grad` | sections/potentials.tex:48 | code | unchanged |
| `potentials_02_torch_param_grad` | sections/potentials.tex:56 | code | reformat |
| `potentials_03_new_potential_codefig` | sections/potentials.tex:130 | codefig | reformat |
| `orbits_01_diffrax_param_grad` | sections/orbits.tex:22 | code | reformat |
| `orbits_02_torchode_param_grad` | sections/orbits.tex:30 | code | unchanged |
| `orbits_03_cstm_jacobian` | sections/orbits.tex:87 | code | reformat |
| `orbits_04_cstm_vmap` | sections/orbits.tex:95 | code | reformat |
| `orbits_05_dxdv_lyapunov` | sections/orbits.tex:155 | code | reformat |
| `actionangle_01_staeckel_param_grad` | sections/actionangle.tex:59 | code | reformat |
| `actionangle_02_staeckel_c_jit_grad` | sections/actionangle.tex:95 | code | reformat |
| `actionangle_03_estimate_delta` | sections/actionangle.tex:108 | code | reformat |
| `actionangle_04_vertical_inverse_jacfwd` | sections/actionangle.tex:158 | code | reformat |
| `dfs_01_constantbeta_torch_sample_grad` | sections/dfs.tex:106 | code | WRONG |
| `dfs_02_king_W0_grad` | sections/dfs.tex:118 | code | WRONG |
| `dfs_03_eddington_tracer_grad` | sections/dfs.tex:126 | code | reformat |
| `dfs_04_qdf_sampleV` | sections/dfs.tex:156 | code | WRONG |
| `streams_01_streamdf` | sections/streams.tex:38 | code | WRONG |
| `streams_02_spray_jvp` | sections/streams.tex:99 | code | WRONG |
| `streams_03_spray_track_jacfwd` | sections/streams.tex:113 | code | WRONG |
| `examples_01_hvs_cstm` | sections/examples.tex:34 | code | reformat |
| `examples_02_hvs_q_fit` | sections/examples.tex:43 | code | reformat |
| `examples_03_neural_inverse` | sections/examples.tex:98 | code | reformat |

Totals: 16 reformat, 8 WRONG, 4 unchanged

## Claims in the surrounding text that the examples contradict

- design.tex (Choosing a backend): "When there is no array to dispatch on, as when a user sets up an \orb\ instance from \python\ floats, the default backend is used". False for Orbit: the IC stays numpy and is integrated in C inside `use('jax')`. True for potentials (and other functions) evaluated at Python floats.
- design.tex: "an \orb\ set up from a tensor on a GPU is integrated there" holds only with an in-backend method (`method='torchode'`/`'torchdiffeq'`); the default method integrates in C on the host. With torchode the times must also be on the GPU.
- dfs.tex (Spherical DFs): "this derivative is $r_i/a$, which the sampler reproduces to $10^{-13}$" holds without `rmin` (with `rmin=0.1` it is off by 2--4%) and for a float64 `a` (float32: 5e-8).
- dfs.tex: the King d/dW0 and the Eddington d/d(potential parameter) run only inside `galpy.backend.use('jax')` (both eagerly and under `jax.jit`); without it, the constructors evaluate the potential at Python floats on numpy and fail on the traced parameter. "differentiable with respect to the parameters of both" therefore needs that qualifier (the tracer-parameter derivative works without it).
- dfs.tex (Disk DFs): "a sample is differentiable in the DF's parameters for a fixed key": for `MWPotential2014` the derivative is NaN (any of hr, sr, sz, hsr, hsz); for `MWPotential` it is finite and correct. `sampleV(key=<jax key>)` also needs `use('jax')` when the qdf was built from floats.
- streams.tex (Particle-spray models): "The whole computation compiles and is differentiable with respect to every parameter": forward mode (jvp/jacfwd) needs `integrate_kwargs={'adjoint':'direct'}`, and the 1e-6 agreement with finite differences needs a jax progenitor Orbit (a numpy progenitor gives 7e-5). `streamTrack()` returns an object, so it is differentiated through its accessors (e.g. `.x(tp)`); and "The fitted track ... is obtained in the same way" holds for running jacfwd but the result is the derivative with the track fit's discrete choices frozen, which a finite difference of the track does not reproduce (it does not converge). The 1e-6 agreement is a property of the particles, not of the fitted track.
- No contradiction found for: the C-STM Jacobian (agrees with finite differences of tightly converged C orbits to 3e-10--1.3e-9, consistent with "a few parts in $10^{10}$"), `jax.vmap` over the C-STM, the c=True Staeckel derivative through C under `jax.jit`, `estimateDeltaStaeckel` inside `jax.jit`, the vertical-inverse torus derivative (3e-10 vs. finite differences), the HVS q fit (recovers 0.8485 for 0.85).

## Once galpy PR #1646 (usability fixes) is merged

The examples now check the fixed behaviour (so they need that PR), and these snippets can be simplified back:

- `design_02_use_context`: the original snippet is right again: `Orbit([floats])` inside an unforced `use('jax')` gets a jax IC (accessors return jax arrays; integrated in jax with `method='diffrax'`, in C with the default method). The design.tex sentence about an Orbit from Python floats can stay.
- `design_03_cuda_orbit`: numpy times work with a CUDA IC under `method='torchode'`; `ts` need not be on the GPU (`method=` is still needed).
- `dfs_02_king_W0_grad`: drop the `with galpy.backend.use('jax'):` (the constructor follows the traced W0).
- `dfs_03_eddington_tracer_grad`: the derivative w.r.t. the potential's parameter runs without a backend context, so the qualifier in the Spherical-DFs claim above is no longer needed.
- `dfs_04_qdf_sampleV`: drop the `with galpy.backend.use('jax'):` (the draw follows the key); the `MWPotential2014` derivative is finite and matches a finite difference, so the Disk-DFs claim holds as written.
- `streams_02_spray_jvp` / `streams_03_spray_track_jacfwd`: `integrate_kwargs={'adjoint':'direct'}` can be dropped (galpy selects the direct adjoint under jvp/jacfwd/hessian itself, except for the jvp of a `jax.jit`-compiled function), and the progenitor can be `Orbit(ic)` from a list (the numpy and backend samplers now agree to ~1e-8; the 7e-5 came from a coarser progenitor grid in the backend sampler, not from the progenitor escaping the differentiable path).

## `design_01_rforce_jax` — sections/design.tex, line 39 (`code`)

**reformat**: Long import split with parentheses; trailing comment shortened to fit 56 characters (the device claim stays in the text and is checked by the example).

Current:

```python
from galpy.potential import MWPotential2014, evaluateRforces
R= jnp.linspace(0.5,2.,1001)
FR= evaluateRforces(MWPotential2014,R,0.) # jax array, on R's device
```

Replacement:

```python
from galpy.potential import (MWPotential2014,
                             evaluateRforces)
R= jnp.linspace(0.5,2.,1001)
FR= evaluateRforces(MWPotential2014,R,0.) # jax array
```

## `design_02_use_context` — sections/design.tex, line 49 (`code`)

**WRONG**: An Orbit does not follow the `use()` context: `Orbit([floats])` inside `use('jax')` keeps a numpy IC and is integrated in C, returning numpy arrays. A potential evaluated at Python floats does follow the context (returns a jax array), so the snippet now shows that. The sentence before it ("as when a user sets up an \orb\ instance from \python\ floats") needs to change accordingly, e.g. "as when a potential is evaluated at \python\ floats".

Current:

```python
import galpy.backend
with galpy.backend.use('jax'):
    o= Orbit([1.,0.1,1.1,0.,0.05,0.]) # IC becomes a jax array
    o.integrate(ts,MWPotential2014)   # ... integrated in jax
```

Replacement:

```python
import galpy.backend
with galpy.backend.use('jax'):
    Phi= evaluatePotentials(MWPotential2014,1.,0.) # jax
```

## `design_03_cuda_orbit` — sections/design.tex, line 56 (`code`)

**WRONG**: Without `method=`, galpy's default `symplec4_c` does not route a concrete torch IC to the in-backend solver: the orbit is integrated in C on the host (with a warning), not with torchode on the GPU. Added `method='torchode'`; long line split. Also: `ts` must be a CUDA tensor here; with numpy times torchode raises (torchode's same_device assertion; galpy moves numpy times to the IC's device for torchdiffeq but not for torchode), a galpy defect the example records.

Current:

```python
o= Orbit(torch.tensor([1.,0.1,1.1,0.,0.05,0.],device='cuda'))
o.integrate(ts,MWPotential2014) # torchode on the GPU
o.R(ts)                           # a CUDA tensor
```

Replacement:

```python
o= Orbit(torch.tensor([1.,0.1,1.1,0.,0.05,0.],
                      device='cuda'))
o.integrate(ts,MWPotential2014,method='torchode')
o.R(ts) # a CUDA tensor
```

## `design_04_namespace_evaluate` — sections/design.tex, line 92 (`code`)

**unchanged**: Fits and runs as is.

Text (no change):

```python
def _evaluate(self,R,z,phi=0.,t=0.):
    xp= get_namespace(R,z,phi,t)
    return -self._amp/xp.sqrt(R**2+z**2+self._b2)
```

## `design_05_sample_key` — sections/design.tex, line 189 (`code`)

**reformat**: The potential is built on its own line so the constructor fits; comments realigned.

Current:

```python
from galpy.backend import random as grandom
df= isotropicHernquistdf(pot=HernquistPotential(amp=2.,a=1.3))
sam= df.sample(n=10**6)                       # numpy, as before
key= grandom.key(42,backend='jax')
sam= df.sample(n=10**6,key=key)               # jax, differentiable
```

Replacement:

```python
from galpy.backend import random as grandom
hp= HernquistPotential(amp=2.,a=1.3)
df= isotropicHernquistdf(pot=hp)
sam= df.sample(n=10**6)          # numpy, as before
key= grandom.key(42,backend='jax')
sam= df.sample(n=10**6,key=key)  # jax, differentiable
```

## `design_06_jit_grad_loss` — sections/design.tex, line 204 (`code`)

**unchanged**: Fits and runs as is; the jitted gradient matches a five-point finite difference to 5e-9.

Text (no change):

```python
def loss(q):
    pot= LogarithmicHaloPotential(normalize=1.,q=q)
    o= Orbit(jnp.asarray(ic))
    o.integrate(ts,pot,method='diffrax')
    return jnp.sum((o.x(ts)-x_obs)**2)
grad_loss= jax.jit(jax.grad(loss))
```

## `potentials_01_jax_param_grad` — sections/potentials.tex, line 48 (`code`)

**unchanged**: Fits and runs as is; matches the analytic derivative to 1e-14.

Text (no change):

```python
from galpy.potential import MiyamotoNagaiPotential
def Rforce_of_a(a):
    mp= MiyamotoNagaiPotential(amp=1.,a=a,b=0.3)
    return mp.Rforce(1.2,0.1)
dFR_da= jax.grad(Rforce_of_a)(0.5)
```

## `potentials_02_torch_param_grad` — sections/potentials.tex, line 56 (`code`)

**reformat**: The `Rforce(...).backward()` line was 57 characters; split at the comma.

Current:

```python
a= torch.tensor(0.5,requires_grad=True)
mp= MiyamotoNagaiPotential(amp=1.,a=a,b=0.3)
mp.Rforce(torch.tensor(1.2),torch.tensor(0.1)).backward()
a.grad # dFR/da
```

Replacement:

```python
a= torch.tensor(0.5,requires_grad=True)
mp= MiyamotoNagaiPotential(amp=1.,a=a,b=0.3)
mp.Rforce(torch.tensor(1.2),
          torch.tensor(0.1)).backward()
a.grad # dFR/da
```

## `potentials_03_new_potential_codefig` — sections/potentials.tex, line 130 (`codefig`)

**reformat**: codefig: the `__init__` signature (57 characters) split over two lines, and the potential of the last lines built on its own line (the `integrate` call was 58 characters). Runs as is; d z(t_end)/d a matches a five-point finite difference of the C orbit in galpy's own MiyamotoNagaiPotential to 1.5e-10.

Current:

```python
from galpy.potential import Potential
from galpy.backend import get_namespace
class MNLikePotential(Potential):
    # a Miyamoto-Nagai disk written for every backend
    def __init__(self,amp=1.,a=1.,b=0.1,normalize=False):
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
    o.integrate(ts,MNLikePotential(a=a,b=0.3,normalize=1.),
                method='diffrax')
    return o.z(ts[-1])
jax.grad(final_z)(0.5)
```

Replacement:

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

**reformat**: The potential is built on its own line (`pp=`, as in the torch snippet) so the `integrate` call fits.

Current:

```python
def final_R(b):
    o= Orbit(jnp.asarray(ic))
    o.integrate(ts,PlummerPotential(amp=1.,b=b),method='diffrax')
    return o.R(ts[-1])
dR_db= jax.grad(final_R)(0.6)
```

Replacement:

```python
def final_R(b):
    o= Orbit(jnp.asarray(ic))
    pp= PlummerPotential(amp=1.,b=b)
    o.integrate(ts,pp,method='diffrax')
    return o.R(ts[-1])
dR_db= jax.grad(final_R)(0.6)
```

## `orbits_02_torchode_param_grad` — sections/orbits.tex, line 30 (`code`)

**unchanged**: Fits and runs as is (with `b` a float32 tensor, the derivative is float32-accurate: 5e-7 relative).

Text (no change):

```python
b= torch.tensor(0.6,requires_grad=True)
o= Orbit(torch.as_tensor(ic))
pp= PlummerPotential(amp=1.,b=b)
o.integrate(torch.as_tensor(ts),pp,method='torchode')
o.R(ts[-1]).backward() # b.grad = dR/db
```

## `orbits_03_cstm_jacobian` — sections/orbits.tex, line 87 (`code`)

**reformat**: The `integrate` call split and the comments shortened. Runs as is: the C state-transition-matrix path is used, and the Jacobian agrees with a finite difference of tightly converged C orbits to 3e-10.

Current:

```python
def end_point(ic):
    o= Orbit(ic)
    o.integrate(ts,MWPotential2014,method='dop853_c') # C + variational
    return jnp.stack([o.R(ts[-1]),o.z(ts[-1])])
M= jax.jacrev(end_point)(jnp.asarray(ic))  # = rows of the STM
```

Replacement:

```python
def end_point(ic):
    o= Orbit(ic)
    o.integrate(ts,MWPotential2014,
                method='dop853_c') # C + variational
    return jnp.stack([o.R(ts[-1]),o.z(ts[-1])])
M= jax.jacrev(end_point)(jnp.asarray(ic)) # STM rows
```

## `orbits_04_cstm_vmap` — sections/orbits.tex, line 95 (`code`)

**reformat**: Comment shortened to fit on one line. `jax.vmap` over the C-STM `jacrev` works and equals per-orbit `jacrev` to 1e-12.

Current:

```python
Ms= jax.vmap(jax.jacrev(end_point))(ics)   # ics.shape == (N,6)
```

Replacement:

```python
Ms= jax.vmap(jax.jacrev(end_point))(ics) # ics: (N,6)
```

## `orbits_05_dxdv_lyapunov` — sections/orbits.tex, line 155 (`code`)

**reformat**: Long calls split; the comment on the deviation vector dropped (it no longer fits).

Current:

```python
o= Orbit([0.8,0.3,0.75,0.,0.2,0.])
o.integrate_dxdv([1.,0.,0.,0.,0.,0.],ts,MWPotential2014,
                 method='dop853_c')  # orbit + deviation vector
o.getOrbit_dxdv()                   # the deviation along the orbit
o.lyapunov(ts,MWPotential2014,method='dop853_c',renorm_every=10)
```

Replacement:

```python
o= Orbit([0.8,0.3,0.75,0.,0.2,0.])
o.integrate_dxdv([1.,0.,0.,0.,0.,0.],ts,
                 MWPotential2014,method='dop853_c')
o.getOrbit_dxdv() # the deviation along the orbit
o.lyapunov(ts,MWPotential2014,method='dop853_c',
           renorm_every=10)
```

## `actionangle_01_staeckel_param_grad` — sections/actionangle.tex, line 59 (`code`)

**reformat**: The potential is built on its own line so the constructor fits on one line.

Current:

```python
def jr_of_a(a):
    aAS= actionAngleStaeckel(pot=MiyamotoNagaiPotential(a=a,b=0.3),
                             delta=0.4,c=False)
    return aAS(R,vR,vT,z,vz)[0].sum()
djr_da= jax.grad(jr_of_a)(0.5)
```

Replacement:

```python
def jr_of_a(a):
    mp= MiyamotoNagaiPotential(a=a,b=0.3)
    aAS= actionAngleStaeckel(pot=mp,delta=0.4,c=False)
    return aAS(R,vR,vT,z,vz)[0].sum()
djr_da= jax.grad(jr_of_a)(0.5)
```

## `actionangle_02_staeckel_c_jit_grad` — sections/actionangle.tex, line 95 (`code`)

**reformat**: Constructor and the jit/grad call split. Runs as is; the compiled derivative goes through a C callback and matches the c=False AD derivative to 1e-8.

Current:

```python
aAS= actionAngleStaeckel(pot=MWPotential2014,delta=0.45,c=True)
def total_jr(R):
    return aAS(R,vR,vT,z,vz)[0].sum()
djr_dR= jax.jit(jax.grad(total_jr))(jnp.asarray([1.1,0.9]))
```

Replacement:

```python
aAS= actionAngleStaeckel(pot=MWPotential2014,
                         delta=0.45,c=True)
def total_jr(R):
    return aAS(R,vR,vT,z,vz)[0].sum()
dJ= jax.jit(jax.grad(total_jr))
djr_dR= dJ(jnp.asarray([1.1,0.9]))
```

## `actionangle_03_estimate_delta` — sections/actionangle.tex, line 108 (`code`)

**reformat**: Long calls split at commas. Runs as is, and the same chain (orbit, delta, actions) runs inside `jax.jit`.

Current:

```python
delta= estimateDeltaStaeckel(MWPotential2014,o.R(ts),o.z(ts))
aAS= actionAngleStaeckel(pot=MWPotential2014,delta=delta,c=False)
jr,lz,jz= aAS(o.R(ts),o.vR(ts),o.vT(ts),o.z(ts),o.vz(ts))
```

Replacement:

```python
delta= estimateDeltaStaeckel(MWPotential2014,
                             o.R(ts),o.z(ts))
aAS= actionAngleStaeckel(pot=MWPotential2014,
                         delta=delta,c=False)
jr,lz,jz= aAS(o.R(ts),o.vR(ts),o.vT(ts),
              o.z(ts),o.vz(ts))
```

## `actionangle_04_vertical_inverse_jacfwd` — sections/actionangle.tex, line 158 (`code`)

**reformat**: The `actionAngleVerticalInverse` call (59 characters) split with a hanging indent; the disk fits on one line. The derivative matches a five-point finite difference that rebuilds the family to 3e-10 (the eager snippet takes ~50 s on one CPU core).

Current:

```python
def torus(sigma):
    disk= IsothermalDiskPotential(amp=1.,sigma=sigma)
    aAVI= actionAngleVerticalInverse(pot=disk,Es=Es,nta=128,
                                      setup_interp=True)
    return aAVI.xvFreqs(J,angles)[:2]  # (z, v_z) on the torus
z,vz= torus(0.5)
dz_dsigma,dvz_dsigma= jax.jacfwd(torus)(0.5)
```

Replacement:

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

**WRONG**: Two problems with the claim that the derivative is r_i/a to 1e-13: (1) with `rmin=0.1` the radii do not scale with `a` (the truncation radius is fixed), and the derivative differs from r_i/a by 2--4%; (2) `torch.tensor(1.,requires_grad=True)` is float32, so the derivative is float32 (5e-8). Without `rmin` and with a float64 `a`, d(sum r_i)/da = sum r_i/a to 2e-16, per particle to 1e-13. The potential is built on its own line.

Current:

```python
a= torch.tensor(1.,requires_grad=True)
with galpy.backend.use('torch'):
    df= constantbetadf(pot=HernquistPotential(amp=2.,a=a),beta=-0.2)
    key= galpy.backend.random.key(7,backend='torch')
    sam= df.sample(n=1000,rmin=0.1,key=key)
dr_da= torch.autograd.grad(sam.r().sum(),a)
```

Replacement:

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

## `dfs_02_king_W0_grad` — sections/dfs.tex, line 118 (`code`)

**WRONG**: As written, `kingdf(W0=<tracer>)` fails both eagerly and under `jax.jit` (TracerArrayConversionError): the constructor evaluates its potential at Python floats (`self._pot(self.rt,0.)`), which go to numpy and meet the traced W0. Inside `galpy.backend.use('jax')` both work. The eager line was dropped because it takes ~2 min on a CPU (the jitted one ~1 min, nearly all compilation); the jitted derivative matches a five-point finite difference of the numpy model to 2e-7.

Current:

```python
def sigmar(W0):
    return kingdf(W0=W0,M=2.3,rt=1.4).sigmar(jnp.asarray(0.5))
jax.grad(sigmar)(3.)
jax.jit(jax.grad(sigmar))(3.)  # the ODE is solved by diffrax here
```

Replacement:

```python
def sigmar(W0):
    with galpy.backend.use('jax'):
        kdf= kingdf(W0=W0,M=2.3,rt=1.4)
        return kdf.sigmar(jnp.asarray(0.5))
dsig= jax.jit(jax.grad(sigmar)) # ODE by diffrax
dsig(3.)
```

## `dfs_03_eddington_tracer_grad` — sections/dfs.tex, line 126 (`code`)

**reformat**: The tracer is built on its own line. Runs as is. But the text's claim that the DF is differentiable w.r.t. the parameters of BOTH densities only holds inside `galpy.backend.use('jax')`: d/d(potential parameter), e.g. NFW `a` or `amp`, fails without it (the constructor evaluates the potential at r=0 as a Python float). With the context, it matches a finite difference to 4e-10.

Current:

```python
def dispersion(a_tracer):
    df= eddingtondf(pot=NFWPotential(amp=1.,a=2.),
                    denspot=HernquistPotential(amp=0.1,a=a_tracer))
    return df.sigmar(jnp.asarray(1.))
jax.grad(dispersion)(0.7)
```

Replacement:

```python
def dispersion(a_tracer):
    hp= HernquistPotential(amp=0.1,a=a_tracer)
    df= eddingtondf(pot=NFWPotential(amp=1.,a=2.),
                    denspot=hp)
    return df.sigmar(jnp.asarray(1.))
jax.grad(dispersion)(0.7)
```

## `dfs_04_qdf_sampleV` — sections/dfs.tex, line 156 (`code`)

**WRONG**: `qdf.sampleV(...,key=<jax key>)` on a qdf built from Python floats raises a TypeError (numpy and jax arrays mixed in the inverse-CDF draw) unless it runs inside `galpy.backend.use('jax')`; wrapped. The comment "differentiable in qdf's parameters" was removed: for `MWPotential2014` the derivative of the sample w.r.t. any DF parameter (hr, sr, sz, hsr, hsz) is NaN (jax_debug_nans points at `_cumcdf` in `_sampleV_icdf`); for `MWPotential` it is finite and matches a finite difference. This is a galpy defect; the text's claim needs a caveat or a fix in galpy.

Current:

```python
qdf= quasiisothermaldf(1./3.,0.2,0.1,1.,1.,pot=MWPotential2014,aA=aAS)
vs= qdf.sampleV(0.9,0.05,n=10**4,key=grandom.key(0,backend='jax'))
sigma_R= jnp.std(vs[:,0])   # differentiable in qdf's parameters
```

Replacement:

```python
qdf= quasiisothermaldf(1./3.,0.2,0.1,1.,1.,
                       pot=MWPotential2014,aA=aAS)
with galpy.backend.use('jax'):
    vs= qdf.sampleV(0.9,0.05,n=10**4,
                    key=grandom.key(0,backend='jax'))
sigma_R= jnp.std(vs[:,0])
```

## `streams_01_streamdf` — sections/streams.tex, line 38 (`code`)

**WRONG**: `time_in_Gyr()` takes `(vo,ro)`: now `time_in_Gyr(220.,8.)`. Long lines split; the `integrate_method` with a hanging indent.

Current:

```python
aAI= actionAngleIsochroneApprox(pot=lp,b=0.8,integrate_method='diffrax')
sdf= streamdf(sigv/220.,progenitor=Orbit(jnp.asarray(prog)),pot=lp,aA=aAI,
              leading=True,nTrackChunks=11,tdisrupt=4.5/time_in_Gyr())
sdf.sample(n=10**4,key=grandom.key(3,backend='jax'))
```

Replacement:

```python
aAI= actionAngleIsochroneApprox(pot=lp,b=0.8,
        integrate_method='diffrax')
sdf= streamdf(sigv/220.,pot=lp,aA=aAI,
              progenitor=Orbit(jnp.asarray(prog)),
              tdisrupt=4.5/time_in_Gyr(220.,8.),
              leading=True,nTrackChunks=11)
sdf.sample(n=10**4,key=grandom.key(3,backend='jax'))
```

## `streams_02_spray_jvp` — sections/streams.tex, line 99 (`code`)

**WRONG**: (1) `jax.jvp` through the default (checkpointed) diffrax adjoint raises "can't apply forward-mode autodiff (jvp) to a custom_vjp function": `integrate_kwargs={'adjoint':'direct'}` is required. (2) With `progenitor=Orbit(ic)` from a list the derivative disagrees with a finite difference of the same sample at 7e-5 (and the jvp primal with a plain call at 1.5e-5); with a jax progenitor, `Orbit(jnp.asarray(ic))`, it agrees to ~1e-6 as the text says. (3) `time_in_Gyr()` needs `(vo,ro)`; replaced by `td` as in the next snippet. The output is now unpacked as `xv,dxv` (shorter, and it is a sample, not a track).

Current:

```python
key= galpy.backend.random.key(1,backend='jax')
def leading_arm(q):
    lp= LogarithmicHaloPotential(normalize=1.,q=q)
    spdf= fardal15spraydf(mass,progenitor=Orbit(ic),pot=lp,
                          tdisrupt=4.5/time_in_Gyr())
    return spdf.sample(n=1500,return_orbit=False,
                       tail='leading',key=key)
track_and_dtrack= jax.jvp(leading_arm,(q0,),(1.,))
```

Replacement:

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

## `streams_03_spray_track_jacfwd` — sections/streams.tex, line 113 (`code`)

**WRONG**: `streamTrack()` returns a `StreamTrack` object, which is not a jax pytree, so `jax.jacfwd(track)` cannot differentiate it; the function now returns the track evaluated at fixed progenitor times, `tr.x(tps)`. Same forward-mode fixes as the previous snippet (direct adjoint, jax progenitor). Caveat: the fitted track is not a smooth function of q (the closest-node assignment and the GCV smoothing level are discrete choices, frozen in the derivative but re-made at q+-h): a finite difference does not converge onto `jacfwd` (relative differences 1--3 for h=1e-3..1e-5, also with fixed smoothing and ntp). The example records this rather than checking a value.

Current:

```python
def track(q):
    lp= LogarithmicHaloPotential(normalize=1.,q=q)
    spdf= fardal15spraydf(mass,progenitor=Orbit(ic),pot=lp,tdisrupt=td)
    return spdf.streamTrack(n=1500,key=key)
track(0.9); jax.jacfwd(track)(0.9)
```

Replacement:

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

## `examples_01_hvs_cstm` — sections/examples.tex, line 34 (`code`)

**reformat**: The coordinate-order comment moved to its own line and the `jacrev` call split. Runs as is: the C state-transition-matrix path is used, through an Orbit initialized from (ra,dec,d,pmra,pmdec,vlos), and the gradient matches a five-point finite difference to 1e-6.

Current:

```python
def closest_approach_ic(vxvv):  # (ra,dec,d,pmra,pmdec,vlos)
    o= Orbit(vxvv,radec=True,ro=8.,vo=220.)
    o.integrate(-ts,pot,method='dop853_c')  # C + variational eqs.
    return jnp.min(o.r(-ts))
J= jax.jacrev(closest_approach_ic)(jnp.asarray(vxvv_obs))
```

Replacement:

```python
def closest_approach_ic(vxvv):
    # vxvv: (ra,dec,d,pmra,pmdec,vlos)
    o= Orbit(vxvv,radec=True,ro=8.,vo=220.)
    o.integrate(-ts,pot,method='dop853_c') # C + STM
    return jnp.min(o.r(-ts))
J= jax.jacrev(closest_approach_ic)(
    jnp.asarray(vxvv_obs))
```

## `examples_02_hvs_q_fit` — sections/examples.tex, line 43 (`code`)

**reformat**: Only the potential needed splitting. `adam_descent` (a dozen lines of optax in the example) recovers q=0.8485 for a mock generated with q=0.85.

Current:

```python
def closest_approach(q):
    pot= LogarithmicHaloPotential(normalize=1.,core=0.2,q=q)
    o= Orbit(jnp.asarray(vxvv),radec=True,ro=8.,vo=220.)
    o.integrate(-ts,pot,method='diffrax')
    return jnp.min(o.r(-ts))
objective= jax.jit(jax.value_and_grad(closest_approach))
q= adam_descent(objective,q0=1.1) # a few lines of optax
```

Replacement:

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

## `examples_03_neural_inverse` — sections/examples.tex, line 98 (`code`)

**reformat**: Comments shortened and the return split over two lines in parentheses. Runs as is (with a small MLP class with an `apply(params,J,theta)` method); the gradient matches a finite difference along a random direction in weight space to 1e-7.

Current:

```python
def loss(params,J,theta):
    z,vz= net.apply(params,J,theta)          # the network's inverse
    Jp,_,thetap= aAV.actionsFreqsAngles(z,vz) # galpy's forward map
    return jnp.mean((Jp-J)**2/J**2)+jnp.mean(wrap(thetap-theta)**2)
grads= jax.jit(jax.grad(loss))(params,J,theta)
```

Replacement:

```python
def loss(params,J,theta):
    z,vz= net.apply(params,J,theta) # the network
    Jp,_,thetap= aAV.actionsFreqsAngles(z,vz) # galpy
    return (jnp.mean((Jp-J)**2/J**2)
            +jnp.mean(wrap(thetap-theta)**2))
grads= jax.jit(jax.grad(loss))(params,J,theta)
```
