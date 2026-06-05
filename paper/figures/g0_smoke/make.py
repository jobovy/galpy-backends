"""G0 harness smoke test: a parabola.

Exercises the @figure decorator, run/verify, and the manifest. Not a paper figure
itself -- just proves the reproducibility machinery works end-to-end.
"""

import matplotlib.pyplot as plt
import numpy

from paper.figures._harness.figure import figure


@figure(
    name="g0_smoke",
    track="G",
    seed=0,
    description="harness smoke test (parabola) -- validates run/verify/manifest",
)
def make():
    x = numpy.linspace(-3.0, 3.0, 200)
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    ax.plot(x, x**2, label=r"$y=x^2$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("G0 harness smoke test")
    ax.legend()
    fig.tight_layout()
    return fig
