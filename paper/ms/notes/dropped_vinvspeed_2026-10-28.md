# Dropped from Section 5.4 on 2026-10-28 (kept for the inverse action-angle paper)

The paragraph and figure (gV2_vertical_inverse_speed) below were removed from the backends paper at JB's request; the figure script and data remain under paper/figures/gV2_vertical_inverse_speed/.

```latex
\figurename~\ref{fig:vinvspeed} shows the speed of the one-dimensional
inverse transformation. Evaluating the torus map at $N$ angles
is fastest with \numpy\ at every $N$ on one CPU core: compiled \jax\
ties it for ten angles and is two to eight times slower for more angles
(49\,s against 10.9\,s at $N=10^6$), while for many tori the compiled
\texttt{vmap} over tori is about 1.5 times faster than the \numpy\ loop
over tori. On a GPU the picture reverses: on the H100 the evaluation
at $N=10^6$ takes 11\,ms, about a thousand times faster than \numpy. The
backends find the angle with the same safeguarded Newton iteration as
\numpy, started at the requested angle. Constructing the map for a
family of $n_E$ energies is faster on every backend than on \numpy, by
a factor of three to ten at $n_E=256$, because the batched Gauss--Newton
solve replaces a loop over tori; on the H100 the compiled construction
takes 4 to 6 milliseconds whatever the size of the family. These numbers are for
the development machine and the H100 and should be read as orders of
magnitude.

\begin{figure*}[t!]
  \includegraphics[width=\textwidth,clip=]{../figures/gV2_vertical_inverse_speed/gV2_vertical_inverse_speed.pdf}
  \caption{Speed of the one-dimensional inverse action--angle
    transformation, \texttt{actionAngleVerticalInverse}, for an
    isothermal disk. (a) Wall time to evaluate positions, velocities, and
    frequencies at $N$ angles on one torus (or, dash-dotted, at 16 angles
    on each of $N/16$ tori) with \numpy\ and on the backends on a single
    CPU core, the TITAN~Xp, and the H100; compilation time is excluded.
    (b) Time to construct the map for a family of $n_E$ energies, warm
    (filled markers) and on the first call (open markers), which for
    eager \jax\ includes compiling every operation for the new
    shapes.}\label{fig:vinvspeed}
\end{figure*}
```
