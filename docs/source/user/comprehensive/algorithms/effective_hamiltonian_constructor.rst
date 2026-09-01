Effective Hamiltonian construction
==================================

The :class:`~qdk_chemistry.algorithms.EffectiveHamiltonianConstructor` algorithm in QDK/Chemistry defines a common interface for downfolding a Hamiltonian from a larger orbital window into a smaller target space.
Following QDK/Chemistry's :doc:`algorithm design principles <../design/index>`, it takes a reference :class:`~qdk_chemistry.data.Wavefunction`, an input :class:`~qdk_chemistry.data.Hamiltonian`, and target-space orbital indices as input and produces an effective :class:`~qdk_chemistry.data.Hamiltonian` as output.
The interface leaves the downfolding approximation and method-specific diagnostics to the concrete implementation.

Overview
--------

Let the orbital window :math:`W` be partitioned into a target space :math:`P` and an external space :math:`Q`:

.. math::

   W = P \cup Q,
   \qquad
   P \cap Q = \varnothing.

Using :math:`\hat P` and :math:`\hat Q` for the corresponding projectors, the full-window Hamiltonian has the block structure

.. math::

   H_W =
   \begin{pmatrix}
      H_{PP} & H_{PQ} \\
      H_{QP} & H_{QQ}
   \end{pmatrix},
   \qquad
   H_{PP} = \hat P H_W \hat P.

An implementation accounts for the influence of :math:`Q` while constructing an effective Hamiltonian that acts only in :math:`P`:

.. math::

   \mathcal{D}\!\left(\lvert \Psi_{\mathrm{ref}} \rangle, H_W, P\right)
   = H_{\mathrm{eff}}^{P},
   \qquad
   H_{\mathrm{eff}}^{P} = \hat P H_{\mathrm{eff}}^{P} \hat P,

where :math:`\mathcal{D}` denotes the selected downfolding implementation.
The interface does not prescribe how the coupling blocks :math:`H_{PQ}` and :math:`H_{QP}` or the external block :math:`H_{QQ}` contribute to :math:`H_{\mathrm{eff}}^{P}`.

Using the EffectiveHamiltonianConstructor
-----------------------------------------

The ``run`` method takes a reference wavefunction, a full-window Hamiltonian, and the target-space indices and returns the effective Hamiltonian acting in :math:`P`.

Input requirements
~~~~~~~~~~~~~~~~~~

The :class:`~qdk_chemistry.algorithms.EffectiveHamiltonianConstructor` requires the following inputs:

Reference wavefunction
   A :class:`~qdk_chemistry.data.Wavefunction` that defines the reference state used by the downfolding method.

Input Hamiltonian
   A :class:`~qdk_chemistry.data.Hamiltonian` expressed over the complete orbital window :math:`W`.

Target-space indices
   A :class:`~qdk_chemistry.data.symmetry.SymmetryBlockedIndexSet` containing the indices of :math:`P`.
   These are absolute molecular-orbital indices, drawn from the same index universe as ``Orbitals.active_indices()``, and must lie within the active space of :math:`W`.

Output contract
~~~~~~~~~~~~~~~

The returned :class:`~qdk_chemistry.data.Hamiltonian` is expressed over :math:`P` and satisfies:

- its orbitals have ``active_indices()`` equal to the requested target-space indices;
- its orbitals classify fully occupied orbitals of :math:`Q = W \setminus P` as inactive and unoccupied orbitals of :math:`Q` as virtual, while preserving the input Hamiltonian's inactive orbitals;
- its inactive Fock matrix, when present, is consistent with the output inactive orbitals and may therefore differ from the input Hamiltonian's inactive Fock matrix;
- the scalar shift from folding in :math:`Q` is added to the constant (zero-body) energy term, and the remaining :math:`Q` contribution is folded into the integrals.

The occupied/virtual partition of :math:`Q` is method dependent and is determined by the concrete implementation from its reference state.
Classifying an orbital of :math:`Q` as virtual describes its occupation in the output orbital metadata; it does not place that orbital in the effective active space.
Consumers of an effective Hamiltonian should not attempt to re-correlate :math:`Q`, since its contribution is already folded in.

Input validation is opt-in.
The ``run`` method does not validate its arguments; each concrete implementation decides whether to check the space contract :math:`P \subseteq W_H` and :math:`W_{\mathrm{ref}} \subseteq W_H` before computing.

The base interface defines no common settings.
Concrete implementations can expose method-specific configuration through the ``settings()`` object.
See :doc:`Settings <settings>` for a general treatment of algorithm settings in QDK/Chemistry.

Available implementations
-------------------------

DUCC
~~~~

.. rubric:: Factory name: ``"ducc"`` (default)

The double unitary coupled-cluster (DUCC) implementation constructs a P-space Hamiltonian from a full-space Hamiltonian and coupled-cluster amplitudes.
It evaluates a truncated Baker-Campbell-Hausdorff transformation

.. math::

   H_{\mathrm{eff}}^P = \hat P e^{-\sigma} H_W e^{\sigma} \hat P,
   \qquad
   \sigma = T_{\mathrm{ext}} - T_{\mathrm{ext}}^\dagger,

where :math:`T_{\mathrm{ext}}` excludes amplitudes whose indices are all in :math:`P`.
The input Hamiltonian must be Hermitian and span the full orbital window, but it can use any Hamiltonian container that provides four-center integrals.
The reference wavefunction must contain real coupled-cluster amplitudes built on a single determinant, use the same full-space orbitals as the Hamiltonian, and occupy a contiguous prefix of orbitals in each spin channel.
The target P-space must be non-empty in each spin channel.

The ``ducc_level`` setting selects the perturbatively consistent A(1), A(4), and A(7) approximations of :cite:`Bauman2022DUCC`.
In all three schemes, the transformed operator is restricted to the target space and truncated to its scalar, one-body, and two-body components.
With :math:`H_N` the normal-ordered Hamiltonian, :math:`F_N` its one-body Fock component, and :math:`\sigma = \sigma_{\mathrm{ext}}`, the retained terms are

.. math::

   \begin{aligned}
   \mathrm{level\ 0\ (A(1)):}\quad
      H_{\mathrm{eff}}^P &\simeq \hat P H \hat P, \\
   \mathrm{level\ 1\ (A(4)):}\quad
      H_{\mathrm{eff}}^P &\simeq \hat P \left(H + [H_N,\sigma]
      + \frac{1}{2} [[F_N,\sigma],\sigma]\right) \hat P, \\
   \mathrm{level\ 2\ (A(7)):}\quad
      H_{\mathrm{eff}}^P &\simeq \hat P \left(H + [H_N,\sigma]
      + \frac{1}{2} [[H_N,\sigma],\sigma]
      + \frac{1}{6} [[[F_N,\sigma],\sigma],\sigma]\right) \hat P.
   \end{aligned}

Level 1 is the second-order perturbatively consistent expansion in Eq. (62) of :cite:`Bauman2019DUCC`; level 2 retains the full double commutator and the Fock-only triple commutator required for third-order consistency.

.. rubric:: Settings

.. csv-table::
    :header: "Setting", "Type", "Default", "Description"
    :widths: 25, 15, 15, 45

    "``ducc_level``", "int", "``2``", "Approximation scheme: 0 selects A(1), 1 selects A(4), and 2 selects A(7)"

Related classes
---------------

- :class:`~qdk_chemistry.data.Wavefunction`: Provides the reference state
- :class:`~qdk_chemistry.data.Hamiltonian`: Represents the full-window input and target-space output Hamiltonians
- :class:`~qdk_chemistry.data.symmetry.SymmetryBlockedIndexSet`: Identifies the target orbital space while preserving symmetry blocks
- :doc:`HamiltonianConstructor <hamiltonian_constructor>`: Builds the input Hamiltonian over the orbital window

Further reading
---------------

- :doc:`ActiveSpaceSelector <active_space>`: Identifies orbital subspaces for correlated calculations
- :doc:`Settings <settings>`: Configures algorithm implementations
- :doc:`Factory Pattern <factory_pattern>`: Discovers, registers, and creates algorithm implementations
