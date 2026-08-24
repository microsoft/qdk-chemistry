Choosing the active space
#########################

.. admonition:: Chapter focus
   :class: chapter-focus

   Which electrons and orbitals must the calculation treat explicitly?

Learning objectives
===================

After completing this chapter, you will be able to:

- Explain why strongly correlated systems require more than one determinant.
- Distinguish inactive, active, and virtual orbitals.
- Select a valence active space with :term:`QDK`/Chemistry.
- Explain how natural orbitals and reduced density matrices support active-space selection.
- Explain how orbital entropies can refine an active-space choice.
- Evaluate the tradeoff between active-space accuracy and problem size.

.. admonition:: Lab notebook assignment
   :class: lab-notebook-assignment

   Complete :ref:`lab-notebook-active-space`.
   Record the evidence used to select the active space, not only the final orbital and electron counts.
   Explain what the energy comparison establishes within the chosen basis and identify the energy that will serve as the algorithmic reference.

Example download
====================

First download :download:`tutorial_orbital_coordinates.py <../../_static/examples/python/tutorial_orbital_coordinates.py>`, which contains the reusable numerical machinery for choosing coordinates inside degenerate orbital subspaces.
Then download :download:`tutorial_choose_active_space.py <../../_static/examples/python/tutorial_choose_active_space.py>` and :download:`tutorial_choose_active_space.ipynb <../../_static/examples/python/tutorial_choose_active_space.ipynb>`, and save all three files in your tutorial working directory.
Open all three files in Visual Studio Code and review the complete scripts, including imports and setup code omitted from the excerpts below.
The script resumes the stretched N\ :sub:`2` workflow from :doc:`Describing the molecule <02_describing_the_molecule>` before constructing and refining the correlated model.
Unlike the earlier examples, this script organizes the calculation into importable functions so that the command-line example and interactive Jupyter notebook use the same tested chemistry workflow rather than duplicate it.

The limits of one determinant
=======================================

As :ref:`the previous chapter explains <tutorial-hartree-fock-wavefunction>`, the `Hartree--Fock method <https://en.wikipedia.org/wiki/Hartree%E2%80%93Fock_method>`_ restricts the `wavefunction <https://en.wikipedia.org/wiki/Wave_function>`_ to one optimized `Slater determinant <https://en.wikipedia.org/wiki/Slater_determinant>`_.
This determinant represents one `electron configuration <https://en.wikipedia.org/wiki/Electron_configuration>`_, a pattern of occupied spin orbitals introduced in :ref:`Orbitals and determinants <tutorial-orbitals-and-determinants>`.
This description is often a useful starting point near an equilibrium geometry, where one configuration dominates the ground-state wavefunction, as discussed for N\ :sub:`2` in :ref:`Specify the molecular system <tutorial-molecular-system>` chapter.
Stretching a chemical bond can make several configurations similar in energy because electrons can no longer be assigned adequately to one fixed pattern of occupied and unoccupied `molecular orbitals <https://en.wikipedia.org/wiki/Molecular_orbital_theory>`_.
The need to combine these multiple important configurations is called `static correlation <https://en.wikipedia.org/wiki/Electronic_correlation>`_.
The stretched N\ :sub:`2` geometry has been selected to demonstrate this regime.
The correlated calculations below evaluate static correlation by constructing a multi-determinant wavefunction and measuring orbital-occupation entropies.

A `configuration interaction <https://en.wikipedia.org/wiki/Configuration_interaction>`_ (:term:`CI`) calculation addresses this limitation by calculating a wavefunction expanded in multiple Slater determinants:

.. math::

   \vert \Psi \rangle = \sum_I c_I \vert \Phi_I \rangle,

where :math:`\vert \Phi_I \rangle` is determinant :math:`I` and :math:`c_I` is its coefficient.
Every determinant :math:`\Phi_I`, including :math:`\Phi_{\mathrm{HF}}`, is constructed from one allowed choice of occupied spin orbitals.
Different choices represent different electron configurations.
Full configuration interaction (:term:`FCI`) includes every determinant consistent with a chosen finite orbital basis and fixed :math:`(n_\alpha,n_\beta)` sector.
It therefore gives the exact eigenvalues of the finite-basis Hamiltonian in that sector, up to numerical solver tolerance.
For this 14-electron calculation in 28 ``cc-pvdz`` spatial orbitals, the :math:`(n_\alpha,n_\beta)=(7,7)` sector contains

.. math::

   \binom{28}{7}\binom{28}{7}
   = 1{,}401{,}950{,}721{,}600
   \approx 1.4\times 10^{12}

determinants.
At a fixed electron-to-orbital ratio, this count grows exponentially with the number of orbitals, making full-basis :term:`FCI` impractical for large systems.
An `active-space model <https://en.wikipedia.org/wiki/Complete_active_space>`_ controls that cost by allowing occupations to vary only among selected orbitals, trading some model accuracy for a smaller determinant space.
If that model contains :math:`n_o` active spatial orbitals, :math:`n_\alpha` active :math:`\alpha` electrons, and :math:`n_\beta` active :math:`\beta` electrons, the choices of occupied :math:`\alpha` and :math:`\beta` spin orbitals give

.. _tutorial-determinant-count:

.. math::

   N_{\mathrm{det}} = \binom{n_o}{n_\alpha}\binom{n_o}{n_\beta}.

.. _tutorial-active-space-definition:

The active space
=======================

.. _tutorial-orbital-partition:

.. figure:: /_static/diagrams/tutorial_qpe_orbital_partition.svg
   :alt: The spatial molecular orbitals are partitioned into inactive orbitals that remain doubly occupied and contribute to the core energy, active orbitals whose occupations vary among determinants and whose correlation is treated explicitly, and virtual orbitals that remain empty and are excluded from the correlated calculation.
   :align: center

   An active-space calculation varies occupations only among the active orbitals; inactive and virtual occupations remain fixed across determinants.

.. admonition:: Why can an inactive orbital still contribute to the molecular energy?
   :class: quiz-question
   :collapsible: closed

   An inactive spatial orbital remains doubly occupied in every determinant.
   Its electrons and their interactions contribute to the core part of the active-space Hamiltonian even though the calculation does not vary their occupations.

A complete active space containing :math:`n_e` active electrons in :math:`n_o` active spatial orbitals is written :term:`CAS`\ :math:`(n_e,n_o)`.
Complete active space configuration interaction (:term:`CASCI`) forms every determinant consistent with those active electron and orbital counts while keeping the molecular orbitals fixed.
Unlike complete active space self-consistent field (:term:`CASSCF`), :term:`CASCI` does not reoptimize the orbitals.

A useful first choice is a generous valence space containing orbitals on both sides of the occupied--virtual boundary.
The :func:`~qdk_chemistry.utils.compute_valence_space_parameters` function determines the numbers of valence electrons and valence spatial orbitals from the Hartree--Fock wavefunction and molecular charge.
The :ref:`qdk_valence selector <qdk-valence-active-space>` uses those numbers to construct an initial active space from orbitals near the :term:`HOMO`--:term:`LUMO` gap.
The highest occupied molecular orbital (:term:`HOMO`) and lowest unoccupied molecular orbital (:term:`LUMO`) define the boundary between occupied and virtual orbitals in the Hartree--Fock reference.
Orbitals near this boundary are the most accessible when low-energy configurations redistribute electrons, so a valence window around the gap is a useful generous starting space for the correlated calculation:

.. literalinclude:: ../../_static/examples/python/tutorial_choose_active_space.py
   :language: python
   :dedent: 4
   :start-after: # start-cell-valence-space
   :end-before: # end-cell-valence-space

For this restricted calculation, matching :math:`\alpha` and :math:`\beta` channels describe the same spatial orbitals, so the code reads one channel and counts each spatial orbital once.
The script reports the resulting active electron and orbital counts and the zero-based indices of the active orbitals.
Use these values with the total number of ``cc-pvdz`` molecular orbitals from :doc:`Describing the molecule <02_describing_the_molecule>` to determine the initial partitioning of inactive, active, and virtual orbitals.

A correlated active-space wavefunction
==============================================

The active-space selector labels orbitals but does not determine how strongly each orbital participates in correlation.
That evidence must come from a correlated wavefunction.
The script constructs the molecular Hamiltonian in the initial valence space and solves it with the :ref:`MACIS CASCI implementation <macis-cas>` :cite:`Williams-Young2023`:

.. literalinclude:: ../../_static/examples/python/tutorial_choose_active_space.py
   :language: python
   :dedent: 4
   :start-after: # start-cell-initial-casci
   :end-before: # end-cell-initial-casci

Using the determinant-count formula above, the initial valence space in this example is small enough to include every determinant rather than approximating the wavefunction with a selected subset.
Larger active-space studies often use selected :term:`CI` (:term:`SCI`) to obtain approximate active-space diagnostics at lower cost, but that additional approximation is unnecessary here.

The ``calculate_one_rdm`` and ``calculate_two_rdm`` settings request the one- and two-particle reduced density matrices (:term:`RDMs <RDM>`).
An :term:`RDM` summarizes the parts of the many-electron wavefunction needed to describe one- or two-particle properties.
The spin-resolved one-particle :term:`RDM` tracks :math:`\alpha` and :math:`\beta` occupations separately, and its diagonal gives the expected occupation of each spin orbital.
A corresponding diagonal element of the two-particle :term:`RDM` gives the joint occupation of a pair of spin orbitals.
Each determinant assigns every spatial orbital one of four *local occupation states*: empty, occupied by one :math:`\alpha` electron, occupied by one :math:`\beta` electron, or doubly occupied.
Here, a local occupation state describes only one orbital, not the complete electronic state of the molecule.
For these occupation quantities, determinant :math:`I` contributes with weight :math:`\lvert c_I\rvert^2`, the squared magnitude of its coefficient in the correlated wavefunction.
Together, the :term:`RDM` elements collect these contributions into the probabilities of the four local occupation states.
If the important determinants give an orbital the same occupation, one probability dominates; if they assign different occupations, the probabilities spread among several local states.

Natural-orbital transformation
==============================

Different sets of molecular orbitals can describe the same active space.
Changing to natural orbitals changes the individual orbital shapes but not the exact :term:`CASCI` energy for that active space.
Orbital-resolved quantities, however, can change because they describe the chosen orbital representation.

Natural orbitals diagonalize the one-particle :term:`RDM` :cite:`Lowdin1956`.
Their eigenvalues are natural-orbital occupation numbers between zero and two for spatial orbitals.
In this basis, the off-diagonal elements vanish, so each occupation number is associated directly with one natural orbital rather than being mixed among several orbitals.
Occupations near two identify nearly doubly occupied orbitals, occupations near zero identify nearly empty orbitals, and fractional occupations can reveal orbitals that require multiple electron configurations.
Natural orbitals provide a useful convention in which the one-particle occupations are directly associated with individual orbitals before applying the orbital-resolved entropy criterion.

The supported :ref:`qdk_natural_orbitals transformation <localizer-qdk-natural-orbitals>` uses the one-particle :term:`RDM` from the initial :term:`CASCI` wavefunction.
It rotates the active orbitals into the natural-orbital representation described above.
The script then rebuilds and resolves the initial valence-space Hamiltonian so that both :term:`RDMs <RDM>` and the orbital diagnostics are expressed consistently in the natural-orbital representation:

.. literalinclude:: ../../_static/examples/python/tutorial_choose_active_space.py
   :language: python
   :dedent: 4
   :start-after: # start-cell-natural-orbitals
   :end-before: # end-cell-natural-orbitals

The two :term:`CASCI` energies should agree to the displayed precision because both calculations span the same complete active space.
The second calculation is needed for the orbital-resolved selection evidence, not to lower the energy.

Active-space refinement with orbital entropies
==============================================

The single-orbital entropy used here is the `von Neumann entropy <https://en.wikipedia.org/wiki/Von_Neumann_entropy>`_ of the reduced density matrix for one spatial orbital :cite:`Boguslawski2015`.
Its eigenvalues are the probabilities :math:`\omega_{a,i}` of the four local occupation states, so the entropy has the Shannon form

.. math::

   s_i^{(1)} = -\sum_{a=1}^{4} \omega_{a,i}\ln\omega_{a,i}.

These probabilities come from diagonal elements of the spin-resolved one- and two-particle :term:`RDMs <RDM>`.
If :math:`n_{i\alpha}` and :math:`n_{i\beta}` are the one-particle occupations and :math:`d_i` is the *double-occupancy probability*—the probability that the :math:`\alpha` and :math:`\beta` spin orbitals belonging to spatial orbital :math:`i` are occupied simultaneously—then

.. math::

   \begin{aligned}
   \omega_{\mathrm{empty},i} &= 1-n_{i\alpha}-n_{i\beta}+d_i, \\
   \omega_{\alpha,i} &= n_{i\alpha}-d_i, \\
   \omega_{\beta,i} &= n_{i\beta}-d_i, \\
   \omega_{\mathrm{double},i} &= d_i.
   \end{aligned}

The one-particle occupation :math:`n_{i\alpha}` includes both the :math:`\alpha`-only and doubly occupied cases, so subtracting :math:`d_i` isolates the :math:`\alpha`-only probability; the same reasoning gives the :math:`\beta`-only probability.
The empty probability is the remainder after accounting for either spin occupation, with :math:`d_i` added back because double occupation was subtracted twice.
The four probabilities therefore sum to one.

An entropy near zero means that one local occupation state consistently dominates.
A larger entropy means that the orbital changes among several local occupation states across the important determinants.
In other words, the important determinants assign different occupations to this orbital together with corresponding occupation differences elsewhere in the active space.
This correlated variation makes the orbital a stronger candidate for explicit treatment.
These high-entropy orbitals carry the strongest static-correlation signal because their occupations vary among the important determinants.
Freezing a high-entropy orbital would prevent its occupation from changing with the occupations of the other orbitals and would therefore remove an important part of the multi-configurational wavefunction.
By contrast, a low-entropy orbital remains close to one local occupation state and is a better candidate to freeze as inactive or virtual.
:term:`QDK`/Chemistry evaluates these probabilities and entropies from the :term:`RDMs <RDM>` stored in the :term:`CASCI` wavefunction.
Automated active-space selection (autoCAS) uses orbital entropies to identify which orbitals should remain active.
The resulting data flow is therefore: the correlated wavefunction determines the local-state probabilities, those probabilities determine one entropy for each orbital, and autoCAS uses the entropies to select the orbitals in the active space.

.. admonition:: Why does autoCAS require a correlated calculation before it can select orbitals?
   :class: quiz-question
   :collapsible: closed

   The selector uses single-orbital entropies derived from local occupation probabilities.
   Those probabilities require one- and two-particle :term:`RDMs <RDM>` from a correlated wavefunction; a Hartree--Fock determinant alone does not provide the required correlation evidence.

The QDK/Chemistry :ref:`qdk_autocas_eos selector <qdk-autocas-eos>` sorts the orbital entropies and selects a high-entropy group separated from the remaining orbitals by a sufficiently large gap :cite:`Stein2016,Stein2019`.
The thresholds are configurable; see :doc:`Active-space selection <../../user/comprehensive/algorithms/active_space>` for their defaults and use with less clearly separated entropy values.
The selector then repartitions the orbitals according to the selected group:

.. literalinclude:: ../../_static/examples/python/tutorial_choose_active_space.py
   :language: python
   :dedent: 4
   :start-after: # start-cell-refine
   :end-before: # end-cell-refine

.. The figure intentionally reveals the selected group before the later
   observed-result question. Showing the entropy gap is necessary to teach how
   autoCAS makes the selection, which outweighs preserving the answer as a surprise.

.. figure:: /_static/diagrams/tutorial_qpe_orbital_entropy.png
   :alt: Entropy-ranked candidate orbitals. Selected orbitals 8, 7, 5, 6, 9, and 4 have entropies of approximately 0.966, 0.966, 0.964, 0.964, 0.554, and 0.548. Excluded orbitals 3 and 2 have entropies of approximately 0.030 and 0.022. A dashed vertical cut separates the sixth and seventh entropy ranks.
   :align: center
   :width: 90%

   Candidate natural orbitals sorted by decreasing single-orbital entropy. autoCAS retains the high-entropy group to the left of the dashed cut.

The asterisks in the script output identify the selected orbitals.
The selected high-entropy group determines the refined active space.
Equal natural-orbital occupations can leave the corresponding orbital vectors free to rotate within a degenerate subspace.
The script chooses a reproducible representation within each selected degenerate block by coordinate-minimizing the mapped Hamiltonian coefficient norm :math:`\lambda=\sum_\ell\lvert h_\ell\rvert`, without changing the orbital subspace or its exact :term:`CASCI` energy.
Among the unselected orbitals, those below the occupied--virtual boundary of the reference determinant become inactive, while those above the boundary become virtual.
Freezing these low-entropy orbitals is still an approximation because low entropy does not mean that their correlation contribution is exactly zero, so the energy comparison below measures part of its cost.
Their entropies are small rather than exactly zero, and allowing excitations involving them can still lower the correlated energy.

.. _tutorial-selected-space-reference:

The algorithmic reference
=================================

The script finishes by solving the refined active-space Hamiltonian with :term:`CASCI`:

.. literalinclude:: ../../_static/examples/python/tutorial_choose_active_space.py
   :language: python
   :dedent: 4
   :start-after: # start-cell-final-casci
   :end-before: # end-cell-final-casci

The resulting determinant count quantifies the reduction in problem size for the quantum-computing stages of the tutorial.
The final :term:`CASCI` energy is the exact ground-state energy of the selected active-space Hamiltonian, up to numerical solver tolerance, and will be the *algorithmic reference energy* for state preparation and phase estimation.
:term:`CASCI` is a full configuration-interaction calculation within the selected active space, but it is not the exact energy of N\ :sub:`2` in the full ``cc-pvdz`` orbital space: fixing the inactive orbitals as doubly occupied and the virtual orbitals as empty excludes correlation involving those orbitals.

The active-space choice
================================

The initial valence space includes more orbitals than the refined active space so that the correlated calculation can first measure the entropy of every candidate orbital.
The refinement then uses this evidence to decide which orbital occupations must remain variable and which can be frozen.
Freezing additional orbital occupations cannot lower the :term:`CASCI` energy.
It leaves the energy unchanged only if the removed determinants contribute nothing to the larger-space ground state; otherwise, as in this example, the energy increases.
The script reports the observed energy increase when reducing the active space.

The observed increase quantifies correlation excluded when reducing the initial valence space.
This active-space model error is separate from the :math:`1\ \mathrm{m}E_{\mathrm{h}}` teaching target, which applies only to the later quantum algorithm's agreement with the compact-model :term:`CASCI` reference.

.. admonition:: Why should the energy increase caused by active-space refinement not be judged against the :math:`1\ \mathrm{m}E_{\mathrm{h}}` teaching target?
   :class: quiz-question
   :collapsible: closed

   The energy increase measures correlation excluded when orbital occupations are frozen during active-space refinement.
   The :math:`1\ \mathrm{m}E_{\mathrm{h}}` target applies later when comparing the phase-estimation energy with the exact :term:`CASCI` energy of the same selected-space Hamiltonian.
   These comparisons measure different approximations.

For this tutorial, the refined active space is accepted as a compact model because it retains the orbitals with the strongest entropy-based correlation evidence while producing a tractable Hamiltonian for validating the quantum workflow.
The energy difference from the initial valence-space calculation remains documented as model error.
The next chapter will determine how the selected active spatial orbitals are represented by qubits.

Running the calculation
=======================

With the Python environment from :doc:`Before you begin <00_before_you_begin>` active, run the complete script from the Visual Studio Code integrated terminal:

.. code-block:: console

   python tutorial_choose_active_space.py

.. admonition:: What initial valence space and determinant count did the script construct?
   :class: quiz-question
   :collapsible: closed

   The script reports ten active electrons in eight active spatial orbitals, written :term:`CAS`\ :math:`(10,8)`, with five :math:`\alpha` and five :math:`\beta` active electrons.
   The active orbital indices are 2 through 9.
   Of the 28 ``cc-pvdz`` molecular orbitals, indices 0 and 1 are initially inactive and the remaining 18 are virtual.
   The determinant count is :math:`\binom{8}{5}\binom{8}{5}=3136`.

.. admonition:: Which orbitals did autoCAS retain, and how much did refinement reduce the problem size?
   :class: quiz-question
   :collapsible: closed

   Orbitals 4 through 9 have entropies from approximately 0.548 to 0.966, separated by a large gap from the remaining values of approximately 0.030 or less.
   autoCAS retains these six orbitals in :term:`CAS`\ :math:`(6,6)`, containing three :math:`\alpha` and three :math:`\beta` active electrons.
   The refined partition has four inactive orbitals, six active orbitals, and 18 virtual orbitals.
   Its determinant count is :math:`\binom{6}{3}\binom{6}{3}=400`, compared with 3,136 determinants in the initial valence space.

.. admonition:: Did the natural-orbital transformation change the CASCI energy?
   :class: quiz-question
   :collapsible: closed

   No change appears at the displayed precision.
   The script reports the signed energy change after the transformation, which is consistent with numerical roundoff near zero.
   Both calculations span the same complete active subspace, so changing the orbital representation does not change the exact :term:`CASCI` energy within that subspace.

Record the orbital representation, initial and refined active-space sizes, selection evidence, determinant counts, and both :term:`CASCI` energies in the :ref:`active-space section of the lab notebook <lab-notebook-active-space>`.
Use the final selected-space energy as the algorithmic reference, while retaining the larger-space result as evidence of the correlation excluded by the compact model.

Candidate-orbital visualization
================================

Download and open :download:`tutorial_choose_active_space.ipynb <../../_static/examples/python/tutorial_choose_active_space.ipynb>` in Visual Studio Code.
Choose **Select Kernel**, select **Python Environments**, and choose the ``.venv`` environment created in :doc:`Before you begin <00_before_you_begin>`.
If ``.venv`` does not appear, open the Command Palette, run **Developer: Reload Window**, and reopen the kernel selector.
Then select **Run All** to execute the shared active-space calculation and generate an interactive molecular-orbital viewer.

The Jupyter notebook displays every candidate natural orbital from the initial valence space, including orbitals that autoCAS did not retain.
Use the viewer to inspect the following information:

Orbital menu
   Selects each candidate natural orbital for comparison.
   The menu follows increasing molecular-orbital index, which corresponds here to decreasing natural occupation.
   The menu is not ordered by entropy.
Isosurface
   Traces points where the orbital wavefunction has a chosen positive or negative value, revealing its lobes, nodes, and spatial extent.
   The surface itself does not encode occupation or entropy.
Natural occupation
   Reports the average number of electrons in the spatial orbital.
   A value near two indicates an almost always doubly occupied orbital, a value near zero indicates an almost always empty orbital, and an intermediate value indicates variable occupation across the correlated wavefunction.
Single-orbital entropy and autoCAS selection
   Reports the uncertainty in the orbital's local occupation and whether autoCAS retained it.
   Larger entropy indicates stronger coupling to the occupations of the other active orbitals.

autoCAS selects the strongly coupled group from gaps in the orbital entropies, not from orbital shapes or a cutoff applied to the natural occupations.
Use the shapes as aids to chemical interpretation, but defend the final active space using the numerical occupation and entropy evidence in the overlays.

Further reading
===============

- :doc:`Orbital localization and transformation <../../user/comprehensive/algorithms/localizer>`
- :doc:`Active-space selection <../../user/comprehensive/algorithms/active_space>`
- :doc:`Multi-configuration calculations <../../user/comprehensive/algorithms/mc_calculator>`
- :doc:`Wavefunctions <../../user/comprehensive/data/wavefunction>`
