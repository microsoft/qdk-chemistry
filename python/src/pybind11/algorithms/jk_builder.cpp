// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "qdk/chemistry/algorithms/microsoft/jk_builder.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <optional>
#include <qdk/chemistry/algorithms/jk_builder.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "factory_bindings.hpp"

namespace py = pybind11;
using namespace qdk::chemistry::algorithms;
using namespace qdk::chemistry::data;

class JKEngineBase : public JKEngine,
                     public pybind11::trampoline_self_life_support {
 public:
  // Python implementations receive views of the caller's matrices, valid
  // during the call.
  void build(const double* P, size_t num_atomic_orbitals, size_t num_densities,
             double* J, double* K, double alpha, double beta,
             double omega) override {
    py::gil_scoped_acquire gil;
    py::function override =
        py::get_override(static_cast<const JKEngine*>(this), "build");
    if (!override) {
      throw std::runtime_error("JKEngine.build is not implemented");
    }
    const std::vector<py::ssize_t> shape{
        static_cast<py::ssize_t>(num_densities),
        static_cast<py::ssize_t>(num_atomic_orbitals),
        static_cast<py::ssize_t>(num_atomic_orbitals)};
    override(densities(P, shape), view(J, shape), view(K, shape), alpha, beta,
             omega);
  }

  void build_gradients(const double* P, size_t num_atomic_orbitals,
                       size_t num_densities, size_t num_atoms, double* dJ,
                       double* dK, double alpha, double beta,
                       double omega) override {
    py::gil_scoped_acquire gil;
    py::function override =
        py::get_override(static_cast<const JKEngine*>(this), "build_gradients");
    if (!override) {
      JKEngine::build_gradients(P, num_atomic_orbitals, num_densities,
                                num_atoms, dJ, dK, alpha, beta, omega);
      return;
    }
    const std::vector<py::ssize_t> shape{
        static_cast<py::ssize_t>(num_densities),
        static_cast<py::ssize_t>(num_atomic_orbitals),
        static_cast<py::ssize_t>(num_atomic_orbitals)};
    const std::vector<py::ssize_t> gradient_shape{
        3, static_cast<py::ssize_t>(num_atoms)};
    override(densities(P, shape), view(dJ, gradient_shape),
             view(dK, gradient_shape), alpha, beta, omega);
  }

 private:
  static py::object view(double* data, const std::vector<py::ssize_t>& shape) {
    if (!data) return py::none();
    return py::array_t<double>(shape, data, py::capsule(data, [](void*) {}));
  }

  static py::array_t<double> densities(const double* P,
                                       const std::vector<py::ssize_t>& shape) {
    py::array_t<double> array(shape, P, py::capsule(P, [](void*) {}));
    array.attr("setflags")(py::arg("write") = false);
    return array;
  }
};

class JKBuilderBase : public JKBuilder,
                      public pybind11::trampoline_self_life_support {
 public:
  std::string name() const override {
    PYBIND11_OVERRIDE_PURE(std::string, JKBuilder, name);
  }

  std::vector<std::string> aliases() const override {
    PYBIND11_OVERRIDE(std::vector<std::string>, JKBuilder, aliases);
  }

  void replace_settings(std::unique_ptr<Settings> new_settings) {
    this->_settings = std::move(new_settings);
  }

 protected:
  std::shared_ptr<JKEngine> _run_impl(std::shared_ptr<BasisSet> basis_set,
                                      double eri_threshold) const override {
    PYBIND11_OVERRIDE_PURE(std::shared_ptr<JKEngine>, JKBuilder, _run_impl,
                           basis_set, eri_threshold);
  }
};

void bind_jk_builder(py::module& m) {
  py::class_<JKEngine, JKEngineBase, py::smart_holder> engine(m, "JKEngine",
                                                              R"(
Coulomb (J) and exchange (K) builds in one atomic-orbital basis set.

Created by :meth:`JKBuilder.run` for a basis set and reused for every build
of a calculation (for example, each SCF iteration). Implementations override
``build`` and, to provide nuclear gradients, ``build_gradients``.
)");

  engine.def(py::init<>(), R"(
Create a ``JKEngine`` instance.

Default constructor for the abstract base class; typically called via
``super().__init__()`` from a derived class.
)");
  // Number of atomic orbitals of (n, n) or (num_densities, n, n) densities
  auto num_atomic_orbitals = [](const py::array& P) -> size_t {
    if (P.ndim() != 2 && P.ndim() != 3) {
      throw std::invalid_argument(
          "densities must have shape (n, n) or (num_densities, n, n)");
    }
    const size_t n = P.shape(P.ndim() - 1);
    if (n == 0 || static_cast<size_t>(P.shape(P.ndim() - 2)) != n) {
      throw std::invalid_argument(
          "densities must hold nonempty square matrices");
    }
    return n;
  };
  engine.def(
      "build",
      [num_atomic_orbitals](
          JKEngine& self,
          py::array_t<double, py::array::c_style | py::array::forcecast> P,
          std::optional<py::array> J, std::optional<py::array> K, double alpha,
          double beta, double omega) {
        const size_t n = num_atomic_orbitals(P);
        auto output = [&](std::optional<py::array>& X,
                          const char* name) -> double* {
          if (!X) return nullptr;
          const bool same_shape =
              X->ndim() == P.ndim() &&
              std::equal(P.shape(), P.shape() + P.ndim(), X->shape());
          if (!X->dtype().is(py::dtype::of<double>()) || !same_shape ||
              !(X->flags() & py::array::c_style) || !X->writeable()) {
            throw std::invalid_argument(
                std::string(name) +
                " must be a writable C-contiguous float64 array with the "
                "shape of densities");
          }
          return static_cast<double*>(X->mutable_data());
        };
        double* j = output(J, "coulomb");
        double* k = output(K, "exchange");
        self.build(P.data(), n, P.size() / (n * n), j, k, alpha, beta, omega);
      },
      py::arg("densities"), py::arg("coulomb"), py::arg("exchange"),
      py::arg("alpha") = 1.0, py::arg("beta") = 0.0, py::arg("omega") = 0.0, R"(
Build J and K for each density matrix.

:math:`J[P]_{\mu\nu} = \sum_{\lambda\sigma} (\mu\nu|\lambda\sigma) P_{\lambda\sigma}`
and :math:`K[P]_{\mu\nu} = \sum_{\lambda\sigma} (\mu\lambda|v|\nu\sigma) P_{\lambda\sigma}`
with the exchange operator
:math:`v = \alpha / r + \beta \operatorname{erfc}(\omega r) / r`.
The matrices are in the atomic orbitals of the engine's basis set.

Args:
    densities (numpy.ndarray): Symmetric density matrices, shape ``(num_densities, n, n)`` or ``(n, n)``.
    coulomb (numpy.ndarray | None): Output Coulomb matrices (writable, C-contiguous, float64, shape of ``densities``), or ``None`` to skip J.
    exchange (numpy.ndarray | None): Output exchange matrices (as ``coulomb``), or ``None`` to skip K.
    alpha (float): Coefficient of the full-range exchange operator.
    beta (float): Coefficient of the short-range exchange operator.
    omega (float): Range-separation parameter.

Implementations called by QDK/Chemistry receive ``densities`` (read-only),
``coulomb`` and ``exchange`` with shape ``(num_densities, n, n)``, as views
valid during the call.
)");
  engine.def(
      "build_gradients",
      [num_atomic_orbitals](
          JKEngine& self,
          py::array_t<double, py::array::c_style | py::array::forcecast> P,
          std::optional<py::array> dJ, std::optional<py::array> dK,
          double alpha, double beta, double omega) {
        const size_t n = num_atomic_orbitals(P);
        py::ssize_t num_atoms = -1;
        auto output = [&](std::optional<py::array>& X,
                          const char* name) -> double* {
          if (!X) return nullptr;
          if (!X->dtype().is(py::dtype::of<double>()) || X->ndim() != 2 ||
              X->shape(0) != 3 ||
              (num_atoms >= 0 && X->shape(1) != num_atoms) ||
              !(X->flags() & py::array::c_style) || !X->writeable()) {
            throw std::invalid_argument(
                std::string(name) +
                " must be a writable C-contiguous float64 array of shape "
                "(3, num_atoms)");
          }
          num_atoms = X->shape(1);
          return static_cast<double*>(X->mutable_data());
        };
        double* dj = output(dJ, "coulomb");
        double* dk = output(dK, "exchange");
        if (num_atoms < 0) return;
        self.build_gradients(P.data(), n, P.size() / (n * n),
                             static_cast<size_t>(num_atoms), dj, dk, alpha,
                             beta, omega);
      },
      py::arg("densities"), py::arg("coulomb"), py::arg("exchange"),
      py::arg("alpha") = 1.0, py::arg("beta") = 0.0, py::arg("omega") = 0.0, R"(
Build the nuclear gradients of the Coulomb and exchange energies.

Derivatives with respect to the nuclear coordinates, at fixed densities, of
:math:`E_J = \tfrac{1}{2} \operatorname{Tr}(P J[P])` for the total density
:math:`P` and of :math:`E_K = -\tfrac{1}{4} \operatorname{Tr}(P K[P])` for one
density matrix or :math:`E_K = -\tfrac{1}{2} \sum_s \operatorname{Tr}(P_s K[P_s])`
for two (alpha and beta), with :math:`J` and :math:`K` as in :meth:`build`.

Args:
    densities (numpy.ndarray): Symmetric density matrices, shape ``(num_densities, n, n)`` or ``(n, n)``.
    coulomb (numpy.ndarray | None): Output gradient of :math:`E_J` (writable, C-contiguous, float64, shape ``(3, num_atoms)``), or ``None`` to skip it.
    exchange (numpy.ndarray | None): Output gradient of :math:`E_K` (as ``coulomb``), or ``None`` to skip it.
    alpha (float): Coefficient of the full-range exchange operator.
    beta (float): Coefficient of the short-range exchange operator.
    omega (float): Range-separation parameter.

Raises:
    RuntimeError: If the engine does not build gradients (the default).

Implementations called by QDK/Chemistry receive ``densities`` (read-only),
``coulomb`` and ``exchange`` with shape ``(3, num_atoms)``, as views valid
during the call.
)");

  py::class_<JKBuilder, JKBuilderBase, py::smart_holder> builder(m, "JKBuilder",
                                                                 R"(
Abstract base class for J/K builders.

A J/K builder creates the Coulomb and exchange engine of a basis set. The
``qdk`` SCF solver builds its J and K matrices with the builder named by its
``jk_builder`` setting.

Examples:
    >>> import qdk_chemistry.algorithms as alg
    >>> engine = alg.create("jk_builder", "qdk").run(basis_set, 1e-12)
    >>> engine.build(densities, coulomb, exchange)
)");

  builder.def(py::init<>(), R"(
Create a ``JKBuilder`` instance.

Default constructor for the abstract base class; typically called via
``super().__init__()`` from a derived class.
)");
  builder.def("run", &JKBuilder::run, py::arg("basis_set"),
              py::arg("eri_threshold"), R"(
Create the J/K engine of a basis set.

Args:
    basis_set (qdk_chemistry.data.BasisSet): Basis set; the engine's matrices are in its atomic orbitals, in their order and with each contracted shell normalized.
    eri_threshold (float): Integral screening threshold.

Returns:
    JKEngine: The engine.
)");
  builder.def("settings", &JKBuilder::settings,
              py::return_value_policy::reference_internal, R"(
Access the builder's configuration settings.

Returns:
    qdk_chemistry.data.Settings: Reference to the settings object.
)");
  builder.def_property(
      "_settings",
      [](JKBuilderBase& instance) -> Settings& { return instance.settings(); },
      [](JKBuilderBase& instance, std::unique_ptr<Settings> new_settings) {
        instance.replace_settings(std::move(new_settings));
      },
      py::return_value_policy::reference_internal, R"(
Internal settings object property.

Allows derived classes to replace the settings object with a specialized
Settings subclass in their constructor.
)");
  builder.def("name", &JKBuilder::name, R"(
The algorithm's name.

Returns:
    str: The name of the algorithm
)");
  builder.def("aliases", &JKBuilder::aliases, R"(
The algorithm's aliases.

Returns:
    list[str]: All registered names for the algorithm
)");
  builder.def("type_name", &JKBuilder::type_name, R"(
The algorithm's type name.

Returns:
    str: The type name of the algorithm
)");
  builder.def("hash", &JKBuilder::hash, py::arg("basis_set"),
              py::arg("eri_threshold"), R"(
Compute a deterministic content hash for a run with these inputs.

Args:
    basis_set (qdk_chemistry.data.BasisSet): Basis set.
    eri_threshold (float): Integral screening threshold.

Returns:
    str: 16-character hex content hash.
)");
  builder.def("__repr__", [](const JKBuilder&) {
    return "<qdk_chemistry.algorithms.JKBuilder>";
  });

  qdk::chemistry::python::bind_create_nested(builder);

  py::class_<microsoft::JKBuilder, JKBuilder, py::smart_holder>(
      m, "QdkJKBuilder", R"(
QDK implementation of the J/K builder.

Its engines evaluate the integrals with Libint2, directly or in memory as
selected by the ``eri_method`` setting, and build nuclear gradients directly.
)")
      .def(py::init<>(), R"(
Default constructor.
)");

  qdk::chemistry::python::bind_algorithm_factory<JKBuilderFactory, JKBuilder,
                                                 JKBuilderBase>(
      m, "JKBuilderFactory");
}
