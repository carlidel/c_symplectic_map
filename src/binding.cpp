#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include "c_symplectic_map.h"

// PYTHON BINDINGS!

PYBIND11_MODULE(c_symplectic_map, m)
{
    py::class_<symplectic_map>(m, "symplectic_map")
        .def(py::init<double, double, double, double, double, double, double, double, double, std::vector<double>, std::vector<double>>())
        .def("reset", &symplectic_map::reset)
        .def("compute", (void (symplectic_map::*)(unsigned int, unsigned int, double)) & symplectic_map::compute)
        .def("compute", (void (symplectic_map::*)(unsigned int, unsigned int, std::vector<double>)) & symplectic_map::compute)
        .def("x", &symplectic_map::x)
        .def("p", &symplectic_map::p)
        .def("x0", &symplectic_map::x0)
        .def("p0", &symplectic_map::p0)
        .def("t", &symplectic_map::t);
}