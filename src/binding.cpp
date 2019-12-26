#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include "c_symplectic_map.h"

// PYTHON BINDINGS!

PYBIND11_MODULE(c_symplectic_map, m)
{
    py::class_<c_symplectic_map>(m, "c_symplectic_map")
        .def(py::init<double, double, double, double, double, double, double, double, double, std::vector<double>, std::vector<double>>())
        .def("reset", &c_symplectic_map::reset)
        .def("compute", (void (c_symplectic_map::*)(unsigned int, unsigned int, double)) & c_symplectic_map::compute)
        .def("compute", (void (c_symplectic_map::*)(unsigned int, unsigned int, std::vector<double>)) & c_symplectic_map::compute)
        .def("x", &c_symplectic_map::x)
        .def("p", &c_symplectic_map::p)
        .def("x0", &c_symplectic_map::x0)
        .def("p0", &c_symplectic_map::p0)
        .def("t", &c_symplectic_map::t);
}