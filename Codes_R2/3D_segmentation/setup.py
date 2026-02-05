from setuptools import setup, Extension
import numpy


setup(
    ext_modules = [
    Extension(
            "CrystalTracer3D.gwdt.gwdt_impl",
            ["CrystalTracer3D/gwdt/gwdt_impl.pyx"],
            language="c++",
            include_dirs=[numpy.get_include()]
        )
    ]
)

