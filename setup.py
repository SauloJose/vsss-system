from pathlib import Path

from setuptools import find_namespace_packages, setup


ROOT = Path(__file__).parent
README = ROOT / "readme.md"

setup(
    name="vsss-system",
    version="3.4.21",
    description="Sistema modular de visao e controle para futebol de robos VSSS",
    long_description=README.read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=find_namespace_packages(
        where="src",
        include=["modules", "modules.*", "ui", "ui.*"],
    ),
    py_modules=["imports"],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.23",
        "opencv-python>=4.8",
        "Pillow>=10",
        "pyserial>=3.5",
        "ttkthemes>=3.2",
        "Unidecode>=1.3",
        "protobuf>=3.20",
        "paho-mqtt>=1.6",
    ],
    extras_require={
        "test": ["pytest>=7"],
    },
    include_package_data=True,
    zip_safe=False,
)
