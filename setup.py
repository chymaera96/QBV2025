from setuptools import setup, find_packages

setup(
    name="qvim_mbn_multi",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchaudio",
        "pytorch-lightning",
        "librosa",
        "pandas",
        "numpy",
        "audiomentations",
        # add any other requirements here
    ],
)