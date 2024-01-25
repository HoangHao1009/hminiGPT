from setuptools import setup

setup(
    name = 'hhhGPT',
    version = '0.2',
    author = 'Ha Hoang Hao',
    packages = ['hGPT'],
    description = 'A mini GPT to pre-train by your given text',
    install_requires = ['torch', 'csv', 'random', 'mmap', 'tqdm', 'sklearn', 'numpy'],
    
)
