# dynamic system identification with machine learning

### WSL

I am using [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) on my windows pc to run the code in this repository.

### package manager
I am using [uv](https://docs.astral.sh/uv/#__tabbed_1_2) to manage python packages.

inital setup to use python version 3.10:
```
uv python install 3.10
uv python pin pypy@3.10
```

create venv by 
```
uv venv
```
and install packages
```
uv sync
```
