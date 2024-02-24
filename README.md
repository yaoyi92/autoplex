[![Testing Linux](https://github.com/QuantumChemist/autoplex/actions/workflows/python-package.yml/badge.svg)](https://github.com/QuantumChemist/autoplex/actions/workflows/python-package.yml)
# autoplex

Software for automated fitting and benchmarking of ML potentials.

Contributions are welcome. Please raise a pull request for contributions first. At least one person has to review the code. In the beginning, Janine will take care of the reviews.

# General Code Structure
- We are currently aiming to follow the code structure below for each submodule (This is an initial idea; of course, this could change depending on the needs in the future)
  - autoplex/submodule/job.py (any jobs defined will be inside this module)
  - autoplex/submodule/flows.py (workflows defined will be hosted in this module)
  - autoplex/submodule/utils.py (all functions that act as utilities for defining flow or job, for example, a small subtask to calculate some metric or plotting, will be hosted in this module)

# General guidelines
- variable names should be descriptive and should use snake case.
- If you define a Maker use python class naming convention (eg: PhononMaker, RssMaker)
- please use numpy docstrings (use an IDE and switch on this docstring type; you can check examples in our code base; the doctring should be useful for other people)
- ensure type hints are added for each variable, function, class, and method (helps code readability)
- write the code in a way that users can also have control to change parameters (mainly applicable, for example, fitting protocols/ flows. In other words, avoid hardcoding. Defaults should be set, but with the possibility to modify them where required.)
- please write unit tests. (testing will be performed with pytest; please look into tests for examples)
- please ensure high coverage of the code based on the tests (you can test this with `coverage`)

# Commit guidelines
1. `pip install pre-commit`.
2. Next, run `pre-commit install` (this will install all the hooks from pre-commit-config.yaml)
3. Step 1 and 2 needs to be done only once in the local repository
4. Procced with modifying the code and adding commits as usual. This should automatically run the linters.
5. To manually run the pre-commit hooks on all files, just use `pre-commit run --all-files`
6. To run pre-commit on a specific file, use `pre-commit run --files path/to/your/modified/module/`

Please check out atomate2 for example code (https://github.com/materialsproject/atomate2)
