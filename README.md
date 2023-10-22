[![Testing Linux](https://github.com/QuantumChemist/autoplex/actions/workflows/python-package.yml/badge.svg)](https://github.com/QuantumChemist/autoplex/actions/workflows/python-package.yml)
# autoplex

Software for automated fitting and benchmarking of ML potentials.

Contributions are welcome. Please raise a pull request for contributions first. At least one person has to review the code. At the beginning, Janine will take care of the reviews.

# General guidelines
- variable names should use snake case.
- please use numpy docstrings
- please write unit tests (testing will be performed with pytest)
- please ensure high coverage of the code based on the tests

# Commit guidelines
1. `pip install pre-commit`
2. Next, run `install pre-commit` (this will install all the hooks from pre-commit-config.yaml)
3. Step 1 and 2, needs to be done only once in the local repository
4. Procced with modifying the code and add commits as usual.

Please check out atomate2 for example code (https://github.com/materialsproject/atomate2)
