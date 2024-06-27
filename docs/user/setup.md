Setup
================

We are referring the user to the [installation guide of atomate2](https://materialsproject.github.io/atomate2/user/install.html) in order to setup the mandatory prerequisites to be able to use `autoplex`.

After setting up `atomate2`, make sure to add `VASP_INCAR_UPDATES: {"NPAR": number}` in your ~/atomate2/config/atomate2.yaml file. 
Set a number that is a divisor of the number of tasks you use for the VASP calculations.