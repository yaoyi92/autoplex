from jobflow import job, Maker
from dataclasses import dataclass, field
from multiprocessing import Pool
from pymatgen.io.ase import AseAtomsAdaptor
import os
import re
import ase.io
from subprocess import run
import numpy as np
from pathlib import Path
from ase.data import atomic_numbers, covalent_radii


@dataclass
class RandomizedStructure(Maker):
    """
    Maker to create random structures by 'buildcell'
    
    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    struct_number : int
        Epected number of generated randomized unit cells.
    tag : (str) 
        name of the seed file for builcell.
    input_file_name: str 
        input file of buildcell to set parameters
    output_file_name : str
        A file to store all generated structures. 
    remove_tmp_files : bool 
        Remove all temporary files raised by buildcell to save memory
    cell_seed_path : str
        Path to the custom buildcell control file, which ends with ".cell". If this file exists, 
        the buildcell_options argument will no longer take effect
    """

    name: str = "Build_random_cells"
    struct_number: int = 20
    tag: str = 'Si'
    output_file_name: str = 'random_structs.extxyz'
    remove_tmp_files: bool = True
    buildcell_options: list = field(default_factory=lambda: [
                                    'VARVOL=15',
                                    'SPECIES=Si%NUM=1',
                                    'NFORM=1-7',
                                    'SYMMOPS=1-8',
                                    'SLACK=0.25',
                                    'OVERLAP=0.1',
                                    'COMPACT',
                                    'MINSEP=1.5',
                                ])
    cell_seed_path: str = None

    @job
    def make(self):
        
        if self.cell_seed_path:

            if not os.path.isfile(self.cell_seed_path):
                raise FileNotFoundError(f"No file found at the specified path: {self.cell_seed_path}")
            else:
                bt_file = self.cell_seed_path

        else: 

            elements = self._extract_elements(self.tag)  # {"Si":1, "O":2}
            make_species = self._make_species(elements)   # Si%NUM=1,O%NUM=2

            r0 = {}
            varvol = {}
            num_atom_formula = 0
            total_varvol_formula = 0

            for ele in elements:

                r0[ele] = covalent_radii[atomic_numbers[ele]]

                if self._is_metal(ele):
                    varvol[ele] = 4.5 * np.power(r0[ele],3)
                else:
                    varvol[ele] = 12.0 * np.power(r0[ele],3)

                total_varvol_formula += varvol[ele] * elements[ele]

                num_atom_formula += elements[ele]

            mean_var = total_varvol_formula/num_atom_formula*len(elements)

            minsep = self._make_minsep(r0)

            self._update_buildcell_options({'VARVOL': mean_var,
                                        'SPECIES': make_species,
                                        'MINSEP': minsep,})
            self._cell_seed(self.buildcell_options, self.tag)
            bt_file = '{}.cell'.format(self.tag)

        with Pool() as pool:
            args = [(i, bt_file, self.tag, self.remove_tmp_files) for i in range(self.struct_number)]
            atoms_group = pool.starmap(self._parallel_process, args)

        output_file = open(self.output_file_name, 'w')
        ase.io.write(output_file, atoms_group, parallel=False, format="extxyz")

        dir_path = Path.cwd()
        path = os.path.join(dir_path, self.output_file_name)

        return path
        

    def _update_buildcell_options(self, updates):
        """
        Update buildcell options based on a dictionary of updates.
        
        Parameters
        ----------
        updates : dict
            A dictionary with option as key and new value as value.
        """
        for i, option in enumerate(self.buildcell_options):
            option_key = option.split('=')[0]
            if option_key in updates:
                self.buildcell_options[i] = f'{option_key}={updates[option_key]}'


    def _cell_seed(self,
                   buildcell_options,
                   tag,):
        
        '''
        Prepares random cells in self.directory
        Arguments:
        buildcell options :: (list of str) e.g. ['VARVOL=20']
        '''

        bc_file = '{}.cell'.format(tag)
        contents = []
        contents.extend(['#' + i + '\n' for i in buildcell_options])
        
        with open(bc_file, 'w') as f:
            f.writelines(contents)


    def _is_metal(self, element_symbol):

        metals = ['Li', 'Be', 'Na', 'Mg', 'Al', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu',
                  'Zn', 'Ga', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
                  'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
                  'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Fr', 'Ra', 'Ac',
                  'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db',
                  'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']

        return element_symbol in metals

    
    def _extract_elements(self, input_str):

        elements = {}
        pattern = re.compile(r'([A-Z][a-z]*)(\d*)')
        matches = pattern.findall(input_str)
        
        for match in matches:
            element, count = match
            count = int(count) if count else 1
            if element in elements:
                elements[element] += count
            else:
                elements[element] = count
        
        return elements


    def _make_species(self, elements):

        output = ""
        for element, count in elements.items():
            output += f"{element}%NUM={count},"
        return output[:-1]


    def _make_minsep(self, r):

        keys = list(r.keys())
        if len(keys) == 1:
            minsep = str(1.5*r[keys[0]])
            return minsep
        else:
            minsep = "1.5 "
            for i in range(len(keys)):
                for j in range(i, len(keys)):
                    el1, el2 = keys[i], keys[j]
                    r1, r2 = r[el1], r[el2]
                    if el1 == el2:
                            result = r1 * 2.0
                    else:
                        if self._is_metal(el1) and self._is_metal(el2):
                            result = (r1 + r2) / 2 * 2.0
                        else:
                            result = (r1 + r2) / 2 * 1.5

                    minsep += f"{el1}-{el2}={result} "

        return minsep[:-1]


    def _parallel_process(i, bt_file, tag, remove_tmp_files):

        tmp_file_name = "tmp." + str(i) + '.' + tag + '.cell'

        run("buildcell",
            stdin=open(bt_file, "r"),
            stdout=open(tmp_file_name, "w"),
            shell=True).check_returncode()
        
        atom = ase.io.read(tmp_file_name, parallel=False)
        atom.info["unique_starting_index"] = i

        if "castep_labels" in atom.arrays:
            del atom.arrays["castep_labels"]

        if "initial_magmoms" in atom.arrays:
            del atom.arrays["initial_magmoms"]

        if remove_tmp_files:
            os.remove(tmp_file_name)

        return atom