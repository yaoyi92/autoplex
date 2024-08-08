from __future__ import annotations

import os
import pytest
import matplotlib.pyplot as plt
from pymatgen.core.structure import Structure
from autoplex.data.common.utils import (
    energy_plot,
    force_plot,
    plot_energy_forces,
    filter_outlier_energy,
    filter_outlier_forces,
    generate_supercell_matrix,
)
from autoplex.data.phonons.utils import update_phonon_displacement_maker, reduce_supercell_size
from atomate2.common.jobs.phonons import get_supercell_size

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=1

fig, ax_list = plt.subplots(nrows=1, ncols=1)
fig.set_size_inches(15, 20)


@pytest.fixture(scope="class")
def mp_1203790():
    return Structure.from_dict(  # mp-1203790
        {'@module': 'pymatgen.core.structure', '@class': 'Structure', 'charge': 0, 'lattice': {
            'matrix': [[-5.18331847, -8.9777624, 0.0], [-5.18331547, 8.97776139, -0.0], [0.0, 0.0, -16.97027249]],
            'pbc': (True, True, True), 'a': 10.36662954254163, 'b': 10.366627167855322, 'c': 16.97027249, 'alpha': 90.0,
            'beta': 90.0, 'gamma': 119.99996435798555, 'volume': 1579.409195104524}, 'properties': {}, 'sites': [
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.95006076, 0.66941597, 0.56452266],
             'xyz': [-8.394261638096292, -2.519562919528026, -9.580103366979623], 'properties': {'magmom': -0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.71935521, 0.04993924, 0.56452366],
                              'xyz': [-3.9874979817357716, -6.009857575864161, -9.580120337252113],
                              'properties': {'magmom': -0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.33058503, 0.28064479, 0.56452266],
             'xyz': [-3.1681979734864054, -0.44835189237021433, -9.580103366979623], 'properties': {'magmom': -0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.71935321, 0.66941397, 0.56452366],
                              'xyz': [-7.1984205663819045, -0.4483433072646852, -9.580120337252113],
                              'properties': {'magmom': -0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.33058603, 0.04993924, 0.56452366],
             'xyz': [-1.9723835104750167, -2.519580249381328, -9.580120337252113], 'properties': {'magmom': -0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.95006076, 0.28064579, 0.56452366],
                              'xyz': [-6.379143149827607, -6.0098488311153755, -9.580120337252113],
                              'properties': {'magmom': -0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.04993924, 0.33058403, 0.43547534],
             'xyz': [-1.972372301903707, 2.5195619095280253, -7.3901351824753965], 'properties': {'magmom': -0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.28064479, 0.95006076, 0.43547534],
                              'xyz': [-6.379135958264228, 6.00985656586416, -7.3901351824753965],
                              'properties': {'magmom': -0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.66941597, 0.71935521, 0.43547534],
             'xyz': [-7.198441149832064, 0.44834190460781453, -7.3901351824753965], 'properties': {'magmom': -0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.28064679, 0.33058603, 0.43547534],
                              'xyz': [-3.1682133736180953, 0.44834229726468555, -7.3901351824753965],
                              'properties': {'magmom': -0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.66941497, 0.95006076, 0.43547534],
             'xyz': [-8.394255612843452, 2.519570261618928, -7.3901351824753965], 'properties': {'magmom': -0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.04993924, 0.71935421, 0.43547534],
                              'xyz': [-3.9874907901723917, 6.009847821115375, -7.3901351824753965],
                              'properties': {'magmom': -0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.04994024, 0.33058403, 0.06452466],
             'xyz': [-1.972377485222177, 2.5195529317656256, -1.0950010625246034], 'properties': {'magmom': -0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.28064479, 0.95005976, 0.06452466],
                              'xyz': [-6.3791307749487585, 6.009847588102771, -1.0950010625246034],
                              'properties': {'magmom': -0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.66941497, 0.71935521, 0.06452466],
             'xyz': [-7.198435966513594, 0.4483508823702138, -1.0950010625246034], 'properties': {'magmom': -0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.28064679, 0.33058603, 0.06452466],
                              'xyz': [-3.1682133736180953, 0.44834229726468555, -1.0950010625246034],
                              'properties': {'magmom': -0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.66941397, 0.95005976, 0.06452466],
             'xyz': [-8.394245246209513, 2.519570261619939, -1.0950010625246034], 'properties': {'magmom': -0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.04994024, 0.71935421, 0.06452466],
                              'xyz': [-3.9874959734908617, 6.009838843352975, -1.0950010625246034],
                              'properties': {'magmom': -0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.95005976, 0.66941597, 0.93547734],
             'xyz': [-8.394256454777823, -2.5195539417656256, -15.875305368020376], 'properties': {'magmom': -0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.71935521, 0.04994024, 0.93547634],
                              'xyz': [-3.9875031650512414, -6.009848598102771, -15.875288397747886],
                              'properties': {'magmom': -0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.33058503, 0.28064479, 0.93547634],
             'xyz': [-3.1681979734864054, -0.44835189237021433, -15.875288397747886], 'properties': {'magmom': -0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.71935321, 0.66941397, 0.93547634],
                              'xyz': [-7.1984205663819045, -0.4483433072646852, -15.875288397747886],
                              'properties': {'magmom': -0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.33058603, 0.04994024, 0.93547634],
             'xyz': [-1.9723886937904866, -2.519571271619938, -15.875288397747886], 'properties': {'magmom': -0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.95005976, 0.28064579, 0.93547634],
                              'xyz': [-6.379137966509138, -6.009839853352975, -15.875288397747886],
                              'properties': {'magmom': -0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.91572454, 0.45786277, 0.86478307],
             'xyz': [-7.119739100492306, -4.110574645544846, -14.675604342638744], 'properties': {'magmom': -0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.54213923, 0.08427546, 0.86478307],
                              'xyz': [-3.246906579729944, -4.110592223746462, -14.675604342638744],
                              'properties': {'magmom': -0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.54213723, 0.45786077, 0.86478307],
             'xyz': [-5.18330672978075, -0.7566144962324817, -14.675604342638744], 'properties': {'magmom': -0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.08427546, 0.54213723, 0.13521793],
                              'xyz': [-3.246894839507694, 4.110573635544846, -2.2946851176337453],
                              'properties': {'magmom': -0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.45786077, 0.91572454, 0.13521793],
             'xyz': [-7.119727360270056, 4.110591213746463, -2.2946851176337453], 'properties': {'magmom': -0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.45786277, 0.54213923, 0.13521793],
                              'xyz': [-5.18332721021925, 0.7566134862324816, -2.2946851176337453],
                              'properties': {'magmom': -0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.08427646, 0.54213723, 0.36478107],
             'xyz': [-3.246900022826164, 4.1105646577824455, -6.1904341570937635], 'properties': {'magmom': -0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.45786077, 0.91572354, 0.36478107],
                              'xyz': [-7.1197221769545855, 4.110582235985073, -6.1904341570937635],
                              'properties': {'magmom': -0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.45786277, 0.54214023, 0.36478107],
             'xyz': [-5.1833323935347195, 0.7566224639938719, -6.1904341570937635], 'properties': {'magmom': -0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.91572354, 0.45786277, 0.63521793],
                              'xyz': [-7.119733917173836, -4.110565667782446, -10.779821362633745],
                              'properties': {'magmom': -0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.54213923, 0.08427646, 0.63521793],
             'xyz': [-3.2469117630454143, -4.1105832459850715, -10.779821362633745], 'properties': {'magmom': -0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.54213723, 0.45785977, 0.63521793],
                              'xyz': [-5.18330154646528, -0.756623473993872, -10.779821362633745],
                              'properties': {'magmom': -0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.87645037, 0.12354863, 0.36130919],
             'xyz': [-5.18331291603564, -6.759373057050692, -6.131515407441183], 'properties': {'magmom': 0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.24709825, 0.12354963, 0.36130919],
                              'xyz': [-1.9211856316214535, -1.1091902799930142, -6.131515407441183],
                              'properties': {'magmom': 0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.87645037, 0.75290175, 0.36130919],
             'xyz': [-8.445448679024407, -1.1091909156386555, -6.131515407441183], 'properties': {'magmom': 0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.12354963, 0.87645137, 0.63868881],
                              'xyz': [-5.18332102396436, 6.759372047050691, -10.838723142013837],
                              'properties': {'magmom': 0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.75290175, 0.87645037, 0.63868881],
             'xyz': [-8.445448308378547, 1.1091892699930135, -10.838723142013837], 'properties': {'magmom': 0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.12354963, 0.24709825, 0.63868881],
                              'xyz': [-1.9211852609755935, 1.1091899056386554, -10.838723142013837],
                              'properties': {'magmom': 0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.12354963, 0.87645137, 0.86131119],
             'xyz': [-5.18332102396436, 6.759372047050691, -14.616685592986162], 'properties': {'magmom': 0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.75290175, 0.87645037, 0.86131119],
                              'xyz': [-8.445448308378547, 1.1091892699930135, -14.616685592986162],
                              'properties': {'magmom': 0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.12354963, 0.24709825, 0.86131119],
             'xyz': [-1.9211852609755935, 1.1091899056386554, -14.616685592986162], 'properties': {'magmom': 0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.87645037, 0.12354863, 0.13869081],
                              'xyz': [-5.18331291603564, -6.759373057050692, -2.353620837558817],
                              'properties': {'magmom': 0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.24709825, 0.12354963, 0.13868981],
             'xyz': [-1.9211856316214535, -1.1091902799930142, -2.353603867286327], 'properties': {'magmom': 0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.87645037, 0.75290175, 0.13868981],
                              'xyz': [-8.445448679024407, -1.1091909156386555, -2.353603867286327],
                              'properties': {'magmom': 0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.666667, 0.333333, 0.8196504],
             'xyz': [-5.183317470001, -2.99259378850793, -13.909690634537496], 'properties': {'magmom': 0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.333333, 0.666667, 0.1803516],
                              'xyz': [-5.183316469999, 2.9925927785079303, -3.060615796007484],
                              'properties': {'magmom': 0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.333333, 0.666667, 0.3196484],
             'xyz': [-5.183316469999, 2.9925927785079303, -5.424520448992515], 'properties': {'magmom': 0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.666667, 0.333333, 0.6803506],
                              'xyz': [-5.183317470001, -2.99259378850793, -11.545735070734993],
                              'properties': {'magmom': 0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [-0.0, -0.0, 0.81830178],
             'xyz': [0.0, 0.0, -13.886804185652032], 'properties': {'magmom': -0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [-0.0, -0.0, 0.18169922],
             'xyz': [0.0, 0.0, -3.0834852746204575], 'properties': {'magmom': -0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [-0.0, -0.0, 0.31829978],
             'xyz': [0.0, 0.0, -5.401634000107052], 'properties': {'magmom': -0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [-0.0, -0.0, 0.68169922],
             'xyz': [0.0, 0.0, -11.568621519620459], 'properties': {'magmom': -0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.93042857, 0.46521428, 0.25],
             'xyz': [-7.234059966285599, -4.1765838305711185, -4.2425681225], 'properties': {'magmom': -0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.53478672, 0.06957143, 0.25],
                              'xyz': [-3.1325805526757406, -4.1765924087342405, -4.2425681225],
                              'properties': {'magmom': -0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.53478472, 0.46521328, 0.25],
             'xyz': [-5.18330670772322, -0.6245963280112683, -4.2425681225], 'properties': {'magmom': -0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.06957243, 0.53478572, 0.75],
                              'xyz': [-3.132579157032871, 4.176573842808719, -12.7277043675],
                              'properties': {'magmom': -0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.46521228, 0.93042757, 0.75],
             'xyz': [-7.2340430206903195, 4.176591398735249, -12.7277043675], 'properties': {'magmom': -0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.46521428, 0.53478772, 0.75],
                              'xyz': [-5.183327232273779, 0.6246132735350587, -12.7277043675],
                              'properties': {'magmom': -0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.79765372, 0.20234828, 0.25],
             'xyz': [-5.1833282295920995, -5.344511000119218, -4.2425681225], 'properties': {'magmom': -0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.40469355, 0.20234628, 0.25],
                              'xyz': [-3.14648015582582, -1.816625916718391, -4.2425681225],
                              'properties': {'magmom': -0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.79765172, 0.59530645, 0.25],
             'xyz': [-7.220144024579049, -1.8166083580833616, -4.2425681225], 'properties': {'magmom': -0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.20234628, 0.79765172, 0.75],
                              'xyz': [-5.1833057104079, 5.344509990119218, -12.7277043675],
                              'properties': {'magmom': -0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.59530545, 0.79765272, 0.75],
             'xyz': [-7.220143417540241, 1.816624906719401, -12.7277043675], 'properties': {'magmom': -0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.20234928, 0.40469455, 0.75],
                              'xyz': [-3.14650028205489, 1.8166073480823524, -12.7277043675],
                              'properties': {'magmom': -0.0}, 'label': 'Si'}]})


@pytest.fixture(scope="class")
def mp_1200830():
    return Structure.from_dict(  # mp-1200830
        {'@module': 'pymatgen.core.structure', '@class': 'Structure', 'charge': 0, 'lattice': {
            'matrix': [[-6.2141887, -6.2141887, 6.2141887], [-6.2141887, 6.2141887, -6.2141887],
                       [6.2141887, -6.2141887, -6.2141887]], 'pbc': (True, True, True), 'a': 10.763290556220392,
            'b': 10.763290556220392, 'c': 10.763290556220392, 'alpha': 109.47122063449069, 'beta': 109.47122063449069,
            'gamma': 109.47122063449069, 'volume': 959.8719531108837}, 'properties': {}, 'sites': [
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.78659388, 0.39175378, 0.13262899],
             'xyz': [-6.498293142493029, -3.277792458677283, 1.629429316776457], 'properties': {'magmom': 0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.3948401, 0.74087421, 0.60824622],
                              'xyz': [-3.2777862444885835, -1.6294355309651567, -5.930078043318272],
                              'properties': {'magmom': 0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.65396589, 0.86737101, 0.25912579],
             'xyz': [-7.843618016776459, -0.28411687087042936, -2.9363962413227167], 'properties': {'magmom': 0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.65396589, 0.25912579, 0.86737101],
                              'xyz': [-0.28411687087042914, -7.843618016776459, -2.9363962413227167],
                              'properties': {'magmom': 0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.78659388, 0.13262899, 0.39175378],
             'xyz': [-3.2777924586772835, -6.498293142493028, 1.6294293167764566], 'properties': {'magmom': 0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.3948401, 0.60824622, 0.74087421],
                              'xyz': [-1.6294355309651563, -3.2777862444885835, -5.930078043318272],
                              'properties': {'magmom': 0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.25912579, 0.65396589, 0.86737101],
             'xyz': [-0.28411687087042914, -2.9363962413227167, -7.843618016776459], 'properties': {'magmom': 0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.13262899, 0.78659388, 0.39175378],
                              'xyz': [-3.2777924586772835, 1.6294293167764566, -6.498293142493028],
                              'properties': {'magmom': 0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.60824622, 0.3948401, 0.74087421],
             'xyz': [-1.6294355309651563, -5.930078043318272, -3.2777862444885835], 'properties': {'magmom': 0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.39175378, 0.78659388, 0.13262899],
                              'xyz': [-6.498293142493029, 1.629429316776457, -3.277792458677283],
                              'properties': {'magmom': 0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.74087421, 0.3948401, 0.60824622],
             'xyz': [-3.2777862444885835, -5.930078043318272, -1.6294355309651567], 'properties': {'magmom': 0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.86737101, 0.65396589, 0.25912579],
                              'xyz': [-7.843618016776459, -2.9363962413227167, -0.28411687087042936],
                              'properties': {'magmom': 0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.74087421, 0.60824622, 0.3948401],
             'xyz': [-5.930078043318271, -3.2777862444885835, -1.6294355309651567], 'properties': {'magmom': 0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.86737101, 0.25912579, 0.65396589],
                              'xyz': [-2.9363962413227167, -7.843618016776459, -0.28411687087042914],
                              'properties': {'magmom': 0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.39175378, 0.13262899, 0.78659388],
             'xyz': [1.629429316776457, -6.498293142493029, -3.277792458677283], 'properties': {'magmom': 0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.60824622, 0.74087421, 0.3948401],
                              'xyz': [-5.930078043318271, -1.6294355309651567, -3.2777862444885835],
                              'properties': {'magmom': 0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.25912579, 0.86737101, 0.65396589],
             'xyz': [-2.9363962413227167, -0.28411687087042914, -7.843618016776459], 'properties': {'magmom': 0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.13262899, 0.39175378, 0.78659388],
                              'xyz': [1.629429316776457, -3.277792458677283, -6.498293142493029],
                              'properties': {'magmom': 0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.21340612, 0.34603411, 0.6051599],
             'xyz': [0.28411065668172863, -2.936402455511417, -4.584753169034843], 'properties': {'magmom': 0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.6051599, 0.21340612, 0.34603411],
                              'xyz': [-2.936402455511417, -4.584753169034843, 0.28411065668172863],
                              'properties': {'magmom': 0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.34603411, 0.6051599, 0.21340612],
             'xyz': [-4.584753169034843, 0.28411065668172886, -2.936402455511417], 'properties': {'magmom': 0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.34603411, 0.21340612, 0.6051599],
                              'xyz': [0.28411065668172863, -4.584753169034843, -2.936402455511417],
                              'properties': {'magmom': 0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.21340612, 0.6051599, 0.34603411],
             'xyz': [-2.936402455511417, 0.28411065668172863, -4.584753169034843], 'properties': {'magmom': 0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.6051599, 0.34603411, 0.21340612],
                              'xyz': [-4.584753169034843, -2.936402455511417, 0.28411065668172886],
                              'properties': {'magmom': 0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [-0.0, -0.0, 0.21736945],
             'xyz': [1.350774779915215, -1.350774779915215, -1.350774779915215], 'properties': {'magmom': -0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [-0.0, 0.21736945, -0.0],
                              'xyz': [-1.350774779915215, 1.350774779915215, -1.350774779915215],
                              'properties': {'magmom': -0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.78263055, 0.78263055, 0.78263055],
             'xyz': [-4.863413920084786, -4.863413920084786, -4.863413920084786], 'properties': {'magmom': -0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.21736945, 0.0, 0.0],
                              'xyz': [-1.350774779915215, -1.350774779915215, 1.350774779915215],
                              'properties': {'magmom': -0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.0, 0.0, -0.0], 'xyz': [0.0, 0.0, 0.0],
             'properties': {'magmom': 0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [-0.0, 0.35932015, 0.22356876],
             'xyz': [-0.8435847537472929, 0.8435847537472929, -3.6221816778773173], 'properties': {'magmom': 0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.64067985, 0.86424861, 0.64067985],
                              'xyz': [-5.370603946252707, -2.592007022122684, -5.370603946252707],
                              'properties': {'magmom': 0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.77643124, 0.77643124, 0.13575139],
             'xyz': [-8.806195722122682, -0.843584753747293, -0.843584753747293], 'properties': {'magmom': 0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.77643124, 0.13575139, 0.77643124],
                              'xyz': [-0.8435847537472929, -8.806195722122684, -0.8435847537472929],
                              'properties': {'magmom': 0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [-0.0, 0.22356876, 0.35932015],
             'xyz': [0.8435847537472929, -0.8435847537472929, -3.6221816778773173], 'properties': {'magmom': 0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.64067985, 0.64067985, 0.86424861],
                              'xyz': [-2.592007022122684, -5.370603946252707, -5.370603946252707],
                              'properties': {'magmom': 0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.13575139, 0.77643124, 0.77643124],
             'xyz': [-0.8435847537472929, -0.8435847537472929, -8.806195722122684], 'properties': {'magmom': 0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.22356876, -0.0, 0.35932015],
                              'xyz': [0.8435847537472929, -3.6221816778773173, -0.8435847537472929],
                              'properties': {'magmom': 0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.35932015, -0.0, 0.22356876],
             'xyz': [-0.8435847537472929, -3.6221816778773173, 0.8435847537472929], 'properties': {'magmom': 0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.86424861, 0.64067985, 0.64067985],
                              'xyz': [-5.370603946252707, -5.370603946252707, -2.592007022122684],
                              'properties': {'magmom': 0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.35932015, 0.22356876, -0.0],
             'xyz': [-3.6221816778773173, -0.8435847537472929, 0.8435847537472929], 'properties': {'magmom': 0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.22356876, 0.35932015, -0.0],
                              'xyz': [-3.6221816778773173, 0.8435847537472929, -0.8435847537472929],
                              'properties': {'magmom': 0.0}, 'label': 'Si'}]})


def test_energy_forces(clean_dir, test_dir):
    parent_dir = os.getcwd()
    os.chdir(test_dir / "data" / "ref_data")
    energy_forces = plot_energy_forces(
        title="regularization 0.1",
        energy_limit=0.0005,
        force_limit=0.15,
        train_name='train_Si.extxyz',
        test_name='test_Si.extxyz'
    )

    assert os.path.isfile("energy_forces_Si.pdf")
    assert os.path.isfile("energy_forces_Si.png")

    for file_name in os.listdir(os.getcwd()):
        if file_name not in ['train_Si.extxyz', 'test_Si.extxyz', 'quip_train_Si.extxyz', 'quip_test_Si.extxyz']:
            os.remove(os.path.join(os.getcwd(), file_name))

    os.chdir(parent_dir)


def test_energy_plot(test_dir, clean_dir):
    parent_dir = os.getcwd()
    os.chdir(test_dir / "data" / "ref_data")
    energy = energy_plot("train_Si.extxyz", "quip_train_Si.extxyz", ax_list, "Energy on training data")
    plt.savefig("test_energy.png")
    assert os.path.isfile("test_energy.png")


def test_force_plot(test_dir, clean_dir):
    parent_dir = os.getcwd()
    os.chdir(test_dir / "data" / "ref_data")
    force = force_plot("train_Si.extxyz", "quip_train_Si.extxyz", ax_list, "Si", "Force on training data - Si", )
    plt.savefig("test_force.png")
    assert os.path.isfile("test_force.png")

    for file_name in os.listdir(os.getcwd()):
        if file_name not in ['train_Si.extxyz', 'test_Si.extxyz', 'quip_train_Si.extxyz', 'quip_test_Si.extxyz']:
            os.remove(os.path.join(os.getcwd(), file_name))

    os.chdir(parent_dir)


def test_filter_outliers(test_dir, clean_dir):
    parent_dir = os.getcwd()
    os.chdir(test_dir / "data" / "ref_data")
    filter_energy = filter_outlier_energy("train_Si.extxyz", "quip_train_Si.extxyz", 0.005)
    filter_forces = filter_outlier_forces("train_Si.extxyz", "quip_train_Si.extxyz", "Si", 0.1)

    assert os.path.isfile("filtered_in_energy_Si.extxyz")
    assert os.path.isfile("filtered_out_energy_Si.extxyz")
    assert os.path.isfile("outliers_energy_Si.extxyz")
    assert os.path.isfile("filtered_in_force_Si.extxyz")
    assert os.path.isfile("filtered_out_force_Si.extxyz")
    assert os.path.isfile("outliers_force_Si.extxyz")

    for file_name in os.listdir(os.getcwd()):
        if file_name not in ['train_Si.extxyz', 'test_Si.extxyz', 'quip_train_Si.extxyz', 'quip_test_Si.extxyz']:
            os.remove(os.path.join(os.getcwd(), file_name))

    os.chdir(parent_dir)


def test_supercell_check(mp_1200830):
    import warnings
    expected_matrix = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]  # this is not a matrix from reduce_supercell_size

    try:  # cubic, prefer 90
        new_matrix = reduce_supercell_size(
            min_length=18,
            max_length=25,
            max_atoms=500,
            limit=15,
            structure=mp_1200830
        )
    except ValueError:
        warnings.warn(
            message="Falling back to a simple supercell size schema. "
                    "Check if this is ok for your use case.",
            stacklevel=2,
        )
        new_matrix = generate_supercell_matrix(
            mp_1200830, [[3, 0, 0], [0, 3, 0], [0, 0, 3]], max_sites=500
        )

    assert new_matrix == expected_matrix


def test_update_phonon_displacement_maker(memory_jobstore, mp_1203790, clean_dir):
    from atomate2.vasp.sets.core import StaticSetGenerator
    from autoplex.data.phonons.flows import TightDFTStaticMakerBigSupercells

    lattice_avg = sum(mp_1203790.lattice.abc) / 3

    update = update_phonon_displacement_maker(lattice_avg, TightDFTStaticMakerBigSupercells())

    expected = TightDFTStaticMakerBigSupercells(
        name='dft phonon static big supercell',
        input_set_generator=StaticSetGenerator(
            user_incar_settings={
                'IBRION': -1,
                'ISPIN': 1,
                'ISMEAR': 0,
                'ISIF': 3,
                'ENCUT': 700,
                'EDIFF': 1e-07,
                'LAECHG': False,
                'LREAL': False,
                'ALGO': 'Normal',
                'NSW': 0,
                'LCHARG': False,
                'SIGMA': 0.05,
                'ISYM': 0,
                'SYMPREC': 1e-09},
            user_kpoints_settings={'reciprocal_density': 155},  # this is the update we are checking for
            user_potcar_settings={},
            user_potcar_functional=None,
            auto_ismear=True,
            auto_ispin=False,
            auto_lreal=False,
            auto_kspacing=False,
            auto_metal_kpoints=True,
            constrain_total_magmom=False,
            validate_magmom=True,
            use_structure_charge=False,
            sort_structure=True,
            force_gamma=True,
            symprec=0.1,
            vdw=None,
            config_dict={
                'INCAR':
                    {'ALGO': 'Fast',
                     'EDIFF': 1e-05,
                     'EDIFFG': -0.02,
                     'ENAUG': 1360,
                     'ENCUT': 680,
                     'GGA': 'PS',
                     'IBRION': 2,
                     'ISIF': 3,
                     'ISPIN': 2,
                     'ISMEAR': 0,
                     'LORBIT': 11,
                     'LASPH': True,
                     'LDAU': True,
                     'LDAUJ': {'F': {'Co': 0, 'Cr': 0, 'Fe': 0, 'Mn': 0, 'Mo': 0, 'Ni': 0, 'V': 0, 'W': 0},
                               'O': {'Co': 0, 'Cr': 0, 'Fe': 0, 'Mn': 0, 'Mo': 0, 'Ni': 0, 'V': 0, 'W': 0}},
                     'LDAUL': {'F': {'Co': 2, 'Cr': 2, 'Fe': 2, 'Mn': 2, 'Mo': 2, 'Ni': 2, 'V': 2, 'W': 2},
                               'O': {'Co': 2, 'Cr': 2, 'Fe': 2, 'Mn': 2, 'Mo': 2, 'Ni': 2, 'V': 2, 'W': 2}},
                     'LDAUTYPE': 2,
                     'LDAUU': {
                         'F': {'Co': 3.32, 'Cr': 3.7, 'Fe': 5.3, 'Mn': 3.9, 'Mo': 4.38, 'Ni': 6.2, 'V': 3.25, 'W': 6.2},
                         'O': {'Co': 3.32, 'Cr': 3.7, 'Fe': 5.3, 'Mn': 3.9, 'Mo': 4.38, 'Ni': 6.2, 'V': 3.25,
                               'W': 6.2}},
                     'LDAUPRINT': 1,
                     'LREAL': False,
                     'LMIXTAU': True,
                     'LCHARG': True,
                     'LAECHG': True,
                     'LELF': False,
                     'LWAVE': False,
                     'LVTOT': True,
                     'MAGMOM': {'Ce': 5, 'Ce3+': 1, 'Co': 0.6, 'Co3+': 0.6, 'Co4+': 1, 'Cr': 5, 'Dy3+': 5, 'Er3+': 3,
                                'Eu': 10, 'Eu2+': 7, 'Eu3+': 6, 'Fe': 5, 'Gd3+': 7, 'Ho3+': 4, 'La3+': 0.6, 'Lu3+': 0.6,
                                'Mn': 5, 'Mn3+': 4, 'Mn4+': 3, 'Mo': 5, 'Nd3+': 3, 'Ni': 5, 'Pm3+': 4, 'Pr3+': 2,
                                'Sm3+': 5, 'Tb3+': 6, 'Tm3+': 2, 'V': 5, 'W': 5, 'Yb3+': 1},
                     'NELM': 200,
                     'NSW': 99,
                     'PREC': 'Accurate',
                     'SIGMA': 0.05},
                'KPOINTS': {'reciprocal_density': 64, 'reciprocal_density_metal': 200},
                'POTCAR_FUNCTIONAL': 'PBE_54',
                'POTCAR': {'Ac': 'Ac', 'Ag': 'Ag', 'Al': 'Al', 'Am': 'Am', 'Ar': 'Ar', 'As': 'As', 'At': 'At',
                           'Au': 'Au', 'B': 'B', 'Ba': 'Ba_sv', 'Be': 'Be', 'Bi': 'Bi_d', 'Br': 'Br', 'C': 'C',
                           'Ca': 'Ca_sv', 'Cd': 'Cd', 'Ce': 'Ce', 'Cf': 'Cf', 'Cl': 'Cl', 'Cm': 'Cm', 'Co': 'Co',
                           'Cr': 'Cr_pv', 'Cs': 'Cs_sv', 'Cu': 'Cu', 'Dy': 'Dy_3', 'Er': 'Er_3', 'Eu': 'Eu_2', 'F': 'F',
                           'Fe': 'Fe', 'Fr': 'Fr_sv', 'Ga': 'Ga_d', 'Gd': 'Gd_3', 'Ge': 'Ge_d', 'H': 'H', 'He': 'He',
                           'Hf': 'Hf_pv', 'Hg': 'Hg', 'Ho': 'Ho_3', 'I': 'I', 'In': 'In_d', 'Ir': 'Ir', 'K': 'K_sv',
                           'Kr': 'Kr', 'La': 'La', 'Li': 'Li_sv', 'Lu': 'Lu_3', 'Mg': 'Mg', 'Mn': 'Mn_pv',
                           'Mo': 'Mo_sv', 'N': 'N', 'Na': 'Na_pv', 'Nb': 'Nb_sv', 'Nd': 'Nd_3', 'Ne': 'Ne', 'Ni': 'Ni',
                           'Np': 'Np', 'O': 'O', 'Os': 'Os', 'P': 'P', 'Pa': 'Pa', 'Pb': 'Pb_d', 'Pd': 'Pd',
                           'Pm': 'Pm_3', 'Po': 'Po_d', 'Pr': 'Pr_3', 'Pt': 'Pt', 'Pu': 'Pu', 'Ra': 'Ra_sv',
                           'Rb': 'Rb_sv', 'Re': 'Re', 'Rh': 'Rh_pv', 'Rn': 'Rn', 'Ru': 'Ru_pv', 'S': 'S', 'Sb': 'Sb',
                           'Sc': 'Sc_sv', 'Se': 'Se', 'Si': 'Si', 'Sm': 'Sm_3', 'Sn': 'Sn_d', 'Sr': 'Sr_sv',
                           'Ta': 'Ta_pv', 'Tb': 'Tb_3', 'Tc': 'Tc_pv', 'Te': 'Te', 'Th': 'Th', 'Ti': 'Ti_sv',
                           'Tl': 'Tl_d', 'Tm': 'Tm_3', 'U': 'U', 'V': 'V_sv', 'W': 'W_sv', 'Xe': 'Xe', 'Y': 'Y_sv',
                           'Yb': 'Yb_3', 'Zn': 'Zn', 'Zr': 'Zr_sv'}},
            inherit_incar=False,
            lepsilon=False,
            lcalcpol=False),
        write_input_set_kwargs={},
        copy_vasp_kwargs={},
        run_vasp_kwargs={'handlers': ()},
        task_document_kwargs={},
        stop_children_kwargs={},
        write_additional_data={}
    )
    assert update == expected
