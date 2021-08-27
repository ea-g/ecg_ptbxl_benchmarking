from experiments.scp_experiment import SCP_Experiment
from configs.wavelet_configs import *
from configs.your_configs import *

datafolder = '../data/ptbxl/'
outputfolder = '../output/'

models = [conf_wavelet_standard_lr, conf_minirocket_standard_lr]

e = SCP_Experiment('wavelet_MR_super', 'superdiagnostic', datafolder, outputfolder, models)
e.prepare()
e.perform()
e.evaluate()
