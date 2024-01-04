from pathlib import Path
import signal

from nni.experiment import Experiment

# Define search space
search_space = {
    #'regularizer_multiplier': {'_type': 'choice', '_value': [0.3, 1, 3, 5, 10]},
    #'mb_dim': {'_type': 'choice', '_value' : [8, 16, 32, 64, 128]},
    #'iteration': {'_type': 'choice', '_value' : [5, 10, 20, 40, 80, 160]},
    'regularizer_multiplier': {'_type': 'choice', '_value': [0.5, 1, 1.5]},
    'mb_dim': {'_type': 'choice', '_value' : [16, 32]},
    'iteration': {'_type': 'choice', '_value' : [40, 80]},
    #'regularizer_multiplier': {'_type': 'uniform', '_value': [0.1, 10]},

}

# Configure experiment
experiment = Experiment('local')
experiment.config.trial_command = 'python Muesli_LunarLander.py'
experiment.config.trial_code_directory = Path(__file__).parent
experiment.config.search_space = search_space
experiment.config.tuner.name = 'Random'
experiment.config.max_trial_number = 150
experiment.config.trial_concurrency = 15


# Run it!
experiment.run(port=8080, wait_completion=False)

print('Experiment is running. Press Ctrl-C to quit.')
signal.pause()