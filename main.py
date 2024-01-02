from pathlib import Path
import signal

from nni.experiment import Experiment

# Define search space
search_space = {
    'regularizer_multiplier': {'_type': 'choice', '_value': [0.3, 1, 3, 5, 10, 20]},
}

# Configure experiment
experiment = Experiment('local')
experiment.config.trial_command = 'python Muesli_LunarLander.py'
experiment.config.trial_code_directory = Path(__file__).parent
experiment.config.search_space = search_space
experiment.config.tuner.name = 'Random'
experiment.config.max_trial_number = 10
experiment.config.trial_concurrency = 8

# Run it!
experiment.run(port=8080, wait_completion=False)

print('Experiment is running. Press Ctrl-C to quit.')
signal.pause()