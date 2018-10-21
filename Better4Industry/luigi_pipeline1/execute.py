import luigi
import yaml
import os
from Feature_Generation.generate_features import GenerateFeatures


# Load config file
def load_config(config_path):
    with open(config_path, 'r') as config_in:
        config = yaml.load(config_in)
    return config


if __name__ == '__main__':
    current_dir = os.getcwd()
    config = load_config('config.yaml')
    feature_file_path = current_dir + config['feature_file']
    try:
        os.remove(feature_file_path)
    except OSError:
        pass
    task = GenerateFeatures(current_dir, config)
    luigi.build([task], workers=1, local_scheduler=True)
