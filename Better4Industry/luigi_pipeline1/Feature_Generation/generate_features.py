import luigi
import pandas as pd
from Data_Prep.generate_base_data import GenerateBase

# Generate Features
class GenerateFeatures(luigi.Task):
    current_dir = luigi.Parameter()
    config = luigi.DictParameter()

    def requires(self):
        return GenerateBase(self.current_dir, self.config)

    def output(self):
        return luigi.LocalTarget(self.current_dir + self.config['feature_file'])

    def is_zero(self, val):
        if val == 0:
            return 1
        return 0

    def run(self):
        df = pd.read_csv(self.input().path)
        print(df.columns)
        new_df = df[list(self.config['origin_cols'])]
        new_df['is_zero_duration'] = df.apply(lambda r: self.is_zero(r['duration']), axis=1)
        new_df['avg_vol_per_ct'] = df['avg_vol']/df['avg_ct']
        new_df.to_csv(self.output().path, index=False)
        return new_df
