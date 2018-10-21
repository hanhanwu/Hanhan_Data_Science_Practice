import luigi
import pandas as pd
from Data_Prep.task_read_raw import ReadRaw


# Generate base data
class GenerateBase(luigi.Task):
    current_dir = luigi.Parameter()
    config = luigi.DictParameter()

    def requires(self):
        return ReadRaw(self.current_dir, self.config)

    def output(self):
        return luigi.LocalTarget(self.current_dir + self.config['base_file'])

    def get_avg(self, total, ct):
        if ct == 0:
            return -1
        return total/ct

    def run(self):
        df = pd.read_csv(self.current_dir + self.config['input_file'])
        df['duration'] = df['months_since_first'] - df['months_since_last']
        df['avg_ct'] = df.apply(lambda r: self.get_avg(r['donation_ct'], r['duration']), axis=1)
        df['avg_vol'] = df.apply(lambda r: self.get_avg(r['volume'], r['duration']), axis=1)
        df.to_csv(self.output().path, index=False)
        return df
