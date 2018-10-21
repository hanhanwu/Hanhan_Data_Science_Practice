import luigi

# Read data from data source
class ReadRaw(luigi.ExternalTask):
    current_dir = luigi.Parameter()
    config = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(self.current_dir + self.config['input_file'])
