import configparser
import polars as pl


def write_csv(data:dict, path:str):
    df = pl.DataFrame(data)
    df.write_csv(path)

def write_meta_data_ini(file: dict, path:str):

    config_parser = configparser.ConfigParser()
    config_parser.read_dict({'EXPERIMENT':file})
    with open(path, 'w') as configfile:
        config_parser.write(configfile)

def write_meta_data_txt(file: dict, path:str):
    with open(path, 'w') as f:
        f.write(str(file))


###Based on the file extension in the metadata, we will use a different function to write our data.

class IORegistry: 

    ##This allows us to save files in multiple formats.
    io_registry = {

    "csv": write_csv,
}
    
    @classmethod
    def get_write_fn(cls, extension: str):
        return cls.io_registry.get(extension, 'csv')


### Path handling: I need to also write some utility functions to handle paths and create them

###For each type of experiment, we create the right way to make it work