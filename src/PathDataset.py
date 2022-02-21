"""
Created by Philippenko, 16th February 2022.

Customization required to run the code on a remote cluster.
"""

def get_path_to_datasets() -> str:
    """Return the path to the datasets. For sake of anonymization, the path to datasets on clusters is not keep on
    GitHub and must be personalized locally"""
    return "."

def get_path_to_pickle() -> str:
    """"Return the path to the pickle folder. """
    return "."