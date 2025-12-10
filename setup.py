from setuptools import find_packages,setup
from typing import List

h_e_dot = '-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    This give required librarys to install
    '''
    reqirements = []
    with open(file_path) as file_obj:
        reqirements = file_obj.readlines()
        reqirements = [req.replace("\n","") for req in reqirements]
        if h_e_dot in reqirements:
            reqirements.remove(h_e_dot)
        
    return reqirements

setup(
    name="Mlproject",
    version='0.0.1',
    author='zaheer',
    author_email='zaheer897778351@gmail.com',
    packages=find_packages(),
    install_requirements = get_requirements('req.txt')
)