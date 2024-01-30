from setuptools import find_packages, setup
from typing import List

HYPHEN_E_Dot = "-e ."

def get_requirements(file_path: str)->List[str]:
    '''
    This function will return the list of requirements
    '''

    requirements = []
    with open(file_path) as file_obj:
        requirements =file_obj.readlines()
        requirements = [req.replace('\n','')for req in requirements]
        
        if HYPHEN_E_Dot in requirements:
            requirements.remove(HYPHEN_E_Dot)

    return requirements


setup(
    name= 'LLM Chat',
    version = '0.0.1',
    author = 'Arjit',
    author_email = 'devgupta0304@gmail.com',
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt')
)