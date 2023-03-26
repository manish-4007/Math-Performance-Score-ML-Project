from setuptools import find_packages, setup
from typing import List

HYN_E_DOT = '-e.'
def get_requirements(file_path:str)->List[str]:
    # This function returns the list of requirements 
    requirements=[]
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n', "") for req in requirements]

        if HYN_E_DOT in requirements:
            requirements.remove(HYN_E_DOT)
    return requirements
        
setup(
    name= 'performance_score_mlproject',
    version='0.0.1',
    author='Manish',
    author_email='manish.rai709130@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements ('requirements.txt')

)

