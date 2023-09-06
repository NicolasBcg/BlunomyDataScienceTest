import os
os.system('python -m pip install --upgrade pip')
requirements=['pandas','numpy','scikit-learn','matplotlib']
for requirement in requirements:
    os.system('pip install '+requirement)