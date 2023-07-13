


## for creating the virtual envorontment

```
    conda env create -f environment.yml
    conda activate <conda env name>
    conda activate myenv
    conda install -n fastapi-gempa -c conda-forge prophet #update keknya
    conda remove <package name>
    conda env update -f environment.yml
    conda update -n base -c defaults conda or conda install conda=23.1.0
    https://anaconda.org/search?q=pystan
    cmdstanpy == 0.9.5
    conda remove --name fastapi-gempa --all
    conda create --name myenv
    conda info --envs #check all envs
    uvicorn main:app --reload
```