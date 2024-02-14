# run with %> source create_pyenv.sh 
ml purge
ml use $OTHERSTAGES
ml Stages/2023  GCC/11.3.0  OpenMPI/4.1.4 
ml PyTorch/1.12.0-CUDA-11.7
ml ecCodes/2.27.0
ml xarray
ml zarr

if [ -d ./pyenv ]; then
    source pyenv/bin/activate
else
    echo "Finished loading modules"
    python -m venv pyenv # env is the directory where the environment is stored
    chmod g+w pyenv -R

    # activate env
    source pyenv/bin/activate
    echo "Finished creating python environment"
    # install packages
    pip install --upgrade pip
    pip install pathlib
    pip install ecmwflibs
    pip install cfgrib
    pip install netcdf4
    pip install xarray
    pip install matplotlib
    pip install opencv-python
    pip install cartopy
    pip install tqdm
    
    echo "Finished installing python packages"
    mkdir figures
fi
export PYTHONPATH=./:$PYTHONPATH
