# echo "initializing conda"
# echo ""

# module load anaconda
# conda init bash
# source ~/.bashrc

echo ""
echo "creating temporary conda environment"
echo ""

conda create -y -n temp python=3.11 -c conda-forge
conda activate temp

echo ""
echo "installing mamba"
echo ""

conda install -y -c conda-forge mamba

echo ""
echo "mamba installation finished, setting up enviornment"
echo ""

mamba env create -y -f chm4411l.yml

echo ""
echo "cleaning up temp environment"
echo ""

conda deactivate
conda remove -n temp --all -y

echo ""
echo "environment setup finished"
echo "to use this course's conda environment, enter the following command:"
echo "conda activate chm4411l"
