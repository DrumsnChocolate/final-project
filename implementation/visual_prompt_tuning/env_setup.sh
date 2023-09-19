conda create -n prompt python=3.11
conda activate prompt

pip install tensorflow
pip install tensorflow-datasets
pip install opencv-python
pip install tensorflow-addons
pip install mock


conda install pytorch torchvision torchaudio cudatoolkit -c pytorch

python -m pip install detectron2 -f \
https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html
pip install opencv-python

conda install tqdm pandas matplotlib seaborn scikit-learn scipy simplejson termcolor
conda install tqdm pandas matplotlib seaborn scikit-learn scipy simplejson termcolor jupyter
conda install -c iopath iopath


# for transformers
pip install timm
pip install ml-collections

pip install chardet
