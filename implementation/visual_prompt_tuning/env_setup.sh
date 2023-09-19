conda create -n prompt python=3.11
conda activate prompt

pip install tensorflow
pip install tensorflow-datasets
pip install opencv-python
pip install tensorflow-addons
pip install mock


conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

#python -m pip install git+https://github.com/facebookresearch/detectron2.git
pip install opencv-python

conda install tqdm pandas matplotlib seaborn scikit-learn scipy simplejson termcolor
conda install tqdm pandas matplotlib seaborn scikit-learn scipy simplejson termcolor jupyter
conda install -c conda-forge iopath fvcore


# for transformers
pip install timm
pip install ml-collections

pip install chardet
