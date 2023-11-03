#conda create -n openmmlab python=3.10 -y
#conda activate openmmlab
conda install pytorch==2.0.0 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
pip install -v -e .
pip install jupyter
pip install ftfy
pip install regex
pip install -r requirements.txt
pip install -r requirements/tests.txt
pip install mim
pip install scikit-learn
