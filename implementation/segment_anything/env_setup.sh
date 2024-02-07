conda create -n segment_anything python=3.10 -y
conda activate segment_anything
conda install pytorch==2.0.0 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -e .
pip install opencv-python pycocotools matplotlib onnxruntime onnx
pip install jupyter
pip install PyYAML
pip install prodict
pip install pandas
pip install scipy
pip install pytest
pip install tqdm
ln -s ../../mmsegmentation/data/ade data/ade
ln -s ../../mmsegmentation/data/cbis data/cbis