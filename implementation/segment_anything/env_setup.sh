conda create -n segment_anything python=3.10 -y
conda activate segment_anything
conda install pytorch==2.0.0 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -e .
pip install opencv-python pycocotools matplotlib onnxruntime onnx
pip install jupyter