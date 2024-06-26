pip install rlds dm-reverb[tensorflow] flax==0.8.0 jax==0.4.20 optax==0.1.8 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install --upgrade "jax[cuda12_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install scipy==1.10.1 pillow matplotlib ipython ipykernel chardet orbax-checkpoint==0.4.4 wandb -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install tensorflow_datasets tensorflow==2.15.0 tensorflow_hub ml-dtypes==0.2.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia