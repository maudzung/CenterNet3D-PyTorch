**NGUYEN MAU DUNG**

This is my notes for `spconv` library installation. For more information, please refer the [official spconv repo](https://github.com/traveller59/spconv).

```
git clone https://github.com/traveller59/spconv.git --recursive
sudo apt-get install libboost-all-dev
python setup.py bdist_wheel
cd ./dist
pip install *
```

**Important notes**: Fix bugs during installation:

1. If you are using pytorch 1.4+ and encounter "nvcc fatal: unknown -Wall", let's comment 2 lines in `Caffe2Targets.cmake` of the `torch` package.
2. Fix critical bugs: [this link](https://github.com/traveller59/spconv/issues/78)

in the `spconv` folder, add the below lines to the `setup.py` file:

```     
subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
# ------------------------Start adding here------------------------
build_make_file = 'build/temp.linux-x86_64-3.6/src/spconv/CMakeFiles/spconv.dir/build.make'
link_file = 'build/temp.linux-x86_64-3.6/src/spconv/CMakeFiles/spconv.dir/link.txt'
dlink = 'build/temp.linux-x86_64-3.6/src/spconv/CMakeFiles/spconv.dir/dlink.txt'

for file in [build_make_file, link_file]:

    with open(file) as f:
        newText = f.read().replace('/usr/local/cuda/bin', env['CUDA_ROOT'])

    with open(file, "w") as f:
        f.write(newText)

for file in [build_make_file, link_file]:
    with open(file) as f:
        newText = f.read().replace('/usr/lib/cuda/lib64/libnvToolsExt.so',
                                   '/usr/lib/x86_64-linux-gnu/libnvToolsExt.so')
    with open(file, "w") as f:
        f.write(newText)

for file in [build_make_file, link_file]:
    with open(file) as f:
        newText = f.read().replace('/usr/lib/cuda/lib64/libcudart.so', '/usr/lib/x86_64-linux-gnu/libcudart.so')
    with open(file, "w") as f:
        f.write(newText)

for file in [build_make_file, link_file]:
    with open(file) as f:
        newText = f.read().replace('/usr/lib/cuda/lib64/libculibos.a', '/usr/lib/x86_64-linux-gnu/libculibos.a')
    with open(file, "w") as f:
        f.write(newText)

for file in [dlink]:
    with open(file) as f:
        newText = f.read().replace('/usr/local/cuda/lib64/libculibos.a',
                                   '/usr/lib/x86_64-linux-gnu/libculibos.a')
    with open(file, "w") as f:
        f.write(newText)
# ------------------------End adding here------------------------
subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)
```

Add the below line to `~/.bashrc.sh`:
```
export CUDA_ROOT=/usr/local/cuda/bin
```

then `source ~/.bashrc.sh`

Done!