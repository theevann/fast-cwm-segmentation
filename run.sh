mkdir -p /tmp/$(whoami)
export TMPDIR=/tmp/$(whoami)
trap "rm -rf '/tmp/$(whoami)'" EXIT
export TORCH_HOME=~/.torch

dir="$(dirname $(readlink -f $0))"
cd $dir

# cd ~/code/fastnet/

if [ -z "$3" ]
then
    CUDA_VISIBLE_DEVICES=0 python3 run.py --script-dir $1 --config-file $2
else
    CUDA_VISIBLE_DEVICES=0 python3 run.py --script-dir $1 --config-file $2 --eval $3
fi