echo "Loading data..."
rsync -rv lille.g5k:/home/alecoutre/rasta/savings/* ../savings/
tensorboard --logdir=../savings
