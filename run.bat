if not exist result mkdir result

python main.py --config ./config.yaml --scenario org
################################################################
python main.py --config ./config.yaml --scenario batch_2

python main.py --config ./config.yaml --scenario batch_4

python main.py --config ./config.yaml --scenario batch_8

python main.py --config ./config.yaml --scenario batch_16
################################################################

python main.py --config ./config.yaml --scenario quant_64_64

