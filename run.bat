@REM 1. Models
python ./main.py --config ./config.yaml --scenario org
python ./main.py --config ./config.yaml --scenario vggnet
python ./main.py --config ./config.yaml --scenario resnet
python ./main.py --config ./config.yaml --scenario ffn

@REM 2. Batch Sizes
python ./main.py --config ./config.yaml --scenario batch_2
python ./main.py --config ./config.yaml --scenario batch_4
python ./main.py --config ./config.yaml --scenario batch_8
python ./main.py --config ./config.yaml --scenario batch_16

@REM 3. Optimizers
python ./main.py --config ./config.yaml --scenario optim_Adam
python ./main.py --config ./config.yaml --scenario optim_AdamW
python ./main.py --config ./config.yaml --scenario optim_SGD

@REM 4. Noise Levels
python ./main.py --config ./config.yaml --scenario noise_1
python ./main.py --config ./config.yaml --scenario noise_2
python ./main.py --config ./config.yaml --scenario noise_3

@REM 5. Label Setting
python ./main.py --config ./config.yaml --scenario have_label

@REM 6. Quantization (Precision)
python ./main.py --config ./config.yaml --scenario quant_64_64
python ./main.py --config ./config.yaml --scenario quant_16_16

@REM 7. Iterations
python ./main.py --config ./config.yaml --scenario iter_100
python ./main.py --config ./config.yaml --scenario iter_200
python ./main.py --config ./config.yaml --scenario iter_400