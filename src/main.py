import argparse
from training import train
from predicting import predict

def main():
    parser = argparse.ArgumentParser(description='Sematic Recognition by BoysGang')
    parser.add_argument('--train', nargs=4, metavar=('DATA_PATH', 'MODEL_PATH', 'WIDTH', 'HEIGHT'), help='Train model for image classification')
    parser.add_argument('--predict', nargs=4, metavar=('IMG_PATH', 'MODEL_PATH', 'WIDTH', 'HEIGHT'), help='Classify image by trained model')
    
    args = parser.parse_args()._get_kwargs()[0]

    if parser.parse_args().train:
        data_path = args[1][0]
        model_path = args[1][1]
        width = int(args[1][2])
        height = int(args[1][3])

        train(data_path, model_path, width, height)

    elif parser.parse_args().predict:
        img_path = args[1][0]
        model_path = args[1][1]
        width = int(args[1][2])
        height = int(args[1][3])

        predict(model_path, img_path, width, height)

if __name__ == '__main__':
    main()