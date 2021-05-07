import argparse
from training import train
from predicting import predict


def main(command_line=None):
    images_width = 80
    images_height = 80
    
    parser = argparse.ArgumentParser('Sematic Recognition by BoysGang')
    
    subprasers = parser.add_subparsers(dest='command')
    
    train_parser = subprasers.add_parser('train', help='train model for image classification')
    train_parser.add_argument('data_path', help='folder with training data, ' +
                'images of each class should be stored in a subfolder named after this class')
    train_parser.add_argument('model_path', help='file to store trained model')
    train_parser.add_argument('--width', help='width to which the images will be resized')
    train_parser.add_argument('--height', help='height to which the images will be resized')
    
    predict_parser = subprasers.add_parser('predict', help='classify image by trained model')
    predict_parser.add_argument('img_path', help='path to the image for classification')
    predict_parser.add_argument('model_path', help='path to trained model')
    predict_parser.add_argument('--width', help='width to which the images will be resized')
    predict_parser.add_argument('--height', help='height to which the images will be resized')
    
    args = parser.parse_args(command_line)
    
    if args.command == "train":
        if args.width:
            images_width = int(args.width)
        
        if args.height:
            images_height = int(args.height)

        train(args.data_path, args.model_path, images_width, images_height)
    
    if args.command == 'predict':
        if args.width:
            images_width = int(args.width)
        
        if args.height:
            images_height = int(args.height)

        predict(args.model_path, args.img_path, images_width, images_height)
        

if __name__ == '__main__':
    main()