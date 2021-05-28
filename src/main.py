import argparse
import os
import dotenv

from prettytable import PrettyTable


def config():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def print_prediction(labels, probabilities):
    table = PrettyTable(['Class', 'Probability'])
    for i in range(len(labels)):
        table.add_row([labels[i], probabilities[i]])

    print(table)


def main(command_line=None):    
    config = dotenv.dotenv_values('.conf')

    images_width = int(config['IMAGE_WIDTH'])
    images_height = int(config['IMAGE_HEIGHT'])

    color_mode = config['COLOR_MODE']
    validation_split = float(config['VALIDATION_SPLIT'])
    rotation_range = int(config['ROTATION_RANGE'])
    shear_range = int(config['SHEAR_RANGE'])
    horizontal_flip = bool(config['HORIZONTAL_FLIP'])
    batch_size = int(config['BATCH_SIZE'])

    training_data_path = config['TRAINING_DATA_PATH']

    models_path = config['MODELS_PATH']

    max_features = int(config['MAX_FEATURES'])
    clusters_num = int(config['CLUSTERS_NUM'])

    epochs = int(config['EPOCHS'])
    acceleration = bool(config['GPU_ACCELERATION'])

    filter_depth = int(config['FILTER_DEPTH'])
    suggest_depth = int(config['SUGGEST_DEPTH'])
    graphs_path = config['GRAPHS_PATH']

    r = int(config['RADIUS'])
    damping = float(config['DAMPING'])
    threshold = float(config['THRESHOLD'])
    
    # cli parser
    parser = argparse.ArgumentParser(description='Sematic Recognition by BoysGang')
    
    subprasers = parser.add_subparsers(dest='command')
    
    # training
    train_help_msg = 'train model for image classification'
    train_parser = subprasers.add_parser('train', help=train_help_msg, description=train_help_msg)
    train_parser.add_argument('classifier', help='classifier (cnn, bovw)')
    train_parser.add_argument('model_name', help='name for model')
    # train_parser.add_argument('data_path', help='folder with training data, ' +
    #             'images of each class should be stored in a subfolder named after this class')
    # train_parser.add_argument('model_path', help='file to store trained model')
    # train_parser.add_argument('--width',
    #             help=f'width to which the images will be resized, default value is {images_width}',
    #             type=int)
    # train_parser.add_argument('--height',
    #             help=f'height to which the images will be resized, default value is {images_height}',
    #             type=int)
    # train_parser.add_argument('--epochs',
    #             help=f'number of epochs for training, default value is {epochs}',
    #             type=int)
    
    # prediction
    predict_help_msg = 'classify image by trained model'
    predict_parser = subprasers.add_parser('predict', help=predict_help_msg, description=predict_help_msg)
    predict_parser.add_argument('classifier', help='classifier (cnn, bovw)')
    predict_parser.add_argument('model_name', help='name for model')
    predict_parser.add_argument('img_path', help='path to the image for classification')
    predict_parser.add_argument('--graph_name',
            help=f'graph name',)
    predict_parser.add_argument('--correction_method',
            help=f'correction method (PairCorrection, SingleCorrection)',)
    # predict_parser.add_argument('model_path', help='path to trained model')
    # predict_parser.add_argument('--graph_path',
    #             help=f'path to the semantic graph (if defined then semantic correction will be used)',)
    
    # graph
    graph_help_msg = 'semantic graph module' 
    graph_parser = subprasers.add_parser('graph', help=graph_help_msg, description=graph_help_msg)

    graph_subparsers = graph_parser.add_subparsers(dest='command')
    
    # graph filter
    graph_filter_msg = 'optimize the graph for the available classifiers' 
    graph_filter_parser = graph_subparsers.add_parser('filter', help=graph_filter_msg, description=graph_filter_msg)
    graph_filter_parser.add_argument('dictionary_path', help='dictionary of semantic relationships, ' +
                'each line contains an edge of the graph')
    graph_filter_parser.add_argument('graph_name',
            help=f'graph name',)
    # graph_filter_parser.add_argument('data_path', help='folder with training data, ' +
    #             'images of each class should be stored in a subfolder named after this class')
    # graph_filter_parser.add_argument('store_path', help='file path to store optimazed graph')
    # graph_filter_parser.add_argument('--depth', help='depth of the path when looking for ' +
    #             f'relationships between existing classes, default value is {filter_depth}', type=int)
    # graph_filter_parser.add_argument('--image_path', help='path to store graph image')

    # graph suggest
    suggest_msg = 'suggest semantic related classes'
    suggest_parser = graph_subparsers.add_parser('suggest', help=suggest_msg, description=suggest_msg)
    suggest_parser.add_argument('dictionary_path', help='dictionary of semantic relationships, ' +
                'each line contains an edge of the graph')
    suggest_parser.add_argument('classes_list', help='list of classes for which to suggest', nargs='+', type=str)
    suggest_parser.add_argument('--depth', help='depth of the path when looking for ' +
                f'relationships between existing classes, default value is {suggest_depth}', type=int)

    # parse cli arguments
    args = parser.parse_args(command_line)
    
    if args.command == "train":
        from ConvolutionalNeuralNetwork import ConvolutionalNeuralNetwork
        from BagOfVisualWords import BagOfVisualWords
        from ClassifierContext import ClassifierContext
        from ImgDataGenerator import ImgDataGenerator

        # if args.width:
        #     images_width = args.width
        # if args.height:
        #     images_height = args.height
        # if args.epochs:
        #     epochs = args.epochs

        img_data_generator = ImgDataGenerator(
            training_data_path, 
            resize_to=(images_width, images_height), 
            color_mode=color_mode,
            validation_split=validation_split,
            rotation_range=rotation_range,
            shear_range=shear_range,
            horizontal_flip=horizontal_flip,
            batch_size=batch_size
        )

        classifier_context = ClassifierContext()

        if (args.classifier == 'bovw'):
            classifier = BagOfVisualWords(
                max_features=max_features,
                clusters_num=clusters_num
            )
        elif (args.classifier == 'cnn'):
            classifier = ConvolutionalNeuralNetwork(
                epochs=epochs,
                acceleration=acceleration,
            )
        
        classifier_context.set_classifier(classifier)
        classifier_context.fit(img_data_generator)

        classifier_context.save(os.path.join(models_path, args.model_name))
    
    elif args.command == 'predict':
        from ConvolutionalNeuralNetwork import ConvolutionalNeuralNetwork
        from BagOfVisualWords import BagOfVisualWords
        from ClassifierContext import ClassifierContext
        from ImgDataGenerator import ImgDataGenerator

        from SemanticGraph import SemanticGraph
        from SemanticCorrection import SemanticCorrection
        from PairCorrection import PairCorrection
        from SingleCorrection import SingleCorrection

        classifier_context = ClassifierContext()

        model_path = os.path.join(models_path, args.model_name)

        if args.classifier == 'bovw':
            classifier = BagOfVisualWords.load(model_path)
        elif args.classifier == 'cnn':
            classifier = ConvolutionalNeuralNetwork.load(model_path)

        classifier_context.set_classifier(classifier)

        labels, probabilities = classifier_context.predict(args.img_path)

        print('Classification result:')
        print_prediction(labels, probabilities)

        if args.graph_name:
            graph = SemanticGraph.load(os.path.join(graphs_path, args.graph_name + '.pkl'))
            semantic_correction = SemanticCorrection(labels, probabilities, graph)

            if args.correction_method == 'SingleCorrection':
                semantic_correction.set_method(SingleCorrection(r, damping))
            elif args.correction_method == 'PairCorrection':
                semantic_correction.set_method(PairCorrection(r, threshold))

            labels, probabilities = semantic_correction.apply()

            print('\nClassification result after semantic correction:')
            print_prediction(labels, probabilities)

    elif args.command == 'filter':
        from SemanticGraph import SemanticGraph
        from SemanticGraphFilter import SemanticGraphFilter

        # if args.depth:
        #     filter_depth = args.depth

        base = SemanticGraph()
        base.read_from_dictionary(args.dictionary_path)
        classes = os.listdir(training_data_path)

        print("Filtering graph for clasess:", classes)
        
        filtered = SemanticGraphFilter.filter(base, classes, filter_depth)
        filtered.save(os.path.join(graphs_path, args.graph_name))

        print("Graph successfully stored to", graphs_path)

    elif args.command == 'suggest':
        from SemanticGraph import SemanticGraph
        from SemanticGraphFilter import SemanticGraphFilter

        if args.depth:
            suggest_depth = args.depth

        base = SemanticGraph()
        base.read_from_dictionary(args.dictionary_path)
        print("Suggesting for clasess:", args.classes_list)

        SemanticGraphFilter.suggest_classifiers(base, args.classes_list, suggest_depth)


if __name__ == '__main__':
    config()
    main()