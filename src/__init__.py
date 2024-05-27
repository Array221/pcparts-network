import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from colorama import Fore, Style


from matplotlib import pyplot as plt


class PCPartsNet:
    classes = [
        'case',
        'cpu',
        'gpu',
        'monitor',
        'motherboard',
        'mouse',
        'powerSupply',
        'ram',
        'speaker',
    ]

    dataset = {
        'train': None,
        'val': None
    }

    epochs = 0

    history = None

    imageSize = (256, 256)

    model = None

    def __init__(self): # Creates network model
        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, 3, activation='relu', input_shape=self.imageSize + tuple([3])),
            tf.keras.layers.AveragePooling2D(2, 2),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.AveragePooling2D(2, 2),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.AveragePooling2D(2, 2),
            tf.keras.layers.Conv2D(128, 3, activation='relu'),
            tf.keras.layers.AveragePooling2D(2, 2),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(9, activation='softmax'),
        ])
        self.model.compile(
            optimizer = 'adam',
            loss = 'categorical_crossentropy',
            metrics = ['accuracy']
        )

    def summary(self): # Displays model summary
        self.model.summary()

    def load(self, path): # Loads model weights and biases from file
        self.model.load_weights(path)

    def save(self, models_directory = 'models'): # Saves model weights and biases to file
        _, acc = self.model.evaluate(self.dataset['val'])
        path = os.path.join(models_directory, f'e{self.epochs:02d}-acc{acc:.2f}.weights.h5')
        self.model.save_weights(path)
        return path

    # Load dataset in format supported by keras and splits it in two categories (training data and validation data)
    def load_dataset(self, dataset_directory = 'dataset', validation_size = 0.05, batch_size = 50, verbose = False):
        self.dataset['train'], self.dataset['val'] = tf.keras.utils.image_dataset_from_directory(
            directory = dataset_directory,
            labels = 'inferred',
            label_mode = 'categorical',
            batch_size = batch_size,
            image_size = self.imageSize,
            seed = 0,
            validation_split = validation_size,
            subset = 'both',
            verbose = verbose,
        )

    def train(self, epochs, models_directory = 'models'): # Trains model with given parameters
        try:
            self.history = self.model.fit(
                self.dataset['train'],
                epochs = epochs,
                validation_data = self.dataset['val'],
                callbacks = [
                    tf.keras.callbacks.ModelCheckpoint(
                        filepath = os.path.join(models_directory, 'e{epoch:02d}-acc{val_accuracy:.2f}.weights.h5'),
                        save_best_only = True,
                        save_weights_only = True,
                        verbose = 1,
                    ),

                ],
                verbose = 1,
            ).history

            self.epochs = len(self.history['accuracy'])
        except KeyboardInterrupt:
            print(f'\n{Fore.RED}Training aborted{Style.RESET_ALL}\n')
            print(f'{Fore.CYAN}Saving weights...{Style.RESET_ALL}')
            path = self.save()
            print(f'\n{Fore.GREEN}Weights saved to {Fore.MAGENTA}{path}{Style.RESET_ALL}\n')
            exit(1)

    def training_chart(self): # Creates training chart based on training history
        fig, ax = plt.subplots(2)
        fig.tight_layout(pad=4.0)
        epochs = range(1, self.epochs + 1)
        if self.epochs <= 10:
            plt.setp(ax, xticks=epochs)

        ax[0].set_title('Loss')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Loss')
        ax[0].grid()
        ax[0].plot(
            epochs,
            self.history['loss'],
            color='green',
            marker='.'
        )
        ax[0].plot(
            epochs,
            self.history['val_loss'],
            color='red',
            marker='.'
        )

        ax[1].set_title('Accuracy')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Accuracy')
        ax[1].grid()
        ax[1].plot(
            epochs,
            self.history['accuracy'],
            color='green',
            marker='.'
        )
        ax[1].plot(
            epochs,
            self.history['val_accuracy'],
            color='red',
            marker='.'
        )

        fig.legend(['Training values', 'Validation values'], loc='lower center', ncols=2)
        return fig

    def evaluate(self, verbose = 1): # Evaluates model with given dataset
        return self.model.evaluate(self.dataset['val'], verbose = verbose)

    def predict(self, image_path): # Run preditcion on selected image
        image = tf.keras.utils.load_img(image_path, target_size=self.imageSize)
        image = tf.keras.utils.img_to_array(image)
        image = tf.expand_dims(image, 0)
        pred = self.model.predict(image, batch_size = 1, verbose = 0)[0]
        results = {}
        for i in range(0, len(self.classes)):
            results[self.classes[i]] = pred[i]
        return dict(sorted(results.items(), key = lambda x:x[1], reverse = True))
