#!/usr/bin/env python3

import os
import typer
import hashlib
import collections

from pathlib import Path
from matplotlib import pyplot as plt
from colorama import Fore, Back, Style

from src import PCPartsNet


app = typer.Typer()


@app.command()
def model_summary():
    """
    Display model summary.
    """
    ppn = PCPartsNet()

    print('')
    ppn.summary()
    print('')


@app.command()
def count_samples(
    dataset_directory: Path = typer.Argument(
        'dataset',
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help='Path to dataset directory.',
    ),
):
    """
    Display samples count for all classes belonging to the dataset.
    """

    images = {}
    total = 0

    for name in os.listdir(dataset_directory):
        path = os.path.join(dataset_directory, name)
        if os.path.isdir(path):
            images[name] = len(os.listdir(path))

    print('')
    print(f'{Style.BRIGHT}{Fore.CYAN}{" Samples count ":=^23}{Style.RESET_ALL}')
    for image_class, count in images.items():
        print(f'{Fore.CYAN}{image_class:15} {Fore.GREEN}{count:>7}{Style.RESET_ALL}')
        total += count
    print(f'\n{Back.MAGENTA}{Fore.CYAN}Total samples: {Fore.GREEN}{total:>8}{Style.RESET_ALL}\n')


@app.command()
def prune_dataset(
    dataset_directory: Path = typer.Argument(
        'dataset',
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help='Path to dataset directory.',
    ),
):
    """
    Removes duplicate samples from dataset.
    """

    print('')
    pruned = 0
    for datasetChild in os.listdir(dataset_directory):
        childDir = os.path.join(dataset_directory, datasetChild)
        if os.path.isdir(childDir):
            print(f'{Fore.CYAN}Processing samples from category: {Fore.YELLOW}{datasetChild}{Style.RESET_ALL}')
            sampleHashes = {}
            for sample in os.listdir(childDir):
                if os.path.isfile(os.path.join(childDir, sample)):
                    sampleHashes[sample] = hashlib.md5(open(os.path.join(childDir, sample), 'rb').read()).hexdigest()
            occurrences = collections.Counter(sampleHashes.values())
            duplicates = {key: value for key, value in sampleHashes.items() if occurrences[value] > 1}
            handledDuplicates = []

            # Remove duplicate images
            for fileName, fileHash in duplicates.items():
                if not fileHash in handledDuplicates:
                    for dupName, dupHash in duplicates.items():
                        if fileName != dupName and fileHash == dupHash:
                            print(f'{Fore.LIGHTRED_EX}Removing duplicate: {Fore.MAGENTA}{dupName}{Style.RESET_ALL}')
                            os.remove(os.path.join(childDir, dupName))
                            pruned += 1
                    handledDuplicates.append(fileHash)

            for fileName, fileHash in sampleHashes.items():
                if fileHash not in duplicates.values():
                    ext = fileName.split('.')[-1]
                    newFileName = f'{fileHash}.{ext}'
                    os.rename(os.path.join(childDir, fileName), os.path.join(childDir, newFileName))

    print(f'\n{Fore.GREEN}Pruning complete.{Style.RESET_ALL}')
    print(f'{Fore.CYAN}Deleted {Fore.BLUE}{pruned}{Fore.CYAN} duplicates.{Style.RESET_ALL}\n')


@app.command()
def train(
    epochs: int = typer.Argument(
        min=1,
        help='Number of training epochs.',
    ),
    dataset_directory: Path = typer.Argument(
        'dataset',
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help='Path to dataset directory.',
    ),
    validation_size: float = typer.Option(
        0.05,
        '--validation-size',
        '-l',
        min=0.01,
        max=0.99,
        help='Fraction of samples designated to create validation dataset.',
    ),
    batch_size: int = typer.Option(
        50,
        '--batch-size',
        '-b',
        min=1,
        help='Maximum size of sample batches.',
    ),
    models_directory: Path = typer.Argument(
        'models',
        exists=True,
        file_okay=False,
        readable=True,
        writable=True,
        resolve_path=True,
        help='Path to models directory.',
    ),
    evaluate: bool = typer.Option(
        False,
        '--evaluate',
        '-e',
        help='Evaluate model after the training.',
    ),
    show_chart: bool = typer.Option(
        False,
        '--show-chart',
        '-c',
        help='Shows training chart after done.',
    ),
    verbose: bool = typer.Option(
        False,
        '--verbose',
        '-v',
        help='Verbose mode.',
    ),
):
    """
    Start model training with specified dataset.
    """

    if not os.path.isdir(models_directory):
        os.mkdir(models_directory)

    print('')
    ppn = PCPartsNet()
    print('')
    ppn.load_dataset(
        dataset_directory=dataset_directory,
        validation_size=validation_size,
        batch_size=batch_size,
        verbose=True if verbose > 0 else False,
    )
    print('')

    ppn.train(epochs=epochs, models_directory=models_directory)
    print(f'\n{Fore.CYAN}Saving weights...{Style.RESET_ALL}')
    ppn.save()
    print('')

    if evaluate:
        print(f'{Fore.CYAN}Evaluating model...{Style.RESET_ALL}')
        ppn.evaluate(verbose=1)
        print('')

    if show_chart:
        print(f'{Fore.YELLOW}Displaying training chart. Close chart window to exit application{Style.RESET_ALL}.\n')
        chart = ppn.training_chart()
        plt.figure(chart)
        plt.show()


@app.command()
def evaluate(
    weights_file: Path = typer.Argument(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help='File with weights for the model.',
    ),
    dataset_directory: Path = typer.Argument(
        'dataset',
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help='Path to dataset directory.',
    ),
    validation_size: float = typer.Option(
        0.05,
        '--validation-size',
        '-l',
        min=0.01,
        max=0.99,
        help='Fraction of samples designated to create validation dataset.',
    ),
    batch_size: int = typer.Option(
        50,
        '--batch-size',
        '-b',
        min=1,
        help='Maximum size of sample batches.',
    ),
    verbose: bool = typer.Option(
        False,
        '--verbose',
        '-v',
        help='Verbose mode.',
    ),
):
    """
    Evaluate given weights with specified dataset.
    """

    print('')
    ppn = PCPartsNet()
    print('')
    ppn.load(weights_file)
    print('')
    ppn.load_dataset(
        dataset_directory=dataset_directory,
        validation_size=validation_size,
        batch_size=batch_size,
        verbose=verbose,
    )
    print('')
    ppn.evaluate()
    print('')


@app.command()
def predict(
    weights_file: Path = typer.Argument(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help='File with weights for the model.',
    ),
    image: Path = typer.Argument(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help='Image to run prediction on.',

    ),
    display_confidence_array: bool = typer.Option(
        False,
        '--display-confidence-array',
        '-a',
        help='Display confidence values for all classes.',
    ),
):
    """
    Run prediction on specified image with given weights.
    """

    ppn = PCPartsNet()
    ppn.load(weights_file)

    prediction = ppn.predict(image)

    print(f'\nPrediction on {Fore.MAGENTA}{image}{Style.RESET_ALL} with weights from {Fore.MAGENTA}{weights_file}{Style.RESET_ALL}:\n')

    imageClass, imageConfidence = next(iter(prediction.items()))

    print(f'Recognized class is {Back.GREEN}{Style.BRIGHT} {imageClass} {Style.RESET_ALL} with confidence of {Fore.GREEN}{Style.BRIGHT}{imageConfidence * 100:.2f}%{Style.RESET_ALL}.\n')

    if display_confidence_array:
        print(f'{Style.BRIGHT}{Fore.CYAN}{" CONFIDENCE ARRAY ":=^22}{Style.RESET_ALL}')
        bold = Back.MAGENTA
        for photoClass, probability in prediction.items():
            print(f'{bold}{Fore.CYAN}{photoClass:15} {Fore.GREEN}{probability * 100:5.2f}%{Style.RESET_ALL}')
            bold = ''
        print('')


if __name__ == '__main__':
    app()
