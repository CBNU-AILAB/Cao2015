import time
import argparse

import numpy as np

import torch
import torchvision

from util import load_model, CIFAR10_SpikeGenerator, SpikeCounter


def validate(val_loader, model, generator, counter, time_interval):
    total_images = 0
    num_corrects = 0
    i = 0

    for image, label in val_loader:
        image = image.squeeze(0).detach().cpu().numpy()
        label = label.squeeze(0).detach().cpu().numpy()

        out = []

        for _ in range(time_interval):
            spiked_image = generator(image)

            out.append(model(spiked_image))

        pred = counter(out)

        total_images += 1
        if label == np.argmax(pred):
            num_corrects += 1

        if i % 100 == 0:
            print("step: {} - acc: {}".format(i, num_corrects/total_images))
        i += 1

    val_acc = num_corrects / total_images

    return val_acc


def app(opt):
    print(opt)

    val_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(
            opt.data,
            train=False,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.CenterCrop(24),
                torchvision.transforms.ToTensor()])),
        batch_size=opt.batch_size)

    model = load_model(opt.pretrained)

    criterion = None

    generator = CIFAR10_SpikeGenerator(3, 24, 24)

    counter = SpikeCounter()

    start = time.time()
    val_acc = validate(val_loader, model, generator, counter, opt.time_interval)
    end = time.time()

    print("elapsed: {} - val_acc: {}".format(end-start, val_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', default='data')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--time_interval', default=300, type=int)
    parser.add_argument('--pretrained', default='pretrained/tailored_cnn.pt')

    app(parser.parse_args())
