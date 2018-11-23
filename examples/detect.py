#!/usr/bin/env python
import os.path
import pkg_resources
import sys

import click
import cv2


XML_PATH = pkg_resources.resource_filename('lbpcascade_animeface', "lbpcascade_animeface.xml")


@click.command()
@click.argument('filename')
@click.option(
    '--cascade-file', default="lbpcascade_animeface.xml", help='Cascade file.')
@click.option('--write/--no-write', default=True, help='Write output to file')
@click.option('--dst-file', default='out.png', help='File destination.')
@click.option('--show/--no-show', default=True, help='Show result.')
def cli(
        filename, cascade_file="lbpcascade_animeface.xml",
        write=True, dst_file='out.png', show=True):
    detect(filename, cascade_file, write, dst_file, show)

    
def detect(
        filename, cascade_file="lbpcascade_animeface.xml",
        write=True, dst_file='out.png', show=True):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    cascade = cv2.CascadeClassifier(cascade_file)
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    faces = cascade.detectMultiScale(gray,
                                     # detector options
                                     scaleFactor = 1.1,
                                     minNeighbors = 5,
                                     minSize = (24, 24))
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    if show:
        cv2.imshow("AnimeFaceDetect", image)
        cv2.waitKey(0)
    if write:
        cv2.imwrite(dst_file, image)
    return {
        'faces': faces,
        'cascade': cascade,
        'image': image,
    }


if __name__ == '__main__':
    cli()
