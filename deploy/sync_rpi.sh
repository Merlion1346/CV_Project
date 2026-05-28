#!/bin/bash
RPI=seong@10.1.217.103

rsync -avz --progress \
    "$(dirname "$0")/rpi_app/" \
    $RPI:~/rpi_app/
