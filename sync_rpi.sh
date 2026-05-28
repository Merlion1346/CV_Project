#!/bin/bash
RPI=seong@10.1.217.103

scp model_int8.onnx $RPI:~/rpi_app/model/
