#!/bin/bash
RPI=seong@10.1.217.103

scp models/model_int8.onnx $RPI:~/rpi_app/build/model/model_int8.onnx
