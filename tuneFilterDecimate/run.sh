#!/bin/bash

nvcc tuneFilterDecimate.cu -o ftd.o
./ftd.o
