#!/bin/bash

nvcc tuneFilterDecimate.cu -o tfd.o
./tfd.o
