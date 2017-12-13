#!/bin/bash

nvcc tuneFilterDecimate.cu -o tfd.o
./tfd.o -b 32
./tfd.o -b 64
./tfd.o -b 128
./tfd.o -b 256
./tfd.o -b 512
./tfd.o -b 1024
./tfd.o -b 2048
./tfd.o -b 4096
./tfd.o -b 8192
./tfd.o -b 16384
./tfd.o -b 32768
./tfd.o -b 65535
