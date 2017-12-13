clear;

Fs = 1e6;
N = 4096;
n = 0:N-1;
f_shift = -350000;

#Load IQ from file
fileName = "inputIQ.txt";
fid = fopen(fileName, 'r');
IQ = fscanf(fid, "%f,%f\n", [2 4096]);
fclose(fid);

#Create complex array from file
arr = complex(IQ(1,:), IQ(2,:));

#Start timer
startTime = cputime;

#Frequency shift
arr_baseband = arr .* (cos(2*pi*f_shift.*n/Fs)+sqrt(-1)*sin(2*pi*f_shift.*n/Fs));

#Decimate using 64 tap FIR
arr_dec = decimate(arr_baseband, 2, 64, "fir");

#Stop timer
stopTime = cputime;
elapsedTime = stopTime - startTime;

#Generate x-axis
Fs = Fs/2;
f_axis = linspace(-Fs/2,Fs/2,length(arr_dec));

#Show Plot
plot(f_axis, 20*log10(fftshift(abs(fft(arr_dec))./length(arr_dec))));
xlim([-Fs/2 Fs/2]);
ylim([-100 10]);
title('Resulting Spectrum - CPU');
xlabel("Frequency (Hz)");
ylabel("Amplitude (dB)");

disp("Execution time (ms): "),
disp(elapsedTime * 1000),