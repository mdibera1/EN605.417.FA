clear;

#Sampling frequency
decimationFactor = 1;
Fs = 1e6 / decimationFactor;

#Select input file
fileName = "outputIQ.txt";
#fileName = "inputIQ.txt";
data = csvread(fileName);

#Generate FFT plot
f_axis = linspace(-Fs/2,Fs/2,length(data));
arr = complex(data(:,1), data(:,2));
freq_spec = fftshift(20*log10( abs(fft(arr))/length(arr) ));

#Show plot
plot(f_axis, freq_spec);
xlabel('Freq (Hz)');
ylabel('dB');
#plot(real(arr(N/2-taps:N/2+taps)));

