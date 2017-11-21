clear;

Fs = 1e6;
N = 4096;
n = 0:N-1;
f1 = Fs/4;
f2 = Fs/8;
taps = 64

I = Q = zeros(1,N);

fileName = "outputIQ.txt";
#fileName = "inputIQ.txt";
data = csvread(fileName);

f_axis = linspace(-Fs/2,Fs/2,length(data));
arr = complex(data(:,1), data(:,2));
freq_spec = fftshift(20*log10( abs(fft(arr))/length(arr) ));

plot(f_axis, freq_spec);
xlabel('Freq (Hz)');
ylabel('dB');
#plot(real(arr(N/2-taps:N/2+taps)));

