clear;

#Needed for awgn
pkg load communications;

#Signal parameters
Fs = 1e6;
N = 4096;
n = 0:N-1;
freqArr = [10000 410000];
ampArr = [1 1];

#Generate sin waves
I = zeros(1, N);
Q = zeros(1, N);
for i = 1:length(freqArr)
  I += ampArr(i)/length(freqArr) .* cos(2*pi.*n*freqArr(i)/Fs);
  Q += ampArr(i)/length(freqArr) .* sin(2*pi.*n*freqArr(i)/Fs);
end  

#Combine into complex pairs
arr = awgn(complex(I,Q), 30);

#Generate FFT plots
f_axis = linspace(-Fs/2,Fs/2,length(arr));
freq_spec = fftshift(20*log10( abs(fft(arr))/length(arr) ));

#Plot figures
plot(f_axis, freq_spec);
xlabel('Freq (Hz)');
ylabel('dB');

#Write files
fileName = "inputIQ.txt";
fid = fopen(fileName, 'w+');
for i = 1:N
  fprintf(fid, "%f,%f\r\n", real(arr(i)), imag(arr(i)));
end
fclose(fid);
