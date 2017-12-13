clear;

#Needed for awgn
pkg load communications;

#Write out files? Or just generate plots?
writeFiles = true

#Signal parameters
Fs = 1e6;
N = 65536;
n = 0:N-1;
freqArr = [-1e5 -2e5 3.4e5 3.5e5 3.6e5];
ampArr = [1 1 1 1 1];

#Generate sin waves
I = zeros(1, N);
Q = zeros(1, N);
for i = 1:length(freqArr)
  I += ampArr(i)/length(freqArr) .* cos(2*pi.*n*freqArr(i)/Fs);
  Q += ampArr(i)/length(freqArr) .* sin(2*pi.*n*freqArr(i)/Fs);
end  

#Combine into complex pairs
arr = awgn(complex(I,Q), 30);

#Poor Decimation (Figure 3 from report)
#arr = downsample(arr, 2);
#Fs = Fs/2;

#Proper Decimation (Figure 4from report)
#arr = decimate(arr, 2, 64, "fir");
#Fs = Fs/2;

#Generate FFT plots
f_axis = linspace(-Fs/2,Fs/2,length(arr));
freq_spec = fftshift(20*log10( abs(fft(arr))/length(arr) ));

#Plot figures
plot(f_axis, freq_spec);
xlim([-Fs/2 Fs/2]);
ylim([-100 0]);
xlabel('Frequency (Hz)');
ylabel('Amplitude (dB)');

#Write files
if writeFiles
  fileName = "inputIQ.txt";
  fid = fopen(fileName, 'w+');
  for i = 1:N
    fprintf(fid, "%f,%f\r\n", real(arr(i)), imag(arr(i)));
  end
  fclose(fid);
end  
