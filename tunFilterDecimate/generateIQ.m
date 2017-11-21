clear;

Fs = 1e6;
N = 4096;
n = 0:N-1;
f1 = 10000;
f2 = 410000;

I = 0.5*cos(2*pi.*n*f1/Fs) + 0.5*cos(2*pi.*n*f2/Fs);
Q = 0.5*sin(2*pi.*n*f1/Fs) + 0.5*sin(2*pi.*n*f2/Fs);

arr = complex(I,Q);

f_axis = linspace(-Fs/2,Fs/2,length(arr));
freq_spec = fftshift(20*log10( abs(fft(arr))/length(arr) ));

plot(f_axis, freq_spec);
xlabel('Freq (Hz)');
ylabel('dB');

fileName = "inputIQ.txt";
fid = fopen(fileName, 'w+');
for i = 1:N
  fprintf(fid, "%f,%f\r\n", I(i), Q(i));
end
fclose(fid);
