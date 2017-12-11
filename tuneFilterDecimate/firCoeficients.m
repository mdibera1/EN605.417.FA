clear;

#Required for fir1
pkg load signal;

#Points to be plotted - freq response
len = 4096; 

#FIR filter characterization
decimation_rate = 2;
f2 = 1/decimation_rate;   #Stop
f1 = 0.9*f2;              #Pass
N = 63;                   #Taps

#Plot parameters
Fs = 2;                   
f_axis = linspace(-Fs/2,Fs/2,len);

#Generate FIR filter
f = [0 f1 f2 1];
m = [1 1 0 0];
hc1 = fir1(N, f2);
fc1 = abs(fftshift(fft(hc1,len)));

#Plot primary filter response
#
#figure;
plot(f_axis, 20*log10(fc1), [f1 f1], [-10000 10000], [f2 f2], [-10000 10000], 'r');
xlim([-Fs/2 Fs/2]);
ylim([-120 10]);
title('Filter 1 Frequency Response');
legend("Response", "Pass", "Stop", "Quantized");
grid on;
#}

#Write filter coefficients to file
fileName = ["fir_dec_", num2str(decimation_rate), "_taps_",  num2str(N+1), ".txt"];
fid = fopen(fileName, 'w+');
for i = 1:N
  fprintf(fid, "%f\r\n", hc1(i));
end
fprintf(fid, "%f\r\n", 0.0);
fclose(fid);

