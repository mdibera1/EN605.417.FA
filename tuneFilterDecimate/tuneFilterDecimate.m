clear;

#Signal parameters
Fs = 1e6;
N = 65536;
n = 0:N-1;
f_shift = -350000;

#Load IQ from file
fileName = "inputIQ.txt";
fid = fopen(fileName, 'r');
IQ = fscanf(fid, "%f,%f\n", [2 65536]);
fclose(fid);

len = 4096;

#for i = 5:16
  #len = 2**i;
  #Create complex array from file
  arr = complex(IQ(1,1:len), IQ(2,1:len));

  #Start timer
  tic;

  #Frequency shift
  arr_baseband = arr .* (cos(2*pi*f_shift.*(0:len-1)/Fs)+sqrt(-1)*sin(2*pi*f_shift.*(0:len-1)/Fs));

  #Decimate using 64 tap FIR
  arr_dec = decimate(arr_baseband, 2, 64, "fir");

  #Stop timer
  toc;
  
  disp(len);
  disp("");
#end
  
#Generate x-axis
Fs = Fs/2;
f_axis = linspace(-Fs/2,Fs/2,length(arr_dec));

#Show Plot
plot(f_axis, 20*log10(fftshift(abs(fft(arr_dec))./length(arr_dec))));
xlim([-Fs/2 Fs/2]);
ylim([-100 0]);
title('Resulting Spectrum - CPU');
xlabel("Frequency (Hz)");
ylabel("Amplitude (dB)");
