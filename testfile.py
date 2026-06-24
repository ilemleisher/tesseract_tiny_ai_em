import h5py
import numpy as np
import matplotlib.pyplot as plt
filename='/data/lbl/run43/raw/continuous_I4_D20260610_T165325/cont_I4_D20260610_T165353_F0001.hdf5'
data = h5py.File(filename)

print(data['adc1'].keys())

waveforms = np.array(data['adc1']['event_1'])

tdata =np.arange(waveforms.shape[1])/1.25e6 #Sampling at 1.25 MHz
print(len(tdata))
fig, ax = plt.subplots()

ax.plot(tdata, waveforms[0])
ax.plot(tdata, waveforms[1])
ax.plot(tdata, waveforms[2])
ax.plot(tdata, waveforms[3])

fig.savefig('testfig.png')
