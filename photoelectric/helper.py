import os
def process_all_data(folder):
    filenames = os.listdir(folder) 
    csv_files = [file for file in filenames if file.endswith('.csv')]
    
    wavelengths = []
    frequencies = []
    cutoff_voltages = []
    
    for filename in csv_files:
        wavelength, frequency, cutoff_voltage = process_spectrum(filename)
        wavelengths.append(wavelength)
        frequencies.append(frequency)
        cutoff_voltages.append(cutoff_voltage)

    table = pd.DataFrame({
        'Wavelength (nm)': wavelengths,
        'Frequency (Hz)': frequencies ,
        'Stopping Potential (eV)': cutoff_voltages 
    })
    table.sort_values(by=['Frequency (Hz)']) #make the table easier to read
    return table