"""
Helper functions for the Photoelectric Effect lab.

This module provides utility functions for processing multiple photoelectric
spectra and aggregating results into a DataFrame.
"""

import os
import pandas as pd


def process_all_data(folder, process_spectrum):
    """
    Process all CSV spectrum files in a folder and aggregate results.
    
    Parameters
    ----------
    folder : str
        Path to folder containing CSV spectrum files.
    process_spectrum : callable
        Function with signature process_spectrum(filename, plot=True) that returns
        (wavelength_nm, frequency_Hz, stopping_potential_V).
    
    Returns
    -------
    pd.DataFrame
        Table with columns: 'Wavelength (nm)', 'Frequency (Hz)', 'Stopping Potential (eV)'
        sorted by frequency.
    """
    filenames = os.listdir(folder) 
    csv_files = [file for file in filenames if file.endswith('.csv')]
    
    wavelengths = []
    frequencies = []
    cutoff_voltages = []
    
    for filename in csv_files:
        filepath = os.path.join(folder, filename)
        wavelength, frequency, cutoff_voltage = process_spectrum(filepath, plot=False)
        wavelengths.append(wavelength)
        frequencies.append(frequency)
        cutoff_voltages.append(cutoff_voltage)

    table = pd.DataFrame({
        'Wavelength (nm)': wavelengths,
        'Frequency (Hz)': frequencies,
        'Stopping Potential (eV)': cutoff_voltages 
    })
    
    # Return the sorted table (sort_values returns a new DataFrame)
    return table.sort_values(by='Frequency (Hz)').reset_index(drop=True)
