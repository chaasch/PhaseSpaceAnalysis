#####################################################################################
#   Project: IAEA Phase Space Analysis
#   Script: AnalyzePhaseSpaceExample
#   Description: Loads an IAEA phase space with PhaseSpaceParser and analyzes it with
#   ParticleAnalyzer: spectrum, energy fluence, efficiency and a comparison of the
#   photon spectrum against SpekPy. Graphs and csv files are written to
#   OUTPUT_DIRECTORY.
#   Author: Campbell Haasch
#   Institution: University of Wisconsin-Madison; Department of Medical Physics
#   Lab: RED Lab/UWMRRC
#   Last Updated: 2026-10-26
#.  *Assisted by Claude Opus 5.5
#####################################################################################

import os
from PhaseSpaceParser import PhaseSpaceParser
from ParticleAnalyzer import ParticleAnalyzer

#Binning and source energy settings for histograms
SPECTRUM_BIN_WIDTH = 0.125      #keV
SPATIAL_BIN_WIDTH = 0.1        #cm
SOURCE_ENERGY = 120 #keV

#tube settings for the SpekPy comparison
VOLTAGE = 120                   #kV
ANODE_ANGLE = 20                #degrees
FILTRATION = [('Be', 3.0)]      #(material, thickness in mm)
SPEKPY_BIN_WIDTH = 0.125          #keV
HVL_MATERIALS = ['Al', 'Cu']

#phase space to analyze: PHASE_SPACE_DIRECTORY/PHASE_SPACE_NAME.IAEAheader and .IAEAphsp
PHASE_SPACE_NAME = 'EGSNRC-120kV'
PHASE_SPACE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIRECTORY = os.path.join(PHASE_SPACE_DIRECTORY, 'Outputs', f'{PHASE_SPACE_NAME}_Output')

#choose which options to output
OUTPUT_GRAPH = True
OUTPUT_CSV = True

def main():
    # --------------------------Load phase space into the particle analyzer---------------------------------
    parser = PhaseSpaceParser('IAEA')
    analyzer = ParticleAnalyzer(output_directory=OUTPUT_DIRECTORY, phase_space_name=PHASE_SPACE_NAME)
    parser.read_phase_space(os.path.join(PHASE_SPACE_DIRECTORY, PHASE_SPACE_NAME + '.IAEAheader'),
                            os.path.join(PHASE_SPACE_DIRECTORY, PHASE_SPACE_NAME + '.IAEAphsp'), analyzer)

    # --------------------------Spectrum and energy fluence-------------------------------------------------
    analyzer.analyze_spectra(SPECTRUM_BIN_WIDTH, output_graph=OUTPUT_GRAPH, output_csv=OUTPUT_CSV)
    analyzer.analyze_fluence(SPATIAL_BIN_WIDTH, output_graph=OUTPUT_GRAPH, output_csv=OUTPUT_CSV, energy_weighted=True)

    # --------------------------Efficiency------------------------------------------------------------------
    efficiency = analyzer.analyze_efficiency(source_energy=SOURCE_ENERGY, output_csv=OUTPUT_CSV)

    # --------------------------SpekPy comparison-----------------------------------------------------------
    comparison = analyzer.compare_spekpy(VOLTAGE, ANODE_ANGLE, FILTRATION, SPEKPY_BIN_WIDTH, HVL_MATERIALS,
                                         output_graph=OUTPUT_GRAPH, output_csv=OUTPUT_CSV)

if __name__ == '__main__':
    main()
