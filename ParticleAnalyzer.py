#####################################################################################
#   Project: IAEA Phase Space Analysis
#   Class: Particle, ParticleAnalyzer
#   Description: Particle describes a single particle stored in a phase space file and
#   holds the particle's position, direction, energy and weight. ParticleAnalyzer holds
#   the particles of a phase space and provides the analysis methods for them.
#   Author: Campbell Haasch
#   Institution: University of Wisconsin-Madison; Department of Medical Physics
#   Lab: RED Lab/UWMRRC
#   Last Updated: 2026-10-26
#.  *Assisted by Claude Opus 5.5
#####################################################################################

import os
import tempfile
import warnings
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
plt.style.use(os.path.join(os.path.dirname(os.path.abspath(__file__)), "PhaseSpaceStyle.mplstyle"))


#IAEA particle type codes
PARTICLE_NAMES = {1: 'Photon', 2: 'Electron', 3: 'Positron', 4: 'Neutron', 5: 'Proton'}
#smallest |cos(theta)| used in the 1/|w| fluence correction so grazing particles don't blow up a bin
MIN_COS_THETA = 0.08

@dataclass
class Particle():
    """
    A single particle from a phase space file. (u, v, w) are the direction cosines along x, y and z,
    and new_history marks the first particle of a new history.
    """
    x: float                        #cm
    y: float                        #cm
    z: float                        #cm
    u: float
    v: float
    weight: float
    w: Optional[float] = None
    energy: Optional[float] = None  #keV
    particle_type: Optional[int] = None
    new_history: bool = False
    extra_floats: list[float] = field(default_factory=list)
    extra_longs: list[int] = field(default_factory=list)

class ParticleAnalyzer():
    """
    Holds the particles of a phase space and analyzes them. Graphs and csv files are written to
    output_directory with file names starting with phase_space_name.

    Functions:
     - add_particle(particle: Particle):
       Appends a particle to self.particle_list.
     - analyze_spectra(bin_width: float, normalize: bool, output_graph: bool, output_csv: bool):
       Weighted energy spectrum (keV) for each particle type.
     - analyze_fluence(spatial_bin_width: float, normalized: bool, output_graph: bool, output_csv: bool, energy_weighted: bool):
       Fluence (or energy fluence) map in the scoring plane for each particle type.
     - analyze_efficiency(source_energy: float, output_csv: bool):
       Particles and energy per original history and particle weight statistics for each particle type.
     - compare_spekpy(voltage: float, anode_angle: float, filtration: list, bin_width: float, hvl_materials: list, output_graph: bool, output_csv: bool):
       Compares the photon spectrum, mean energy and HVL1, HVL2, homogeneity coefficient and effective energy in each material against SpekPy.
    """

    def __init__(self, output_directory:str = None, phase_space_name:str = None, particle_list:list[Particle] = None,orig_histories:int = None):
        self.output_directory = output_directory if output_directory != None else '.'
        self.phase_space_name = phase_space_name if phase_space_name != None else 'phase_space'


        if particle_list != None:
            self.particle_list = particle_list
        else:
            self.particle_list = []

        self.orig_histories = orig_histories
        os.makedirs(self.output_directory, exist_ok=True)

    """Append a particle to the particle list."""
    def add_particle(self, particle:Particle):
        self.particle_list.append(particle)

    """Weighted energy spectrum for each particle type with bin_width in keV. If normalized it is per keV
       per unit total weight of that particle type so each spectrum integrates to 1"""
    def analyze_spectra(self, bin_width:float, normalize:bool = True, output_graph:bool = True, output_csv:bool = True):
        energy, weight, particle_type = np.array([(p.energy, p.weight, p.particle_type) for p in self.particle_list]).T
        edges = np.arange(np.floor(energy.min() / bin_width), np.floor(energy.max() / bin_width) + 2) * bin_width
        spectra = {}

        #iterate through each particle type in the phase space
        for type_code in np.unique(particle_type).astype(int):
            selected = particle_type == type_code
            counts, _ = np.histogram(energy[selected], bins=edges, weights=weight[selected])
            if normalize:
                counts = counts / (weight[selected].sum() * bin_width)
            spectra[PARTICLE_NAMES.get(type_code, f'Type_{type_code}')] = counts

        if output_graph:
            for name, counts in spectra.items():
                fig, ax = plt.subplots()
                ax.stairs(counts, edges)
                ax.set_xlabel('Energy [keV]')
                ax.set_ylabel('Particles per keV ' if normalize else 'Particles')
                ax.set_title(f'{self.phase_space_name} {name} Spectrum')
                fig.savefig(os.path.join(self.output_directory, f'{self.phase_space_name}_{name}_spectrum.png'), dpi=300)
                plt.close(fig)
        if output_csv:
            #one row per bin: bin center, then a value for each particle type
            np.savetxt(os.path.join(self.output_directory, f'{self.phase_space_name}_spectrum.csv'),
                       np.column_stack([(edges[:-1] + edges[1:]) / 2, *spectra.values()]), fmt='%.10g', delimiter=',',
                       header=','.join(['energy_keV', *spectra.keys()]), comments='')
        return edges, spectra


    """Fluence in the scoring plane binned in x and y for each particle type. Each particle crossing the plane counts
       weight / |cos(theta)| (times its energy for energy fluence) per unit area, normalized to a maximum of 1 if requested"""
    def analyze_fluence(self, spatial_bin_width: float, normalized:bool = True, output_graph: bool = True, output_csv:bool = True, energy_weighted:bool = False):
        x, y, w, weight, energy, particle_type = np.array([(p.x, p.y, p.w, p.weight, p.energy, p.particle_type) for p in self.particle_list]).T
        contribution = weight / np.maximum(np.abs(w), MIN_COS_THETA)
        if energy_weighted:
            contribution = contribution * energy
        x_edges = np.arange(np.floor(x.min() / spatial_bin_width), np.floor(x.max() / spatial_bin_width) + 2) * spatial_bin_width
        y_edges = np.arange(np.floor(y.min() / spatial_bin_width), np.floor(y.max() / spatial_bin_width) + 2) * spatial_bin_width
        fluences = {}
        for type_code in np.unique(particle_type).astype(int):
            selected = particle_type == type_code
            fluence, _, _ = np.histogram2d(x[selected], y[selected], bins=[x_edges, y_edges], weights=contribution[selected])
            fluence = fluence / spatial_bin_width**2
            if normalized:
                fluence = fluence / fluence.max()
            fluences[PARTICLE_NAMES.get(type_code, f'Type_{type_code}')] = fluence

        name = 'energy_fluence' if energy_weighted else 'fluence'
        units = 'keV cm$^{-2}$' if energy_weighted else 'cm$^{-2}$'
        if output_graph:
            for particle_name, fluence in fluences.items():
                fig, ax = plt.subplots()
                mesh = ax.pcolormesh(x_edges, y_edges, fluence.T)
                fig.colorbar(mesh, ax=ax, label=f"{name.replace('_', ' ').capitalize()} ({'normalized' if normalized else units})")
                ax.set_xlabel('x [cm]')
                ax.set_ylabel('y [cm]')
                ax.set_aspect('equal')
                ax.set_title(f"{self.phase_space_name} {particle_name} {name.replace('_', ' ')}")
                fig.savefig(os.path.join(self.output_directory, f'{self.phase_space_name}_{particle_name}_{name}.png'), dpi=150)
                plt.close(fig)
        if output_csv:
            #one row per bin: bin center x, bin center y, then a value for each particle type
            x_centers, y_centers = np.meshgrid((x_edges[:-1] + x_edges[1:]) / 2, (y_edges[:-1] + y_edges[1:]) / 2, indexing='ij')
            np.savetxt(os.path.join(self.output_directory, f'{self.phase_space_name}_{name}.csv'),
                       np.column_stack([x_centers.ravel(), y_centers.ravel(), *(fluence.ravel() for fluence in fluences.values())]),
                       fmt='%.10g', delimiter=',', header=','.join(['x_cm', 'y_cm', *fluences.keys()]), comments='')
        return x_edges, y_edges, fluences

    """Weighted particles and energy reaching the scoring plane per original history for each particle type,
       and if the source particle energy (keV) is given, the fraction of the source energy that reaches the plane.
       Also gives the original histories and the min, max and mean particle weight of each type"""
    def analyze_efficiency(self, source_energy:float = None, output_csv:bool = True):
        if self.orig_histories == None:
            raise ValueError("orig_histories must be set to calculate efficiency")
        energy, weight, particle_type = np.array([(p.energy, p.weight, p.particle_type) for p in self.particle_list]).T
        efficiency = {}
        for type_code in np.unique(particle_type).astype(int):
            selected = particle_type == type_code
            entry = {
                'orig_histories': self.orig_histories,
                'particles_per_history': float(weight[selected].sum()) / self.orig_histories,
                'energy_per_history_keV': float((weight[selected] * energy[selected]).sum()) / self.orig_histories,
            }
            if source_energy != None:
                entry['energy_efficiency'] = entry['energy_per_history_keV'] / source_energy
            entry['min_weight'] = float(weight[selected].min())
            entry['max_weight'] = float(weight[selected].max())
            entry['mean_weight'] = float(weight[selected].mean())
            efficiency[PARTICLE_NAMES.get(type_code, f'Type_{type_code}')] = entry

        if output_csv:
            #one row per particle type
            columns = list(next(iter(efficiency.values())).keys())
            with open(os.path.join(self.output_directory, f'{self.phase_space_name}_efficiency.csv'), 'w') as file:
                file.write(','.join(['particle', *columns]) + '\n')
                for name, entry in efficiency.items():
                    file.write(','.join([name, *(f'{entry[column]:.10g}' for column in columns)]) + '\n')
        return efficiency

    """Compare the photon spectrum against SpekPy for a tungsten tube at voltage (kV) and anode_angle (degrees) with
       filtration given as [(material, thickness in mm)]. Both spectra are normalized to unit area on bin_width (keV) bins
       and graphed with their ratio. Mean energy and, in each of hvl_materials, the first and second HVL,
       homogeneity coefficient and effective energy are calculated with SpekPy for both so they are comparable"""
    def compare_spekpy(self, voltage:float, anode_angle:float, filtration:list[tuple[str, float]], bin_width:float = 0.5,
                       hvl_materials:list[str] = ('Al', 'Cu'), output_graph:bool = True, output_csv:bool = True):
        #spekpy is only needed for this comparison, so it is imported here rather than for the whole module
        import spekpy

        edges, spectra = self.analyze_spectra(bin_width, output_graph=False, output_csv=False)
        energy = (edges[:-1] + edges[1:]) / 2
        phase_space_spectrum = spectra['Photon'] / spectra['Photon'].sum()

        #SpekPy warns about divide by zero in its internal matrix math, which is harmless
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            spek = spekpy.Spek(kvp=voltage, th=anode_angle, dk=bin_width)
            spek.multi_filter(filtration)
            spek_energy, spek_fluence = spek.get_spectrum()
            spek_spectrum = np.interp(energy, spek_energy, spek_fluence, left=0, right=0)
            spek_spectrum = spek_spectrum / spek_spectrum.sum()

            #load the phase space spectrum into SpekPy so its mean energy and HVL use the same attenuation data
            with tempfile.NamedTemporaryFile('w', suffix='.csv', delete=False) as file:
                np.savetxt(file, np.column_stack([energy, phase_space_spectrum]), delimiter=',')
            phase_space_spek = spekpy.Spek.load_from_file(file.name, ',')
            os.remove(file.name)

            #mean energy in keV, then for each material the first and second HVL in mm, the homogeneity
            #coefficient (HVL1 / HVL2) and the effective energy in keV (monoenergetic energy with the same HVL1)
            comparison = {}
            for source, source_spek in [('phase_space', phase_space_spek), ('spekpy', spek)]:
                comparison[source] = {'mean_energy_keV': source_spek.get_emean()}
                for material in hvl_materials:
                    comparison[source][f'hvl1_{material}_mm'] = source_spek.get_hvl1(matl=material)
                    comparison[source][f'hvl2_{material}_mm'] = source_spek.get_hvl2(matl=material)
                    comparison[source][f'hc_{material}'] = source_spek.get_hc(matl=material)
                    comparison[source][f'eeff_{material}_keV'] = source_spek.get_eeff(matl=material)

        if output_graph:
            filter_description = ', '.join(f'{thickness} mm {material}' for material, thickness in filtration) or 'no filtration'
            fig, (spectrum_ax, ratio_ax) = plt.subplots(2, 1, figsize=(9, 7), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
            spectrum_ax.stairs(phase_space_spectrum, edges, label=self.phase_space_name, color='black')
            spectrum_ax.stairs(spek_spectrum, edges, label=f'SpekPy {voltage} kV, {anode_angle}° W, {filter_description}', color='tab:orange')
            spectrum_ax.set_ylabel(f'Fraction of photons per {bin_width} keV')
            spectrum_ax.set_title(f'{self.phase_space_name} Photon Spectrum vs SpekPy')
            spectrum_ax.legend()
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = np.where(phase_space_spectrum > 0, spek_spectrum / phase_space_spectrum, np.nan)
            ratio_ax.stairs(ratio, edges, color='tab:orange')
            ratio_ax.axhline(1, color='black', linewidth=0.8)
            ratio_ax.set_ylim(0, 2)
            ratio_ax.set_ylabel('SpekPy / phase space')
            ratio_ax.set_xlabel('Energy (keV)')
            fig.tight_layout()
            fig.savefig(os.path.join(self.output_directory, f'{self.phase_space_name}_spekpy_comparison.png'), dpi=300)
            plt.close(fig)
        if output_csv:
            #one row per shared bin: bin center, then the normalized phase space and SpekPy values
            np.savetxt(os.path.join(self.output_directory, f'{self.phase_space_name}_spekpy_comparison.csv'),
                       np.column_stack([energy, phase_space_spectrum, spek_spectrum]), fmt='%.10g', delimiter=',',
                       header='energy_keV,phase_space,spekpy', comments='')
            #one row per quantity: phase space value, SpekPy value and the percent difference of SpekPy from the phase space
            with open(os.path.join(self.output_directory, f'{self.phase_space_name}_spekpy_hvl_comparison.csv'), 'w') as file:
                file.write('quantity,phase_space,spekpy,percent_difference\n')
                for quantity, phase_space_value in comparison['phase_space'].items():
                    spekpy_value = comparison['spekpy'][quantity]
                    percent_difference = 100 * (spekpy_value - phase_space_value) / phase_space_value
                    file.write(f'{quantity},{phase_space_value:.10g},{spekpy_value:.10g},{percent_difference:.10g}\n')
        return comparison

