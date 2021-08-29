#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 18:04:10 2020

@author: rbenda

ECRIRE multipoles dans un fichier output (format .punch / Stone's DMA ?)
"""


import numpy as np
#import matplotlib
import math
import time

import scipy.spatial.distance

from extraction_QM_info import computes_nb_primitive_GTOs  
from extraction_QM_info import import_correspondence_pGTOs_atom_index
from extraction_QM_info import import_basis_set_exponents_primitive_GTOs
from extraction_QM_info import import_position_nuclei_associated_basis_function_pGTO


from auxiliary_functions import number_final_expansion_centers
from auxiliary_functions import coordinates_final_expansion_centers
from auxiliary_functions import list_bond_centers
from auxiliary_functions import boolean_final_expansion_site_is_atom
from auxiliary_functions import conversion_quadrupole_tensor_cartesian_to_spherical
from auxiliary_functions import conversion_quadrupole_tensor_spherical_to_cartesian
from auxiliary_functions import writes_DMA_output_values

from multipoles_DMA import compute_DMA_multipole_moment_final_expansion_center_all_sites_all_orders
from multipoles_DMA import computes_total_quadrupole_from_local_multipoles_DMA_sites
from multipoles_DMA import computes_total_dipole_from_local_multipoles_DMA_sites
from multipoles_DMA import compute_DMA_multipole_moment_final_expansion_center_all_sites_all_orders_TEST

from ESP_potential import compute_electrostatic_potential_generated_DMA_multipoles
from ESP_potential import compute_electrostatic_potential_generated_multipoles_overlap_centers
from ESP_potential import reads_grid_file
from ESP_potential import writes_ESP_values_DMA_model

#from cclib.io import ccread
from cclib.io import ccopen


#########################################################################################################
#Launching the DMA computations : setting the main user-defined options
#(redistribution strategy, which sites/points are kep as multipolar final expansion centers, etc.)

#Choice of the input file (output of a QM calculation)

#Directory of QM output files :
QM_outputs_files_dir='/home/rbenda/Desktop/THESE/CODE_MULTIPOLES/FINAL_CODE/QM_output_files/'

Grid_points_files_dir='/home/rbenda/Desktop/THESE/CODE_MULTIPOLES/FINAL_CODE/Grid_points_files/'

QM_ESP_files_dir='/home/rbenda/Desktop/THESE/CODE_MULTIPOLES/FINAL_CODE/QM_ESP_files_dir/'

ESP_files_dir='/home/rbenda/Desktop/THESE/CODE_MULTIPOLES/FINAL_CODE/ESP_files_dir/'

DMA_output_files_dir='/home/rbenda/Desktop/THESE/CODE_MULTIPOLES/FINAL_CODE/DMA_output_files_dir/'

#####################
#QM calculations with Pople's type basis set (e.g. 6-31G(d)):
###########
#Gaussian calculations : (g09)

##Open-shell calculations (UHF or ROHF, or U-DFT or RO-DFT):
#filename = "CU2_ion_alone_ROHF_6-31Gd_no_PCM_pop_full_iop_3_33.log"
#filename = "CU2_ion_alone_PBE0_6-31Gd_no_PCM_pop_full.log"
#filename = "NI2_ion_alone_ROHF_6-31Gd_no_PCM.log"

##Closed-shell calculations :
#filename = "QL_Bipyr_alone_SP_PBE0_6-31Gd_no_PCM.log"
#filename = "Pyridine_alone_Opt_HF_6-31Gd_no_PCM_pop_full_iop_3_33.log"
#filename = "NA1_ion_alone_PBE0_6-31Gd_no_PCM.log"
#filename = "AL3_ion_alone_PBE0_6-31Gd_no_PCM.log"
#filename = "CLO_ion_alone_PBE0_6-31Gd_no_PCM.log"
#filename = "HCO3-_ion_alone_Opt_PBE0_6-31Gd_no_PCM.log"
#filename = "HO-_ion_alone_Opt_PBE0_6-31Gd_no_PCM.log"
#filename = "H3O+_ion_alone_Opt_PBE0_6-31Gd_no_PCM.log"
#filename = "PO4_3-_ion_alone_Opt_PBE0_6-31Gd_no_PCM.log"
#filename = "HClO_alone_Opt_PBE0_6-31Gd_no_PCM.log"
#filename = "NO3-_ion_alone_Opt_PBE0_6-31Gd_no_PCM.log"
#filename = "Benzene_Opt_PBE0_6-31Gd_no_PCM.log"
filename = "H2O_Opt_PBE0_6-31Gd_no_PCM.log"
#filename = "NH3_Opt_PBE0_6-31Gd_no_PCM_symm_C3v.log"
#filename = "CH4_Opt_PBE0_6-31Gd_no_PCM_symm_Td.log"

###Closed-shell and fully symmetric calculations
####Test on local quadrupole symmetry 
#filename = "O2_Opt_PBE0_6-31Gd_x_axis.log"
#filename = "O2_Opt_PBE0_6-31Gd_y_axis.log"
#filename = "O2_Opt_PBE0_6-31Gd_z_axis.log"
#filename = "O2_Opt_PBE0_6-31Gd_z_axis_NoSymm.log"

###########
#GAMESS calculations :
##Open-shell calculations (UHF or ROHF, or U-DFT or RO-DFT):
#filename = "Fe3+_ROHF_6-31Gd.out"
#filename = "Fe3+_ROHF_6-31Gd_ISPHR=1.out"
#filename = "Fe3+_UHF_6-31Gd.out"

#filename = "Pyridine_CU2_ion_pos1_Opt_ROHF_6-31G.out"
#filename = "Pyridine_FE2_ion_pos1_ROHF_6-31G_Canonical2_ROHF_1_step_print_guess.out"
#filename = "Porphyrin_FE2_ion_pos1_ROHF_6-31G_Canonical2_ROHF_1_step_print_guess.out"

#filename = "Porphyrin_model_FE2_ion_SP_ROHF_6-31G_Canonical2_DIIS.out"
#filename = "Porphyrin_FE2_ion_2D_SP_ROHF_6-31G_Euler_DIIS_1_step_print_guess.out"
#filename = "Roussin_red_dianion_Opt_ROHF_6-31G_DIIS_Roothan_SUITE.out"


##Closed-shell calculations :
#filename ="Benzene_HF_6-31G.out"
#With complete symmetry imposed
#filename ="Benzene_HF_6-31G_D6h.out"
#filename ="CH4_HF_6-31Gd_Td_GAMESS.out"
#filename ="NH3_HF_6-31Gd_C3v_GAMESS.out"
#filename ="H2O_HF_6-31Gd_C2v.out"

##########
#Psi 4 calculations :

#####################

#####################
#DFT calculations with Dunning's type basis set (cc-pVXZ, X=S,D,T,Q, etc.)
##############
#Gaussian calculations (g09) :

#filename = "H2O_PBE0_aug-cc-pVTZ_no_PCM.log"

#filename = "H2O_SP_PBE0_aug-cc-pVDZ_no_PCM.log"
#filename = "H2O_SP_PBE0_aug-cc-pVTZ_no_PCM.log"
#filename = "H2O_SP_PBE0_aug-cc-pVQZ_no_PCM.log"
#filename = "H2O_SP_PBE0_aug_cc-pV5Z_no_PCM.log"
#filename = "H2O_SP_PBE0_aug_cc-pV6Z_no_PCM.log"

#filename = "H2O_SP_PBE0_cc-pVDZ_no_PCM.log"
#filename = "H2O_SP_PBE0_cc-pVTZ_no_PCM.log"
#filename = "H2O_SP_PBE0_cc-pVQZ_no_PCM.log"
#filename = "H2O_SP_PBE0_cc-pV5Z_no_PCM.log"
#filename = "H2O_SP_PBE0_cc-pV6Z_no_PCM.log"

#Psi4 :
#filename = "output_O2_Opt_PBE0_cc-pVDZ_psi4.dat"
 

#####################
#Wave-function theory calculations :
    
#Gaussian
#filename = "Pyridine_CU2_ion_pos1_Opt_ROHF_6-31Gd_SP_CCSD_6-31G.log"

#filename = "H2O_SP_HF_aug-cc-pVDZ_no_PCM.log"
#filename = "H2O_SP_HF_aug-cc-pVTZ_no_PCM.log"
#filename = "H2O_SP_HF_aug-cc-pVQZ_no_PCM.log"
#filename = "H2O_SP_HF_aug-cc-pV5Z_no_PCM.log"
#filename = "H2O_SP_HF_aug-cc-pV6Z_no_PCM.log"

#filename = "H2O_SP_HF_cc-pVDZ_no_PCM.log"
#filename = "H2O_SP_HF_cc-pVTZ_no_PCM.log"
#filename = "H2O_SP_HF_cc-pVQZ_no_PCM.log"
#filename = "H2O_SP_HF_cc-pV5Z_no_PCM.log"
#filename = "H2O_SP_HF_cc-pV6Z_no_PCM.log"

#filename = "H2O_SP_CCSD_cc-pVDZ_no_PCM.log"
#filename = "H2O_SP_CCSD_cc-pVTZ_no_PCM.log"
#filename = "H2O_SP_CCSD_cc-pVQZ_no_PCM.log"

#filename = "H2O_SP_CCSD_aug-cc-pVDZ_no_PCM.log"
#filename = "H2O_SP_CCSD_aug-cc-pVTZ_no_PCM.log"
#filename = "H2O_SP_CCSD_aug-cc-pVQZ_no_PCM.log"

#filename = "H2O_SP_MP2_cc-pVDZ_no_PCM.log"
#filename = "H2O_SP_MP2_cc-pVTZ_no_PCM.log"
#filename = "H2O_SP_MP2_cc-pVQZ_no_PCM.log"

#filename = "H2O_SP_MP2_aug-cc-pVDZ_no_PCM.log"
#filename = "H2O_SP_MP2_aug-cc-pVTZ_no_PCM.log"
#filename = "H2O_SP_MP2_aug-cc-pVQZ_no_PCM.log"

#Psi4 :
#filename = "output_H2O_SP_HF_cc-pVDZ.dat"


############
#Problem for Psi4 calculations with Pople's basis sets :
#Tr(S*D) is not equal to the total number of electrons
    
#filename = "output_O2_PBE0_6-31Gd_y_axis.dat"

#filename = "output_O2_PBE0_6-31Gd_z_axis.dat"

#filename = "output_NH3_PBE0_6-31Gd_psi4.dat"

######################
#To deal with the specificities of the basis treatment depending on the
#different quantum chemistry codes, we precise the code which
#has generated the input for the DMA analysis (output of a quantum code) 
#Example : 'SP' shells (in 6-31G(d) Pople's type basis sets) : count only for 
#one contracted shell in Gaussian / GAMESS
#while they count for 2 contracted shells (one S, one P contracted shell) in Psi4

input_type='Gaussian'
#input_type='GAMESS'
#input_type='Psi4'

###############################"
#Constants
#1 Bohr in Angstroms
conversion_bohr_angstrom=0.529177209
#1 Debye in e.Ang :
conversion_Debye_in_e_Ang=0.20819434
##=> 1 Debye = 0.20819434/0.529177209 e.Bohr (a.u.) = 0.3934302847120538 e.Bohr
conversion_Debye_atomic_units=conversion_Debye_in_e_Ang/conversion_bohr_angstrom

#Conversion of Debye.Ang (quadrupole units) in e.Bohr**2 (atomic units)
#=> 1 Debye.Ang (=1 Buckingham) = 0.20819434/(0.529177209)**2 e.Bohr**2 (a.u.) = 0.7434754899129714 a.u.
#Debye.Ang units for quadrupole moments are used e.g. in Gaussian outputs
conversion_Debye_Ang_atomic_units=conversion_Debye_atomic_units/conversion_bohr_angstrom
###############################

##############################################
parser = ccopen(QM_outputs_files_dir+filename)

QM_data = parser.parse()


##################################
#Indicates whether with the given basis set, and with the QM code used for the calculation,
#d-shells are of Cartesian or of Spherical form :
boolean_cartesian_D_shells=True

#Examples : 
# *** In Gaussian program : 
#    - for 6-31G(d) Pople's type basis set 
#           => boolean_cartesian_D_shells = True
#    - for (aug)-cc-pVTZ Dunning's type basis set 
#           => boolean_cartesian_D_shells = False

# *** In Psi4 program : 
#Indicated by 'Spherical Harmonics' variable in Psi4 output.dat file 
#cf . "Spherical Harmonics    = FALSE" (e.g. for 6-31G(d) Pople's style basis set)
#cf.  "Spherical Harmonics    = TRUE" (e.g. for cc-pVDZ Dunning's type basis set)

# *** In GAMESS program : 
##Indicated by 'ispher' variable (in the $contrl section) in GAMESS
##(by default : ispher = -1 : cartesian basis functions / 
#ispher=1 : spherical basis functions -- 'pure' shells for p, d, f, etc. -type shells)
##################################

##################################
#Indicates whether with the given basis set, and with the QM code used for the calculation,
#f-shells are of Cartesian or of Spherical form :
  
# *** In Psi4 program : 
#- 6-31G(d) or Pople's type basis sets 
#=> CARTESIAN f-basis functions are used
#==> boolean_cartesian_F_shells=False
#(i.e. 10 basis functions of f-type, for each F contracted shell)

# *** In Gaussian program : 
#6-31G(d), Pople's type basis set or (aug)-cc-pVXZ Dunning's type basis sets
# => 7 basis functions of f-type (spherical form)
#=> boolean_cartesian_F_shells=True
boolean_cartesian_F_shells=False
##################################

##################################
#*** Psi4 exceptions for indexation orders of p-type GTOs or cartesian d-type GTOs :
#("exceptions" relatively to Gaussan or GAMESS primitive GTOs indexation orders)

#psi4_exception_P_GTOs_indexation_order=True if indexation order : p0, p+1, p-1 
#instead of px, py, pz
#psi4_exception_P_GTOs_indexation_order=False : usual order PX, PY, PZ (i.e. p+1, p-1, p0)
psi4_exception_P_GTOs_indexation_order=False

#psi4_exception_D_GTOs_indexation_order=True
# if indexation order : dxx, dxy, dxz, dyy, dyz, dzz
#instead of dxx, dyy, dzz, dxy, dxz, dyz (case of Gaussian / GAMESS)
#psi4_exception_D_GTOs_indexation_order=False : usual order for cartesian D shells 
#(in the case boolean_cartesian_D_shells=True) : XX, YY, ZZ, XY, XZ, YZ
#or usual order for spherical d GTOs (d0, d+1, d-1, d+2, d-2), in the case 
#Q-CHEM : other indexation order dxx, dxy, dyy, dxz, dyz, dzz !
#boolean_cartesian_D_shells=False
psi4_exception_D_GTOs_indexation_order=False
##################################

########################################################################
########################################################################
#GENERAL options (not related to the particular molecule example)

#Coordinates read from the last optimization step of the QM output file :
#Evaluation of 'atomcoords' attribute at the last index 
#(case of a geometry optimization => last [optimal] geometry)
atomic_coordinates=QM_data.atomcoords[len(QM_data.atomcoords)-1]

                                     
nb_atoms=len(atomic_coordinates)

#Whether all atomic sites (nuclei) are kept as final expansion centers :
#(True also if there are additional final DMA sites, on top of nuclei)
boolean_all_atomic_sites=True

#In the case that 'boolean_all_atomic_sites'==False : list of nuclei (indexes)
#that we want to omit (will not be final DMA expansion sutes)
atomic_sites_to_omit=[]

###############
#AMOEBA philosophy : only nuclei as final sites :
index_philosophy=1

###############
#Nuclei positions AND bond centers are retained as final sites
#index_philosophy=2

###############
#Nuclei positions AND user-defined (possibly random, possibly also bond centers) additional points
#are retained as final sites
#index_philosophy=3

###############
#User-defined (possibly random, possibly also bond centers) are retained as final sites
#(possibly not including any nuclei)
##In the case of omitting nuclei ==> add to the final total monopole the sum 
##of (positive) charge (Z index) of omitted nuclei
#index_philosophy=4

###############
#All overlap centers kept as final DMA expansion sites
#index_philosophy=5

###############
#Stone's strategy : redistribution of multipole moments to nearest neighbor
#(i.e. nearest final expansion site) only
redistribution_strategy=1

###############
#Vigne-Maeder's strategy : (Fabienne Vigné-Maeder, and Pierre Claverie, J. Chem. Phys. 88, 4934 (1988))
#redistribution of multipole moments to all 
#final expansion sites but at the prorata of the inverse distances
#of the overlap center to the final sites
#redistribution_strategy=2

#############################################################
#EXTRA final expansion DMA sites (additionally e.g. to the nuclei)

if (index_philosophy==1):
    #########
    #Case of no user extra site for final multipole expansion
    user_extra_final_sites=[]

elif (index_philosophy==2):
    #########
    #Case of all bond centers added to the nuclei as final expansion sites
    #=> goes along with 'index_philosophy=2' option 
    user_extra_final_sites=list_bond_centers(atomic_coordinates,QM_outputs_files_dir+filename)

#########
#Cas of user-defined (possibly random) additional sites :  
#(keep line 93 and uncomment below)
#user_extra_final_sites.append([0,0,0])
#user_extra_final_sites.append([0,0,0.5])
#etc... 

nb_primitive_GTOs=computes_nb_primitive_GTOs(QM_data,input_type,boolean_cartesian_D_shells,boolean_cartesian_F_shells)
total_nb_primitive_GTOs=nb_primitive_GTOs[1]
#print('nb_primitive_GTOs')
#print(nb_primitive_GTOs)

#Index of the atom to which each primitive GTO is associated  :
correspondence_basis_pGTOs_atom_index=import_correspondence_pGTOs_atom_index(nb_primitive_GTOs,nb_atoms)

##nb_primitive_GTOs=len(correspondence_basis_pGTOs_atom_index)

basis_set_exponents_primitive_GTOs=import_basis_set_exponents_primitive_GTOs(QM_data,boolean_cartesian_D_shells,boolean_cartesian_F_shells)

position_nuclei_associated_basis_function_pGTO=import_position_nuclei_associated_basis_function_pGTO(atomic_coordinates,nb_primitive_GTOs,nb_atoms,QM_data)

positions_final_expansion_sites=coordinates_final_expansion_centers(atomic_coordinates,correspondence_basis_pGTOs_atom_index,boolean_all_atomic_sites,atomic_sites_to_omit,user_extra_final_sites,index_philosophy,basis_set_exponents_primitive_GTOs,position_nuclei_associated_basis_function_pGTO,total_nb_primitive_GTOs)

#Number of final expansion centers :
##M=number_final_expansion_centers(atomic_coordinates,boolean_all_atomic_sites,atomic_sites_to_omit,user_extra_final_sites,index_philosophy,nb_primitive_GTOs,correspondence_basis_pGTOs_atom_index)
M=len(positions_final_expansion_sites)

#print('POSITIONS OF THE FINAL DMA EXPANSION SITES :')
#print(positions_final_expansion_sites)
#print(' ')

#Orders up to which multipole moments are computed for each site

#For instance up to order 2 (quadrupole) for all sites (cf. AMOEBA):
l_max_user=[4 for i in range(M)]

#However, the user can specify its own strategy (e.g. computing multipole moments
#on H atoms up to order one only), e.g. for ClO- (with Cl and O as final DMA sites) :
##l_max_user=[2,3]


##############################
#File in which the calculations options are written ('logfile') :
    
logfile_DMA_name='DMA_'+str(filename.split('.')[0])+'_red_strategy_'+str(redistribution_strategy)+'_philosophy_'+str(index_philosophy)+'_order_'+str(l_max_user[0])+'_info.log'

#Opening the file of specified name, in 'writing' mode
logfile_DMA  = open(DMA_output_files_dir+logfile_DMA_name,"w")

logfile_DMA.write('----------------------------------------')
logfile_DMA.write('\n')
logfile_DMA.write('DMA Analysis (logfile)')
logfile_DMA.write("\n")
logfile_DMA.write("Robert Benda, Eric Cancès 2020")
logfile_DMA.write("\n")
logfile_DMA.write('----------------------------------------')
logfile_DMA.write("\n")
logfile_DMA.write("\n")
    
filename_DMA_output_values='DMA_'+str(filename.split('.')[0])+'_red_strategy_'+str(redistribution_strategy)+'_philosophy_'+str(index_philosophy)+'_order_'+str(l_max_user[0])

#Opening the file of specified name, in 'writing' mode
file_DMA_output_values  = open(DMA_output_files_dir+filename_DMA_output_values,"w")

result_multipoles=compute_DMA_multipole_moment_final_expansion_center_all_sites_all_orders(QM_data,input_type,boolean_cartesian_D_shells,boolean_cartesian_F_shells,psi4_exception_P_GTOs_indexation_order,psi4_exception_D_GTOs_indexation_order,atomic_coordinates,boolean_all_atomic_sites,atomic_sites_to_omit,user_extra_final_sites,index_philosophy,redistribution_strategy,l_max_user,logfile_DMA,DMA_output_files_dir)

#######################################################################
#######################################################################        
 

total_dipole_moment=computes_total_dipole_from_local_multipoles_DMA_sites(0,0,0,result_multipoles,positions_final_expansion_sites)

#(not computed from all local multipole moments at all overlap centers)
#Computes the total quadrupolar tensor (of the whole molecule) with respect to the origin (0,0,0)
#Rq/Q : what if one atom is already centered at this origin point (0,0,0) ??
total_quadrupolar_tensor_cartesian=computes_total_quadrupole_from_local_multipoles_DMA_sites(0.,0.,0.,result_multipoles,positions_final_expansion_sites)

total_quadrupolar_tensor_cartesian_Debye_Ang=np.dot((conversion_bohr_angstrom**2/conversion_Debye_in_e_Ang),total_quadrupolar_tensor_cartesian)

Identity=np.diag([1,1,1],0)

trace_quadrupolar_tensor_cartesian_traceless=sum([total_quadrupolar_tensor_cartesian_Debye_Ang[i][i] for i in range(3)])

total_quadrupolar_tensor_spherical_traceless=total_quadrupolar_tensor_cartesian_Debye_Ang-np.dot((1/3.)*trace_quadrupolar_tensor_cartesian_traceless,Identity)


total_quadrupolar_tensor_spherical=conversion_quadrupole_tensor_cartesian_to_spherical(total_quadrupolar_tensor_cartesian)

######################################################################     
######################################################################
#Writing the results in an output file :
    
##Add arguments :
writes_DMA_output_values(file_DMA_output_values,result_multipoles,total_dipole_moment,total_quadrupolar_tensor_spherical_traceless,total_quadrupolar_tensor_spherical,positions_final_expansion_sites,atomic_sites_to_omit,index_philosophy,l_max_user,QM_data,atomic_coordinates,filename)
######################################################################
######################################################################
logfile_DMA.close()

file_DMA_output_values.close()


#######################################################
#######################################################

###END OF CODE LAUCHING
#########################################################################################################


##################################################################################
##################################################################################


##################################################################################
################################################################################################
#Evaluation of the electrostatic potential (ESP) generated by the multipoles moments
#at the final (reduced number) of DMA sites
    

###########################
#Reading the grid points coordinates

#filename_grid="CH4_Opt_PBE0_6-31Gd_no_PCM_symm_Td_opt_5134.grid"
filename_grid="H2O_Opt_PBE0_6-31Gd_no_PCM.grid"
#filename_grid="NH3_Opt_PBE0_6-31Gd_no_PCM_symm_C3v.grid"

grid_points=reads_grid_file(Grid_points_files_dir,filename_grid)

total_nb_grid_points=len(grid_points)

#Coordinates of each grid point :
#x_i_grid=grid_points[i][0]
#y_i_grid=grid_points[i][1]
#z_i_grid=grid_points[i][2]

#########################

#########################
#Calculation of the electrostatic potential (ESP) generated by multipole moments
#at the final DMA sites 

t_begin_ESP_DMA_final_sites=time.clock()

ESP_potential_DMA_final_sites_multipoles_grid_points=[compute_electrostatic_potential_generated_DMA_multipoles(grid_points[i][0],grid_points[i][1],grid_points[i][2],result_multipoles,positions_final_expansion_sites) for i in range(len(grid_points))]

t_end_ESP_DMA_final_sites=time.clock()

print('Time to compute the ESP potential from DMA final sites : '+str(t_end_ESP_DMA_final_sites-t_begin_ESP_DMA_final_sites))
print(' ')

#####################
#File name of the output .ESP file
if (redistribution_strategy==2):
    filename_ESP_DMA_final_sites_output="H2O_Opt_PBE0_6-31Gd_no_PCM_ESP_DMA_sites_order_"+str(l_max_user[0])+"_Vigne-Maeder_strategy.ESP"

elif (redistribution_strategy==1):
    filename_ESP_DMA_final_sites_output="H2O_Opt_PBE0_6-31Gd_no_PCM_ESP_DMA_sites_order_"+str(l_max_user[0])+"_Stone_strategy.ESP"

#Writing in the file :
    
writes_ESP_values_DMA_model(ESP_potential_DMA_final_sites_multipoles_grid_points,grid_points,total_nb_grid_points,filename_ESP_DMA_final_sites_output,ESP_files_dir)
######################

################################################################################################
##################################################################################
print('----------------------------------------------------')
print('----------------------------------------------------')



##################################################################################
################################################################################################
#Evaluation of the exact multipolar electrostatic potential 
#generated by the multipoles moments at ALL the final overlap centers P_{alpha,beta}
#=> needed for the evaluation of the EXACT multipolar expansion

print('EXACT MULTIPOLAR EXPANSION (multipole moments at ALL overlap centers')

#Redistribution strategy should not play here as no multipole moments
#are redistributed (components are simply summed up to the corresponding overlap
#center, which belongs to the set of final expansion centers)
redistribution_strategy_all_overlap_centers=1

boolean_all_atomic_sites_all_overlap_centers=True

atomic_sites_to_omit_all_overlap_centers=[]

index_philosophy_all_overlap_centers=5

#########
#User extra final sites : no need to specify in the case index_philosophy=5 (all overlap centers retained as final sites)
user_extra_final_sites_all_overlap_centers=[]

positions_final_expansion_site_all_overlap_centers=coordinates_final_expansion_centers(atomic_coordinates,correspondence_basis_pGTOs_atom_index,boolean_all_atomic_sites_all_overlap_centers,atomic_sites_to_omit_all_overlap_centers,user_extra_final_sites_all_overlap_centers,index_philosophy_all_overlap_centers,basis_set_exponents_primitive_GTOs,position_nuclei_associated_basis_function_pGTO,total_nb_primitive_GTOs)

l_max_user_all_overlap_centers=[4 for i in range(len(positions_final_expansion_site_all_overlap_centers))]

print(' ')
print(' ')
#print('POSITIONS OF THE FINAL DMA EXPANSION SITES :')
#print(positions_final_expansion_site_all_overlap_centers)
print(' ')

   
logfile_DMA_all_overlap_centers_name='DMA_'+str(filename.split('.')[0])+'_red_strategy_'+str(redistribution_strategy)+'_ALL_OVERLAP_CENTERS_order_'+str(l_max_user[0])+'_info.log'

#Opening the file of specified name, in 'writing' mode
logfile_DMA_all_overlap_centers  = open(DMA_output_files_dir+logfile_DMA_all_overlap_centers_name,"w")

logfile_DMA_all_overlap_centers.write('----------------------------------------')
logfile_DMA_all_overlap_centers.write('\n')
logfile_DMA_all_overlap_centers.write('DMA Analysis (logfile)')
logfile_DMA_all_overlap_centers.write("\n")
logfile_DMA_all_overlap_centers.write("Robert Benda, Eric Cancès 2020")
logfile_DMA_all_overlap_centers.write("\n")
logfile_DMA_all_overlap_centers.write('----------------------------------------')
logfile_DMA_all_overlap_centers.write("\n")
logfile_DMA_all_overlap_centers.write("\n")

result_multipoles_all_overlap_centers=compute_DMA_multipole_moment_final_expansion_center_all_sites_all_orders(QM_data,input_type,boolean_cartesian_D_shells,boolean_cartesian_F_shells,psi4_exception_P_GTOs_indexation_order,psi4_exception_D_GTOs_indexation_order,atomic_coordinates,boolean_all_atomic_sites_all_overlap_centers,atomic_sites_to_omit_all_overlap_centers,user_extra_final_sites_all_overlap_centers,index_philosophy_all_overlap_centers,redistribution_strategy_all_overlap_centers,l_max_user_all_overlap_centers,logfile_DMA_all_overlap_centers,DMA_output_files_dir)

filename_DMA_all_overlap_centers_output_values='DMA_'+str(filename.split('.')[0])+'_red_strategy_'+str(redistribution_strategy)+'_ALL_OVERLAP_CENTERS_order_'+str(l_max_user[0])

#Opening the file of specified name, in 'writing' mode
file_DMA_all_overlap_centers_output_values  = open(DMA_output_files_dir+filename_DMA_all_overlap_centers_output_values,"w")

writes_DMA_output_values(file_DMA_all_overlap_centers_output_values,result_multipoles_all_overlap_centers,total_dipole_moment,total_quadrupolar_tensor_spherical_traceless,total_quadrupolar_tensor_spherical,positions_final_expansion_site_all_overlap_centers,atomic_sites_to_omit,index_philosophy_all_overlap_centers,l_max_user_all_overlap_centers,QM_data,atomic_coordinates,filename)

file_DMA_all_overlap_centers_output_values.close()

print('MULTIPOLES MOMENTS with respect to overlap centers :')

for i in range(len(result_multipoles_all_overlap_centers)):
    
    x_S=positions_final_expansion_site_all_overlap_centers[i][0]
    y_S=positions_final_expansion_site_all_overlap_centers[i][1]
    z_S=positions_final_expansion_site_all_overlap_centers[i][2]
    
    test_center_S_atom=boolean_final_expansion_site_is_atom(x_S,y_S,z_S,atomic_coordinates)
    
    #Case of a redistribution center (final DMA site)
    #which is an atom
    if (test_center_S_atom[0]==True):

        ##print('SITE = ATOM  '+' n° '+str(i)+' (mass '+str(QM_data.atommasses[test_center_S_atom[1]])+'), at POSITION : ['+str(x_S)+','+str(y_S)+','+str(z_S)+']')
        print('SITE = ATOM  '+' n° '+str(i)+' (n° '+str(QM_data.atomnos[test_center_S_atom[1]])+'), at POSITION : ['+str(x_S)+','+str(y_S)+','+str(z_S)+']')
        
        print(' ')
        
        print('Monopole :     '+str(result_multipoles_all_overlap_centers[i][0]))
        print(' ')

        if (l_max_user_all_overlap_centers[i]>=1):
            print('Dipole :       '+str(result_multipoles_all_overlap_centers[i][1]))
            print('               |Q1|='+str(np.linalg.norm(result_multipoles_all_overlap_centers[i][1])))
            print(' ')
        if (l_max_user_all_overlap_centers[i]>=2):
            print('Quadrupole :   '+str(result_multipoles_all_overlap_centers[i][2]))
            print('               |Q2|='+str(np.linalg.norm(result_multipoles_all_overlap_centers[i][2])))
            ##Conversion of the quadrupolar moments from spherical to cartesian tensor :
            local_quadrupole_cartesian_form=conversion_quadrupole_tensor_spherical_to_cartesian(result_multipoles_all_overlap_centers[i][2])
            print('               Q_xx='+str(local_quadrupole_cartesian_form[0][0])+', Q_yy='+str(local_quadrupole_cartesian_form[1][1])+', Q_zz='+str(local_quadrupole_cartesian_form[2][2])+', Q_xy='+str(local_quadrupole_cartesian_form[0][1])+', Q_xz='+str(local_quadrupole_cartesian_form[0][2])+', Q_yz='+str(local_quadrupole_cartesian_form[1][2]))
            print(' ')
        if (l_max_user_all_overlap_centers[i]>=3):
            print('Octopole :     '+str(result_multipoles_all_overlap_centers[i][3]))
            print('               |Q3|='+str(np.linalg.norm(result_multipoles_all_overlap_centers[i][3])))
            print(' ')
        if (l_max_user_all_overlap_centers[i]>=4):
            print('Hexadecapole : '+str(result_multipoles_all_overlap_centers[i][4]))
            print('               |Q4|='+str(np.linalg.norm(result_multipoles_all_overlap_centers[i][4])))
            print(' ')
        print(' ')
        
    #Case of a redistribution center (final DMA site)
    #which is NOT an atom        
    else:
        print('SITE '+' n° '+str(i)+'), at POSITION : ['+str(x_S)+','+str(y_S)+','+str(z_S)+']')
        print(' ')
        
        print('Monopole :     '+str(result_multipoles_all_overlap_centers[i][0]))
        print(' ')
        if (l_max_user_all_overlap_centers[i]>=1):             
            print('Dipole :       '+str(result_multipoles_all_overlap_centers[i][1]))
            print('               |Q1|='+str(np.linalg.norm(result_multipoles_all_overlap_centers[i][1])))
            print(' ')
        if (l_max_user_all_overlap_centers[i]>=2):
            print('Quadrupole :   '+str(result_multipoles_all_overlap_centers[i][2]))
            print('               |Q2|='+str(np.linalg.norm(result_multipoles_all_overlap_centers[i][2])))
            ##Conversion of the quadrupolar moments from spherical to cartesian tensor :
            local_quadrupole_cartesian_form=conversion_quadrupole_tensor_spherical_to_cartesian(result_multipoles_all_overlap_centers[i][2])
            print('               Q_xx='+str(local_quadrupole_cartesian_form[0][0])+', Q_yy='+str(local_quadrupole_cartesian_form[1][1])+', Q_zz='+str(local_quadrupole_cartesian_form[2][2])+', Q_xy='+str(local_quadrupole_cartesian_form[0][1])+', Q_xz='+str(local_quadrupole_cartesian_form[0][2])+', Q_yz='+str(local_quadrupole_cartesian_form[1][2]))
            print(' ')
        if (l_max_user_all_overlap_centers[i]>=3):
            print('Octopole :     '+str(result_multipoles_all_overlap_centers[i][3]))
            print('               |Q3|='+str(np.linalg.norm(result_multipoles_all_overlap_centers[i][3])))
            print(' ')
        if (l_max_user_all_overlap_centers[i]>=4):
            print('Hexadecapole : '+str(result_multipoles_all_overlap_centers[i][4]))
            print('               |Q4|='+str(np.linalg.norm(result_multipoles_all_overlap_centers[i][4])))
        print(' ')


print('Check total sum of monopoles (= total charge) :')  
print(np.sum([result_multipoles_all_overlap_centers[i][0] for i in range(len(result_multipoles_all_overlap_centers))]))
print(' ')
   
logfile_DMA_all_overlap_centers.close()     
###############


filename_ESP_overlap_centers_output="H2O_Opt_PBE0_6-31Gd_no_PCM_ESP_potential_all_overlap_centers_order_"+str(l_max_user_all_overlap_centers[0])+".ESP"

#########################
#Calculation of the electrostatic potential (ESP) generated by multipole moments
#at all the overlap centers 'exact multipolar expansion' = exact part of he multipolar part
#of the whole potential (penetration term excluded)

t_begin_ESP_overlap_centers=time.clock()

ESP_potential_overlap_centers_multipoles_grid_points=[compute_electrostatic_potential_generated_multipoles_overlap_centers(grid_points[i][0],grid_points[i][1],grid_points[i][2],result_multipoles_all_overlap_centers,positions_final_expansion_site_all_overlap_centers) for i in range(len(grid_points))]

t_end_ESP_overlap_centers=time.clock()

print('Time to compute the ESP potential from overlap centers : '+str(t_end_ESP_overlap_centers-t_begin_ESP_overlap_centers))
print(' ')

writes_ESP_values_DMA_model(ESP_potential_overlap_centers_multipoles_grid_points,grid_points,total_nb_grid_points,filename_ESP_overlap_centers_output,ESP_files_dir)


