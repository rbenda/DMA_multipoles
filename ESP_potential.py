#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 13:47:36 2020

@author: rbenda
"""

import numpy as np
import time
import math

import scipy.spatial.distance

from auxiliary_functions import real_solid_harmonics_real_multipoles


############################
#CONSTANTS :
#1 Bohr in Angstroms (Gaussian QM output given in atomic units 
#i.e. bohr and Hartrees ==> e.g. 'zeta' gaussian exponents
#provided in Bohr^{-2}
conversion_bohr_angstrom=0.529177209

conversion_eV_in_Joules=1.602176634*10**(-19)
avogadro_number=6.02214076*10**(23)

#1 J/mol = 1.0364269656262173e-05 eV
conversion_kJ_per_mol_in_eV=1000*(1/(conversion_eV_in_Joules*avogadro_number))
#1 kcal/mol = 0.04337965064628533 (at 20°C ?)
conversion_kcal_per_mol_in_eV=4.1855*conversion_kJ_per_mol_in_eV

#BEWARE of units for epsilon_0 : must be in e/(V*Bohr) as the multipole moments of order l 
#are divided by a length**(l+1) [converted from Angstroms to Bohrs] times epsilon_0
#Value in F/m (SI units) :
##epsilon_0=8.85418782e-12 
#After conversion in e/(V*Bohr) : 1/(10^{10}*(1/0.529177209)) *  (1/(1.60217662*10^{-19})) * 8.85418782 *10^{-12}
epsilon_0=0.0029244181571875605

#After conversion in e**2/(Ha*Bohr) (multiplication by 27.2114 from e/(V*Bohr)) 
#==> 1/(4*pi*epsilon_0_atomic_units) = 1 a.u.
epsilon_0_atomic_units=0.07957747202001116

#Conversion from Hartrees/e (atomic units for the potential) into Volts :
#1 Ha/e = 27.211386245988 V
#If result of ESP wanted in Volts : multiply by this constant
conversion_atomic_units_Volts=27.211386245988

#Factor of conversion from Volts to kcal/mol / e (units of Tinker computed potential) :
#1 V = (1/0.043) kcal/mol / e
conversion_Volts_kcal_mol_per_e=(1./conversion_kcal_per_mol_in_eV)

#(1/epsilon_0)*(1 e/Bohr) \approx 7952.287499081084 kcal/mol /e
##conversion_epsilon_0_times_e_per_bohr_in_kcal_mol_per_e=7952.287499081084

#1 Ha \approx 627.3 kcal/mol (useful for the conversion of the electrostatic potentiel from Ha/e (a.u.) to kcal/mol / e)
conversion_Hartree_to_kcal_per_mol=conversion_Volts_kcal_mol_per_e*conversion_atomic_units_Volts
############################



#############################################################################################
#Potential (in atomic units : Ha/e) generated at the point (x,y,z) in space by the sets of final distributed multipoles
#'result_multipoles' previously computed previously thanks to the function 'compute_DMA_multipole_moment_final_expansion_center_all_sites_all_orders()'
#'positions_final_expansion_sites' computed previously thanks to the function 'coordinates_final_expansion_centers()'
#(positions of the final DMA sites)
def compute_electrostatic_potential_generated_DMA_multipoles(x,y,z,result_multipoles,positions_final_expansion_sites):
    
    #Unit : e/Bohr
    total_ESP=0
    
    vec_r=[x,y,z]
    
    #Number of final DMA sites
    #Ex : = nb of atoms for 'index_philosophy=1' (AMOEBA philosophy : only atoms as expansion centers)
    M=len(positions_final_expansion_sites)
    
    ##################################
    #Version 1 : (not vectorized)
    
    #Sum on all the contributions from the DMA final sites
    for i in range(M):
        
        x_S=positions_final_expansion_sites[i][0]
        y_S=positions_final_expansion_sites[i][1]
        z_S=positions_final_expansion_sites[i][2]

        #'len(result_multipoles)' is the order up to which multipole moments were computed on the i^th final site S_i
        #Ex : = 2 if multipoles up to quadrupoles were computed
        for l in range(len(result_multipoles[i])):
            
            for m in range(-l,l+1):
                
                #Distance from the i^th final DMA site S_i and the point \vec{r}=[x,y,z] at
                #which we evaluate the potential
                #Conversion in Bohrs
                distance=(1/conversion_bohr_angstrom)*scipy.spatial.distance.pdist([vec_r,positions_final_expansion_sites[i]])[0]
                
                #Result of 'real_solid_harmonics_real_multipoles' is in Bohr**l
                #as 'result_multipoles[i][l]' (multipole moments of order l, in the i^{th} DMA site)
                #is in e.Bohr**l (atomic units)
                ##print(result_multipoles[i][l])
                ##print(l+m)
                ##Facteur 4*math.pi ??
                total_ESP+=(1/(2*l+1))*result_multipoles[i][l][l+m]*real_solid_harmonics_real_multipoles(vec_r,l,m,x_S,y_S,z_S)/distance**(2*l+1)
    
    ##################################

    ##################################
    #Version 2 (vectorized) :
        
    #tab_contributions=[[np.sum([(1/(2*l+1))*result_multipoles[i][l][l+m]*real_solid_harmonics_real_multipoles(vec_r,l,m,positions_final_expansion_sites[i][0],positions_final_expansion_sites[i][1],positions_final_expansion_sites[i][2])/((1/conversion_bohr_angstrom)*scipy.spatial.distance.pdist([vec_r,positions_final_expansion_sites[i]])[0])**(2*l+1) for m in range(-l,l+1)]) for l in range(len(result_multipoles[i]))] for i in range(M)]
   
    ##################################  
    
    #Final result in (e/Bohr)*(1/[epsilon_0])  (where [epsilon_0]=F/m expressed in good units : e/(V*Bohr))
    #=> result in V ==> conversion in kcal/mol / e (other units of potential values, used in Tinker)
    #4*math.pi*epsilon_0_atomic_units=1 in atomic units
    ##return total_ESP/(4*math.pi*epsilon_0_atomic_units)   
    return total_ESP
    
    #return np.sum(tab_contributions)
#############################################################################################



#############################################################################################
#Potential generated at the point (x,y,z) in space by the sets of final distributed multipoles
#'result_multipoles' previously computed previously thanks to the function 'compute_DMA_multipole_moment_final_expansion_center_all_sites_all_orders()'
#'positions_final_expansion_sites' computed previously thanks to the function 'coordinates_final_expansion_centers()'
#(positions of the final DMA sites)


#############################################################################################
#Computes the exact multipolar potential, i.e. the potential generated by the distribution of 
#multipole moments located at all overlap centers (case index_philosophy=5)
#'result_multipoles_overlap_centers' computed as :
#compute_DMA_multipole_moment_final_expansion_center_all_sites_all_orders(QM_data,boolean_cartesian_D_shells,boolean_cartesian_F_shells,boolean_all_atomic_sites_origin,atomic_sites_to_omit_origin,user_extra_final_sites_origin,index_philosophy_origin,redistribution_strategy_origin,l_max_user_origin)
#with all overlap centers chosen as final DMA site (index_philosophy=5)
def compute_electrostatic_potential_generated_multipoles_overlap_centers(x,y,z,result_multipoles_overlap_centers,positions_final_expansion_sites):
    
    #Unit : e/Bohr
    #total_ESP=0
    
    vec_r=[x,y,z]
    
    #Number of final DMA sites
    #Ex : = nb of atoms for 'index_philosophy=1' (AMOEBA philosophy : only atoms as expansion centers)
    M=len(positions_final_expansion_sites)
    
    ###################################
    #Version 1 : (not vectorized)
    """
    #Sum on all the contributions from the DMA final sites)=all overlap centers 
    for i in range(M):
        
        x_S=positions_final_expansion_sites[i][0]
        y_S=positions_final_expansion_sites[i][1]
        z_S=positions_final_expansion_sites[i][2]

        #'len(result_multipoles)' is the order up to which multipole moments were computed on the i^th final site S_i
        #Ex : = 2 if multipoles up to quadrupoles were computed
        for l in range(len(result_multipoles_overlap_centers[i])):
    
            for m in range(-l,l+1):
                
                #Distance from the i^th final DMA site S_i and the point \vec{r}=[x,y,z] at
                #which we evaluate the potential
                #Conversion in Bohrs
                distance=(1/conversion_bohr_angstrom)*scipy.spatial.distance.pdist([vec_r,positions_final_expansion_sites[i]])[0]
                
                #Result of 'real_solid_harmonics_real_multipoles' is in Bohr**l (atmoic units)
                #as 'result_multipoles[i][l]' (multipole moments of order l, in the i^{th} DMA site)
                #is in e.Bohr**l (atomic units)
                ##Facteur 4*math.pi ??
                total_ESP+=(1/(2*l+1))*result_multipoles_overlap_centers[i][l][l+m]*real_solid_harmonics_real_multipoles(vec_r,l,m,x_S,y_S,z_S)/distance**(2*l+1)
    """
    ###################################

    ###################################
    #Version 2 : (vectorized version)
    tab_contributions=[[np.sum([(1/(2*l+1))*result_multipoles_overlap_centers[i][l][l+m]*real_solid_harmonics_real_multipoles(vec_r,l,m,positions_final_expansion_sites[i][0],positions_final_expansion_sites[i][1],positions_final_expansion_sites[i][2])/((1/conversion_bohr_angstrom)*scipy.spatial.distance.pdist([vec_r,positions_final_expansion_sites[i]])[0])**(2*l+1) for m in range(-l,l+1)]) for l in range(0,len(result_multipoles_overlap_centers[i]))] for i in range(M)] 
    ###################################
    
    #Final result in (e/Bohr)*(1/[epsilon_0])  (where [epsilon_0]=F/m expressed in good units : e/(V*Bohr))
    #=> result in V ==> conversion in kcal/mol / e (other units of potential values, used in Tinker) 
    #4*math.pi*epsilon_0_atomic_units=1 in atomic units
    ##return total_ESP/(4*math.pi*epsilon_0_atomic_units) 
    #return total_ESP
    
    return np.sum(tab_contributions)
#############################################################################################


################################################################################################
#Function that reads and stores the coordinates of a grid file
#(grid points on a surface around a molecule)
#'Grid_points_files_dir' = directory where are stored grid points files
#'filename_grid' = the name of the grid file (obtain using Tinker ./potential executable)
def reads_grid_file(Grid_points_files_dir,filename_grid):

    file_object  = open(Grid_points_files_dir+filename_grid,"r")
    
    #String object :
    grid_points_readlines=file_object.readlines()
    #print(grid_points_surface_around_molecule)
    
    #Total number of grid points = number of lines in the .grid file :
    #total_nb_grid_points=len(grid_points_readlines)
    
    #List of coordinates of the grid points (each in a format 
    #of table of length 3, of 'string' objets converted to float format)
    grid_points=[]
    
    for i in range(len(grid_points_readlines)):
        
        grid_points.append([float(grid_points_readlines[i].split("    ")[1+k]) for k in range(3)])

    return grid_points
################################################################################################


################################################################################################
#Input arguments :
#- ESP_potential_DMA_final_sites_multipoles_grid_points = array of values of
#  the ESP potential at the grid points
# - filename_ESP_DMA_final_sites_output = name of the output ESP file where ESP values are written
# - ESP_files_dir = directory where .ESP files are stored
def writes_ESP_values_DMA_model(ESP_potential_DMA_final_sites_multipoles_grid_points,grid_points,total_nb_grid_points,filename_ESP_DMA_final_sites_output,ESP_files_dir):
    
    #Opening the file of specified name, in 'writing' mode
    file_object_DMA_final_sites_output  = open(ESP_files_dir+filename_ESP_DMA_final_sites_output,"w")

    #As first line, we write the total number of grid points :
    file_object_DMA_final_sites_output.write(str(total_nb_grid_points))
    file_object_DMA_final_sites_output.write("\n")
    file_object_DMA_final_sites_output.write("x_i  y_i  z_i  ESP value (a.u)")
    file_object_DMA_final_sites_output.write("\n")
    
    #######
    #Writing the DMA ESP potential in an output file :
    for i in range(len(grid_points)):
    
        ##Print also coordinates of the grid points (+ESP value) : 
        #==> so that format of a Gaussian .cube file ??
        file_object_DMA_final_sites_output.write(str(grid_points[i][0])+' '+str(grid_points[i][1])+' '+str(grid_points[i][2])+' '+str(ESP_potential_DMA_final_sites_multipoles_grid_points[i]))
        file_object_DMA_final_sites_output.write("\n")
        
    file_object_DMA_final_sites_output.close()
################################################################################################




###########################################################################
#Fills a table with the values of the ESP generated at the grid points
#by the (DMA) multipole moments
def build_ESP_multipoles_moments_from_ESP_file(ESP_files_dir,filename_ESP_output):
    
    file_object_ESP_output  = open(ESP_files_dir+filename_ESP_output,"r")    

    ESP_multipoles=file_object_ESP_output.readlines()
    
    #We skip the first line (total number of grid points)
    total_nb_grid_points=len(ESP_multipoles)-1
    print('Total number of grid points = '+str(total_nb_grid_points))
    print(' ')
    
    #Values of the ESP potential generated by the series of multipole moments
    #(that has given birth to this .ESP file)
    MM_ESP_values_multipoles=[]
    
    #Reading the DMA ESP potential in the .ESP file : 
    #(2 first lines : number of grid points / explanation line)
    for i in range(2,1+total_nb_grid_points):
    
        MM_ESP_values_multipoles.append(float(ESP_multipoles[i].split(' ')[3]))
    
    file_object_ESP_output.close()
    
    return MM_ESP_values_multipoles
###########################################################################

###########################################################################
#Reads the value of the QM potential (ESP) in a .cube file generated by Gaussian
def build_QM_ESP_from_cube_file(QM_ESP_files_dir,filename_QM_cube_file):
        
    file_object_QM_cube_file  = open(QM_ESP_files_dir+filename_QM_cube_file,"r")
        
    cube_file_readlines=file_object_QM_cube_file.readlines()
    
    #List of QM ESP values at all grid points  => must be of length total_nb_grid_points)
    QM_ESP_values=[]
    
    #We skip the first (2+1+3+nb_atoms) lines in the cube file
    #The third line is "NAtoms, X-Origin, Y-Origin, Z-Origin"
    ##nb_atoms=int(float(cube_file_readlines[2].split('    ')[1]))
    
    ##########
    #For a cube file generated by Psi4 :
    tab=cube_file_readlines[2].split('  ')
    for h in range(len(tab)):
        #The first encountered number is the total number of atoms :
        if (tab[h]!=''):
            nb_atoms=int(float(tab[h][1]))
    #########
    
    print('Nb of atoms ='+str(nb_atoms))
    print(' ')
    
    index_init=2+1+3+nb_atoms
    
    for i in range(index_init,int(len(cube_file_readlines))):
        
        #For Gaussian generated cube files :
        QM_ESP_values.append(float(cube_file_readlines[i].split(" ")[4]))
        
        #For Psi4  generated cube files

    return QM_ESP_values
###########################################################################

###########################################################################
#Reads the value of the QM potential (ESP) in a grid_esp.dat file generated by Psi4
#(values of the QM ESP at all the points (x,y,z) listed in the grid.dat file)
#thanks to the syntax :
#'E, wfn = prop('scf', properties=["GRID_ESP", "GRID_FIELD"], return_wfn=True)'
#in the Psi4 input.dat file
def build_QM_ESP_from_Psi4_dat_file(QM_ESP_files_dir,filename_Psi4_QM_dat_file):
        
    file_object_QM_ESP_file  = open(QM_ESP_files_dir+filename_Psi4_QM_dat_file,"r")
        
    cube_file_readlines=file_object_QM_ESP_file.readlines()
    
    #List of QM ESP values at all grid points  => must be of length total_nb_grid_points)
    QM_ESP_values=[]
  
    for i in range(int(len(cube_file_readlines))):

        QM_ESP_values.append(float(cube_file_readlines[i]))

    return QM_ESP_values
###########################################################################

############################################################################
#Relative Root Mean Square Deviation (RRMSD) definition used also e.g. in :
#- Fabienne Vigné-Maeder and Pierre Claverie. The exact multicenter multipolar part of a molecular charge distribution and
#its simplified representations. The Journal of Chemical Physics, 88(8):4934–4948, apr 1988 (Table II)
#- Ren, Ponder, J Comput Chem (23) 1497–1506, 2002
def computes_RRMS_deviation_QM_ESP_DMA_final_sites_ESP(total_nb_grid_points,ESP_values_DMA,QM_ESP_values):
    
    #print('Check total nb of grid points : '+str(total_nb_grid_points)+' = '+str(len(QM_ESP_values))+' = '+str(len(ESP_values_DMA)))
        
    numerator_RRMS_deviation_QM_ESP_DMA_final_sites_ESP=0
    denominator_RRMS_deviation_QM_ESP_DMA_final_sites_ESP=0
    
    for k in range(total_nb_grid_points):
        
        numerator_RRMS_deviation_QM_ESP_DMA_final_sites_ESP+=(QM_ESP_values[k]-ESP_values_DMA[k])**2
        denominator_RRMS_deviation_QM_ESP_DMA_final_sites_ESP+=QM_ESP_values[k]**2
        
    RRMS_deviation_QM_ESP_DMA_final_sites_ESP=numerator_RRMS_deviation_QM_ESP_DMA_final_sites_ESP/denominator_RRMS_deviation_QM_ESP_DMA_final_sites_ESP
    #(non-dimensional quantity)
    
    #Final result in % deviation (non-dimensional quantity)
    return 100*RRMS_deviation_QM_ESP_DMA_final_sites_ESP
############################################################################

############################################################################
#Definition of the mean absolute deviation taken from
#Chipot et al. J. Phys. Chem. 1993, 97, 6628-6636 (equation (25))
def computes_mean_absolute_deviation_QM_ESP_DMA_final_sites_ESP(total_nb_grid_points,ESP_values_DMA,QM_ESP_values):
    
    #print('Check total nb of grid points : '+str(total_nb_grid_points)+' = '+str(len(QM_ESP_values))+' = '+str(len(ESP_values_DMA)))
        
    
    mean_absolute_deviation_QM_ESP_DMA_final_sites_ESP=0
    
    for k in range(total_nb_grid_points):
        
        #Non-dimensiolan quantity
        mean_absolute_deviation_QM_ESP_DMA_final_sites_ESP+=abs(QM_ESP_values[k]-ESP_values_DMA[k])/abs(QM_ESP_values[k])
            
    #Final result in % deviation (non-dimensional quantity)
    return 100*mean_absolute_deviation_QM_ESP_DMA_final_sites_ESP/total_nb_grid_points
   
############################################################################


############################################################################
#Definition of the RMSD deviation used also in :
#Chipot et al. J. Phys. Chem. 1993, 97, 6628-6636 (equation (24))
#or (equivalently) Ren, Ponder, J Comput Chem (23) 1497–1506, 2002
def computes_RMS_deviation_QM_ESP_DMA_final_sites_ESP(total_nb_grid_points,ESP_values_DMA,QM_ESP_values):
  
    RMS_deviation_QM_ESP_DMA_final_sites_ESP=0
    
    for k in range(total_nb_grid_points):
        
        RMS_deviation_QM_ESP_DMA_final_sites_ESP+=(QM_ESP_values[k]-ESP_values_DMA[k])**2
            
    #Final result in atomic units
    #(same unit as [QM_ESP_values]=[ESP_values_DMA])
    return math.sqrt(RMS_deviation_QM_ESP_DMA_final_sites_ESP/total_nb_grid_points)
############################################################################ 





############################################################################
#Reads 'isa_multipoles.txt' or 'becke_multipoles.txt' files, output of Hipart code
#(computation of multipole moments by integration in the real space, decomposed into Voronoi-like cells)
#Those multipole moments are provided in atomic units by Hipart
def reads_multipoles_output_Hipart(Hipart_files_dir,hipart_multipole_txt_filename):

    file_object_hipart_multipoles  = open(Hipart_files_dir+hipart_multipole_txt_filename,"r")
        
    hipart_multipoles_file_readlines=file_object_hipart_multipoles.readlines()
    
    #We skip te first 4 lines whose structure is :
    #'number of atoms: ..'
    #'number of fields: ..'
    #'Multipoles'
    #'----------'
    nb_atoms=len(hipart_multipoles_file_readlines)-4
    
    #List of list of multipole moments for each atom :
    #Order for each atom :  (0,0)  , (1,0) , (1,1+) , (1,1-) , (2,0) , (2,1+) , (2,1-) , (2,2+), (2,2-) , (3,0) , (3,1+) , (3,1-) , (3,2+), (3,2-)  , (3,3+) , (3,3-), (4,0),(4,1+) , (4,1-) ,(4,2+) ,(4,2-) ,(4,3+) ,(4,3-)  , (4,4+) ,(4,4-)
    multipole_moments_atoms=[]
    
    for i in range(4,4+nb_atoms):
        
        #print(isa_multipoles_file_readlines[i].split(" "))
        #print(' ')
        
        list_multipoles_atom=hipart_multipoles_file_readlines[i].split(" ")
        
        #Index (after splitting) of the first multipole moment of the list of multipole moments of atom 'i' (monopole)
        index_first_multipoles=14
        
        list_multipoles_atom_cleaned=[]
        
        #We store only the float numbers and not the '' or '|' or 'O' (element name string)
        for k in range(index_first_multipoles,len(list_multipoles_atom)):
            
            if (list_multipoles_atom[k]!=''):
                
                list_multipoles_atom_cleaned.append(float(list_multipoles_atom[k]))
                    
        multipole_moments_atom_i=[]
        
        #Local monopole on atom i :
        multipole_moments_atom_i.append([list_multipoles_atom_cleaned[0]])
        
        #Local dipole on atom i
        multipole_moments_atom_i.append([list_multipoles_atom_cleaned[1+k] for k in range(3)])

        #Local quadrupole on atom i
        multipole_moments_atom_i.append([list_multipoles_atom_cleaned[4+k] for k in range(5)])

        #Local octopole on atom i
        multipole_moments_atom_i.append([list_multipoles_atom_cleaned[9+k] for k in range(7)])
        
        #Local hexadecapole on atom i
        multipole_moments_atom_i.append([list_multipoles_atom_cleaned[16+k] for k in range(9)])
       
        multipole_moments_atoms.append(multipole_moments_atom_i)
        
    return multipole_moments_atoms
############################################################################


############################################################################
#Reads 'isa_multipoles.txt' or 'becke_multipoles.txt' files, output of Hipart code
#(computation of multipole moments by integration in the real space, decomposed into Voronoi-like cells)
#Those multipole moments are provided in atomic units by Hipart
#AND reorders them so that : [[Q_(00)],[Q_(1,-1),Q(1,0),Q(1,1)],[Q(2,-2),Q(2,-1),Q(2,0),Q(2,1),Q(2,2)],...]
#(same order as that given by output of 'compute_DMA_multipole_moment_final_expansion_center_all_sites_all_orders()' in multipoles_DMA.py)
def reads_reorders_multipoles_output_Hipart(Hipart_files_dir,hipart_multipole_txt_filename):

    file_object_hipart_multipoles  = open(Hipart_files_dir+hipart_multipole_txt_filename,"r")
        
    hipart_multipoles_file_readlines=file_object_hipart_multipoles.readlines()
    
    #We skip te first 4 lines whose structure is :
    #'number of atoms: ..'
    #'number of fields: ..'
    #'Multipoles'
    #'----------'
    nb_atoms=len(hipart_multipoles_file_readlines)-4
    
    #List of list of multipole moments for each atom :
    #Order for each atom :  (0,0)  , (1,0) , (1,1+) , (1,1-) , (2,0) , (2,1+) , (2,1-) , (2,2+), (2,2-) , (3,0) , (3,1+) , (3,1-) , (3,2+), (3,2-)  , (3,3+) , (3,3-), (4,0),(4,1+) , (4,1-) ,(4,2+) ,(4,2-) ,(4,3+) ,(4,3-)  , (4,4+) ,(4,4-)
    multipole_moments_atoms=[]
    
    for i in range(4,4+nb_atoms):
        
        #print(isa_multipoles_file_readlines[i].split(" "))
        #print(' ')
        
        list_multipoles_atom=hipart_multipoles_file_readlines[i].split(" ")
        
        #Index (after splitting) of the first multipole moment of the list of multipole moments of atom 'i' (monopole)
        index_first_multipoles=14
        
        list_multipoles_atom_cleaned=[]
        
        #We store only the float numbers and not the '' or '|' or 'O' (element name string)
        for k in range(index_first_multipoles,len(list_multipoles_atom)):
            
            if (list_multipoles_atom[k]!=''):
                
                list_multipoles_atom_cleaned.append(float(list_multipoles_atom[k]))
                    
        multipole_moments_atom_i=[]
        
        #Local monopole on atom i :
        multipole_moments_atom_i.append([list_multipoles_atom_cleaned[0]])
        
        
        #Local dipole on atom i
        multipole_moments_atom_i.append([])
        #First Q(1,-1) :
        multipole_moments_atom_i[1].append(list_multipoles_atom_cleaned[3])
        #Then Q(1,0) :
        multipole_moments_atom_i[1].append(list_multipoles_atom_cleaned[1])
        #Then Q(1,+1) :
        multipole_moments_atom_i[1].append(list_multipoles_atom_cleaned[2])
           
        #Local quadrupole on atom i
        multipole_moments_atom_i.append([])
        #First Q(2,-2) :
        multipole_moments_atom_i[2].append(list_multipoles_atom_cleaned[8])
        #Then Q(2,-1) :
        multipole_moments_atom_i[2].append(list_multipoles_atom_cleaned[6])
        #Then Q(2,0) :
        multipole_moments_atom_i[2].append(list_multipoles_atom_cleaned[4])
        #Then Q(2,1) :
        multipole_moments_atom_i[2].append(list_multipoles_atom_cleaned[5])
        #Then Q(2,2) :
        multipole_moments_atom_i[2].append(list_multipoles_atom_cleaned[7])


        #Local octopole on atom i
        multipole_moments_atom_i.append([])
        #First Q(3,-3):
        multipole_moments_atom_i[3].append(list_multipoles_atom_cleaned[15])
        #Then Q(3,-2):
        multipole_moments_atom_i[3].append(list_multipoles_atom_cleaned[13])
        #Then Q(3,-1):
        multipole_moments_atom_i[3].append(list_multipoles_atom_cleaned[11])
        #Then Q(3,0):
        multipole_moments_atom_i[3].append(list_multipoles_atom_cleaned[9])
        #Then Q(3,1):
        multipole_moments_atom_i[3].append(list_multipoles_atom_cleaned[10])
        #Then Q(3,2):
        multipole_moments_atom_i[3].append(list_multipoles_atom_cleaned[12])
        #Then Q(3,3):
        multipole_moments_atom_i[3].append(list_multipoles_atom_cleaned[14])
        
        #Local hexadecapole on atom i
        multipole_moments_atom_i.append([])
        #First Q(4,-4):
        multipole_moments_atom_i[4].append(list_multipoles_atom_cleaned[24])
        #Then Q(4,-3) :
        multipole_moments_atom_i[4].append(list_multipoles_atom_cleaned[22])
        #Then Q(4,-2) :
        multipole_moments_atom_i[4].append(list_multipoles_atom_cleaned[20])
        #Then Q(4,-1) :
        multipole_moments_atom_i[4].append(list_multipoles_atom_cleaned[18])
        #Then Q(4,0) :
        multipole_moments_atom_i[4].append(list_multipoles_atom_cleaned[16])
        #Then Q(4,1) :
        multipole_moments_atom_i[4].append(list_multipoles_atom_cleaned[17])
        #Then Q(4,2) :
        multipole_moments_atom_i[4].append(list_multipoles_atom_cleaned[19])
        #Then Q(4,3) :
        multipole_moments_atom_i[4].append(list_multipoles_atom_cleaned[21])
        #Then Q(4,4) :
        multipole_moments_atom_i[4].append(list_multipoles_atom_cleaned[23])
        
        #Pour automatiser :
        #Suite d'indices [16+9-2*k (k=0..4)] puis [16] puis [16+1+2*k ; k=0..3]
            
        multipole_moments_atoms.append(multipole_moments_atom_i)
        
    return multipole_moments_atoms
############################################################################




