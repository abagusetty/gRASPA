#pragma once
#include "sycl_device.hpp"
#include <iostream>
#include <filesystem>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>
void create_movie_file(size_t Cycle, Atoms* System, Components SystemComponents, ForceField FF, Boxsize Box, std::vector<std::string> AtomNames);

static inline void create_Restart_file(size_t Cycle, Atoms* System, Components SystemComponents, ForceField FF, Boxsize Box, std::vector<std::string> AtomNames);

static inline void WriteBox_LAMMPS(Atoms* System, Components SystemComponents, ForceField FF, Boxsize Box, std::ofstream& textrestartFile, std::vector<std::string> AtomNames)
{
  size_t NumComponents = SystemComponents.Total_Components;
  size_t NumberOfAtoms = 0;
  for(size_t i = 0; i < NumComponents; i++)
  {
    NumberOfAtoms += SystemComponents.NumberOfMolecule_for_Component[i] * SystemComponents.Moleculesize[i];
    printf("Printing: Component: %zu (%s), NumMol: %zu, Molsize: %zu\n", i, SystemComponents.MoleculeName[i].c_str(), SystemComponents.NumberOfMolecule_for_Component[i], SystemComponents.Moleculesize[i]);
  }
  textrestartFile << "# LAMMPS data file written by WriteLammpsdata function in RASPA(written by Zhao Li)" << '\n';
  textrestartFile << NumberOfAtoms << " atoms" << '\n';
  textrestartFile << 0 << " bonds" << '\n';
  textrestartFile << 0 << " angles" << '\n';
  textrestartFile << 0 << " dihedrals" << '\n';
  textrestartFile << 0 << " impropers" << '\n';
  textrestartFile << 0 << " " << Box.Cell[0] << " xlo xhi" << '\n';
  textrestartFile << 0 << " " << Box.Cell[4] << " ylo yhi" << '\n';
  textrestartFile << 0 << " " << Box.Cell[8] << " zlo zhi" << '\n';
  textrestartFile << Box.Cell[3] << " " << Box.Cell[6] << " " << Box.Cell[7] << " xy xz yz" << '\n' << '\n';
  textrestartFile << FF.size << " atom types" << '\n'; 
  textrestartFile << 0 << " bond types" << '\n';
  textrestartFile << 0 << " angle types" << '\n';
  textrestartFile << 0 << " dihedral types" << '\n';
  textrestartFile << 0 << " improper types" << '\n' << '\n';
  //textrestartFile << std::puts("%zu bonds\n", NumberOfAtoms);
  textrestartFile << "Masses" << '\n' << '\n';
  for(size_t i = 0; i < FF.size; i++)
  { 
    double mass = 0.0;
    textrestartFile << i+1 << " " << mass << " # " << AtomNames[i] << '\n';
  }
  textrestartFile << '\n' << "Pair Coeffs" << '\n' << '\n'; 
  for(size_t i = 0; i < FF.size; i++)
  { 
    const size_t row = i*FF.size+i;
    textrestartFile << i+1 << " " << FF.epsilon[row]/120.2/4.184*1.2 << " " << FF.sigma[row] << " # " << AtomNames[i] << '\n';
  }
}

static inline void WriteAtoms_LAMMPS(Atoms* System, Components SystemComponents, Boxsize Box, std::ofstream& textrestartFile, std::vector<std::string> AtomNames)
{
  size_t NumComponents = SystemComponents.Total_Components;
  textrestartFile << '\n' << "Atoms" << '\n' << '\n';
  size_t Atomcount=0; size_t Molcount=0;
  for(size_t i = 0; i < NumComponents; i++)
  {
    Atoms Data = System[i];
    printf("Component %zu, Molsize: %zu\n", i, Data.Molsize);
    double3 Wrap_shift;
    for(size_t j = 0; j < Data.size; j++)
    {
      //Wrap To Box//
      double3 pos = Data.pos[j];
      if(j % Data.Molsize == 0) 
      {
        WrapInBox(pos, Box.Cell, Box.InverseCell, Box.Cubic);
        Wrap_shift = pos - Data.pos[j];
      }
      else
      {
        pos += Wrap_shift;
      }
      textrestartFile << Atomcount + 1 << " " << Molcount + Data.MolID[j] + 1
                      << " " << Data.Type[j] + 1 << " " << Data.charge[j] << " "
                      << pos.x() << " " << pos.y() << " " << pos.z() << " # "
                      << AtomNames[Data.Type[j]] << '\n';
      Atomcount++;
    }
    Molcount+=SystemComponents.NumberOfMolecule_for_Component[i];
  }
}

static inline void WriteAtoms_Restart(Atoms* System, Components SystemComponents, std::ofstream& textrestartFile, std::vector<std::string> AtomNames)
{
  size_t NumComponents = SystemComponents.Total_Components;
  textrestartFile << "Reactions: 0" << "\n";
  //size_t Atomcount=0; 
  size_t Molcount=0;
  for(size_t i = SystemComponents.NumberOfFrameworks; i < NumComponents; i++)
  {
    Atoms Data = System[i];
    textrestartFile << '\n' << "Component: " <<  i - SystemComponents.NumberOfFrameworks << "   Adsorbate " << SystemComponents.NumberOfMolecule_for_Component[i] << " molecules of " << SystemComponents.MoleculeName[i] << '\n';
    textrestartFile << "------------------------------------------------------------------------" << "\n";
    size_t molsize = SystemComponents.Moleculesize[i];
    //First write positions//
    for(size_t j = 0; j < Data.size; j++)
      textrestartFile << "Adsorbate-atom-position:" << " " << Molcount+Data.MolID[j] << " " << j - Data.MolID[j]*molsize << " " << std::setprecision (15) << Data.pos[j].x() << "  " << std::setprecision (15) << Data.pos[j].y() << "  " << std::setprecision (15) << Data.pos[j].z() << '\n';
    //Then write velocities (Zhao's note: Not implemented yet)//
    for(size_t j = 0; j < Data.size; j++)
      textrestartFile << "Adsorbate-atom-velocity:" << " " << Molcount+Data.MolID[j] << " " << j - Data.MolID[j]*molsize << " " << 0.0 << "  " << 0.0 << "  " << 0.0 << '\n';
    //Then write force (Zhao's note: Not implemented yet)//
    for(size_t j = 0; j < Data.size; j++)
      textrestartFile << "Adsorbate-atom-force:" << " " << Molcount+Data.MolID[j] << " " << j - Data.MolID[j]*molsize << " " << 0.0 << "  " << 0.0 << "  " << 0.0 << '\n';
    //Then write charge//
    for(size_t j = 0; j < Data.size; j++)
      textrestartFile << "Adsorbate-atom-charge:" << " " << Molcount+Data.MolID[j] << " " << j - Data.MolID[j]*molsize << " " << Data.charge[j] << '\n';
    //Then write scale//
    for(size_t j = 0; j < Data.size; j++)
      textrestartFile << "Adsorbate-atom-scaling:" << " " << Molcount+Data.MolID[j] << " " << j - Data.MolID[j]*molsize << " " << Data.scale[j] << '\n';
    //Finally write fixed (Zhao's note: Not implemented yet)//
    for(size_t j = 0; j < Data.size; j++)
      textrestartFile << "Adsorbate-atom-fixed:" << " " << Molcount+Data.MolID[j] << " " << j - Data.MolID[j]*molsize << " " << "0 0 0" << '\n';
    Molcount+=SystemComponents.NumberOfMolecule_for_Component[i];
  }
}

static inline void WriteComponent_Restart(
    Atoms *System, Components SystemComponents, std::ofstream &textrestartFile,
    Boxsize &Box, double3 &MaxTranslation, double3 &MaxRotation)
{
  textrestartFile << "Components: " << SystemComponents.Total_Components - SystemComponents.NumberOfFrameworks << " (Adsorbates " << SystemComponents.TotalNumberOfMolecules-SystemComponents.NumberOfFrameworks << ", Cations 0)" << "\n";
  textrestartFile << "========================================================================\n"; 
  for(size_t i = SystemComponents.NumberOfFrameworks; i < SystemComponents.Total_Components; i++)
  {
    textrestartFile << "Components 0 (" << SystemComponents.MoleculeName[i] << ") " << "\n";
    int fracID = -1;
    if(SystemComponents.hasfractionalMolecule[i])
    {
      fracID = SystemComponents.Lambda[i].FractionalMoleculeID;
      textrestartFile << "Fractional-molecule-id component " << i-SystemComponents.NumberOfFrameworks << ": " << fracID << "\n";
      textrestartFile << "Lambda-factors component " << i-SystemComponents.NumberOfFrameworks << ": " << SystemComponents.Lambda[i].WangLandauScalingFactor << "\n";
      textrestartFile << "Number-of-biasing-factors component " << i-SystemComponents.NumberOfFrameworks << ": " << SystemComponents.Lambda[i].binsize << "\n";
      textrestartFile << "Biasing-factors component " << i-SystemComponents.NumberOfFrameworks << ": "; 
      for(size_t j = 0; j < SystemComponents.Lambda[i].binsize; j++)
        textrestartFile << SystemComponents.Lambda[i].biasFactor[j] << " ";
      textrestartFile << "\n";
      textrestartFile << "Maximum-CF-Lambda-change component " << i-SystemComponents.NumberOfFrameworks << ": 0.50000\n"; //Zhao's note: continuous lambda not implemented
      textrestartFile << "Maximum-CBCF-Lambda-change component " << i-SystemComponents.NumberOfFrameworks << ": 0.50000\n"; //Zhao's note: continuous lambda not implemented
    }
    textrestartFile << "\n";
    textrestartFile << "Maximum-translation-change component "
                    << i - SystemComponents.NumberOfFrameworks << ": "
                    << MaxTranslation.x() << " " << MaxTranslation.y() << " "
                    << MaxTranslation.z() << "\n";
    textrestartFile << "Maximum-translation-in-plane-change component " << i-SystemComponents.NumberOfFrameworks << ": " << "0.000000,0.000000,0.000000" << "\n";
    textrestartFile << "Maximum-rotation-change component "
                    << i - SystemComponents.NumberOfFrameworks << ": "
                    << MaxRotation.x() << " " << MaxRotation.y() << " "
                    << MaxRotation.z() << "\n";
    textrestartFile << "\n";
  }
}

static inline void
WriteCellInfo_Restart(Atoms *System, Components SystemComponents,
                      std::ofstream &textrestartFile, Boxsize &Box,
                      double3 MaxTranslation, double3 MaxRotation)
{
  //size_t Atomcount=0;
  textrestartFile << "Cell info:\n"; 
  textrestartFile << "========================================================================\n";
  textrestartFile << "number-of-unit-cells: 1 1 1\n"; //Zhao's note: not allowing for multiple unit cells for now//
  //First write Unit Box sizes//
  textrestartFile << "unit-cell-vector-a:" << " " << Box.Cell[0] << " " << Box.Cell[1] << " " << Box.Cell[2] << '\n';
  textrestartFile << "unit-cell-vector-b:" << " " << Box.Cell[3] << " " << Box.Cell[4] << " " << Box.Cell[5] << '\n';
  textrestartFile << "unit-cell-vector-c:" << " " << Box.Cell[6] << " " << Box.Cell[7] << " " << Box.Cell[8] << '\n';
  textrestartFile << "\n";
  //Then write Total Box sizes//
  textrestartFile << "cell-vector-a:" << " " << Box.Cell[0] << " " << Box.Cell[1] << " " << Box.Cell[2] << '\n';
  textrestartFile << "cell-vector-b:" << " " << Box.Cell[3] << " " << Box.Cell[4] << " " << Box.Cell[5] << '\n';
  textrestartFile << "cell-vector-c:" << " " << Box.Cell[6] << " " << Box.Cell[7] << " " << Box.Cell[8] << '\n';
  textrestartFile << "\n";
  //Cell lengths//
  textrestartFile << "cell-lengths:" << " " << Box.Cell[0] << " " << Box.Cell[4] << " " << Box.Cell[8] << " " << "\n";
  textrestartFile << "cell-angles:" << " " << 90.00 << " " << 90.00 << " " << 90.00 << " " << "\n";
  textrestartFile << "\n\n";
  //Maximum changes for MC-moves (Zhao's note: Not implemented)//
  textrestartFile << "Maximum changes for MC-moves:" << "\n"; 
  textrestartFile << "========================================================================" << "\n";
  textrestartFile << "Maximum-volume-change: 0.006250" << "\n";
  textrestartFile << "Maximum-Gibbs-volume-change: 0.025000" << "\n";
  textrestartFile << "Maximum-box-shape-change: 0.100000 0.100000 0.100000, 0.100000 0.100000 0.100000, 0.100000 0.100000 0.100000" << "\n";
  textrestartFile << "\n\n";
  //Acceptance targets for MC-moves (Zhao's note: Not implemented)//
  textrestartFile << "Acceptance targets for MC-moves:" << "\n";
  textrestartFile << "========================================================================" << "\n";
  textrestartFile << "Target-volume-change: 0.500000" << "\n"; 
  textrestartFile << "Target-box-shape-change: 0.500000" << "\n";
  textrestartFile << "Target-Gibbs-volume-change: 0.500000" << "\n";
  textrestartFile << "\n\n";
  //Write Component Data//
  WriteComponent_Restart(System, SystemComponents, textrestartFile, Box, MaxTranslation, MaxRotation);
}

static inline void create_movie_file(size_t Cycle, Atoms* System, Components SystemComponents, ForceField FF, Boxsize Box, std::vector<std::string> AtomNames, size_t SystemIndex)
{
  std::ofstream textrestartFile{};
  std::string dirname="Movies/System_" + std::to_string(SystemIndex) + "/";
  std::string fname  = dirname + "/" + "result.data";
  std::filesystem::path cwd = std::filesystem::current_path();

  std::filesystem::path directoryName = cwd /dirname;
  std::filesystem::path fileName = cwd /fname;
  std::filesystem::create_directories(directoryName);
  
  textrestartFile = std::ofstream(fileName, std::ios::out);
  WriteBox_LAMMPS(System, SystemComponents, FF, Box, textrestartFile, AtomNames);
  WriteAtoms_LAMMPS(System, SystemComponents, Box, textrestartFile, AtomNames);
}

static inline void create_Restart_file(
    size_t Cycle, Atoms *System, Components SystemComponents, ForceField FF,
    Boxsize Box, std::vector<std::string> AtomNames,
    double3 MaxTranslation, double3 MaxRotation, size_t SystemIndex)
{
  std::ofstream textrestartFile{};
  std::string dirname="Restart/System_" + std::to_string(SystemIndex) + "/";
  std::string fname  = dirname + "/" + "restartfile";
  std::filesystem::path cwd = std::filesystem::current_path();

  std::filesystem::path directoryName = cwd /dirname;
  std::filesystem::path fileName = cwd /fname;
  std::filesystem::create_directories(directoryName);
  textrestartFile = std::ofstream(fileName, std::ios::out);
  WriteCellInfo_Restart(System, SystemComponents, textrestartFile, Box, MaxTranslation, MaxRotation);
  WriteAtoms_Restart(System, SystemComponents, textrestartFile, AtomNames);
}

static inline void WriteAllData(Atoms* System, Components SystemComponents, std::ofstream& textrestartFile, std::vector<std::string> AtomNames)
{
  size_t NumComponents = SystemComponents.Total_Components;
  //size_t Atomcount=0;
  size_t Molcount=0;
  for(size_t i = SystemComponents.NumberOfFrameworks; i < NumComponents; i++)
  {
    Atoms Data = System[i];
    size_t molsize = SystemComponents.Moleculesize[i];
    //First write positions//
    textrestartFile << "x y z charge scale scaleCoul Type" << '\n';
    for(size_t j = 0; j < Data.size; j++)
      textrestartFile << " " << Molcount + Data.MolID[j] << " "
                      << j - Data.MolID[j] * molsize << " " << Data.pos[j].x()
                      << "  " << Data.pos[j].y() << "  " << Data.pos[j].z()
                      << " " << Data.charge[j] << " " << Data.scale[j] << " "
                      << Data.scaleCoul[j] << " " << Data.Type[j] << '\n';
  }
}

static inline void Write_All_Adsorbate_data(size_t Cycle, Atoms* System, Components SystemComponents, ForceField FF, Boxsize Box, std::vector<std::string> AtomNames, size_t SystemIndex)
{
  std::ofstream textrestartFile{};
  std::filesystem::path cwd = std::filesystem::current_path();

  std::string dirname="AllData/System_" + std::to_string(SystemIndex) + "/";
  std::string fname  = dirname + "/" + "AllAdsorbate.data";

  std::filesystem::path directoryName = cwd /dirname;
  std::filesystem::path fileName = cwd /fname;
  std::filesystem::create_directories(directoryName);
  textrestartFile = std::ofstream(fileName, std::ios::out);
  WriteAllData(System, SystemComponents, textrestartFile, AtomNames);
}

static inline void Write_Lambda(size_t Cycle, Components SystemComponents, size_t SystemIndex)
{
  std::ofstream textrestartFile{};
  std::filesystem::path cwd = std::filesystem::current_path();

  std::string dirname="Lambda/System_" + std::to_string(SystemIndex) + "/";
  std::string fname  = dirname + "/" + "Lambda_Histogram.data";

  std::filesystem::path directoryName = cwd /dirname;
  std::filesystem::path fileName = cwd /fname;
  std::filesystem::create_directories(directoryName);
  textrestartFile = std::ofstream(fileName, std::ios::out);
  for(size_t i = 0; i < SystemComponents.Total_Components; i++)
  {
    if(SystemComponents.hasfractionalMolecule[i])
    {
      textrestartFile << "Component " << i << ": " << SystemComponents.MoleculeName[i] << '\n';
      textrestartFile << "BIN SIZE : " << SystemComponents.Lambda[i].binsize << '\n';
      textrestartFile << "BIN WIDTH: " << SystemComponents.Lambda[i].delta << '\n';
      textrestartFile << "WL SCALING FACTOR: " << SystemComponents.Lambda[i].WangLandauScalingFactor << '\n';
      textrestartFile << "FRACTIONAL MOLECULE ID: " << SystemComponents.Lambda[i].FractionalMoleculeID << '\n';
      textrestartFile << "CURRENT BIN: " << SystemComponents.Lambda[i].currentBin << '\n';
      textrestartFile << "BINS: ";
      for(size_t j = 0; j < SystemComponents.Lambda[i].binsize; j++)
        textrestartFile << j << " ";
      textrestartFile << "\nHistogram: ";
      for(size_t j = 0; j < SystemComponents.Lambda[i].binsize; j++)
        textrestartFile << SystemComponents.Lambda[i].Histogram[j] << " ";
      textrestartFile << "\nBIAS FACTOR: ";
      for(size_t j = 0; j < SystemComponents.Lambda[i].binsize; j++)
        textrestartFile << SystemComponents.Lambda[i].biasFactor[j] << " ";
    }
  }
}

static inline void Write_TMMC(size_t Cycle, Components SystemComponents, size_t SystemIndex)
{
  std::ofstream textTMMCFile{};
  std::filesystem::path cwd = std::filesystem::current_path();

  std::string dirname="TMMC/System_" + std::to_string(SystemIndex) + "/";
  std::string fname  = dirname + "/" + "TMMC_Statistics.data";

  std::filesystem::path directoryName = cwd /dirname;
  std::filesystem::path fileName = cwd /fname;
  std::filesystem::create_directories(directoryName);
  textTMMCFile = std::ofstream(fileName, std::ios::out);
  for(size_t i = 0; i < SystemComponents.Total_Components; i++)
  {
    if(SystemComponents.Tmmc[i].DoTMMC)
    {
      textTMMCFile << "Component " << i << ": " << SystemComponents.MoleculeName[i] << '\n';
      textTMMCFile << "Min Macrostate : " << SystemComponents.Tmmc[i].MinMacrostate << '\n';
      textTMMCFile << "Max Macrostate : " << SystemComponents.Tmmc[i].MaxMacrostate << '\n';
      textTMMCFile << "Wang-Landau Factor : " << SystemComponents.Tmmc[i].WLFactor << '\n';
      textTMMCFile << "N CM[-1] CM[0] CM[1] WLBias ln_g TMBias lnpi Forward_lnpi Reverse_lnpi Histogram" << '\n';
      for(size_t j = SystemComponents.Tmmc[i].MinMacrostate; j < SystemComponents.Tmmc[i].MaxMacrostate + 1; j++)
      {
        size_t newj = j - SystemComponents.Tmmc[i].MinMacrostate;
        textTMMCFile << j << " " << SystemComponents.Tmmc[i].CMatrix[newj].x() << " " << SystemComponents.Tmmc[i].CMatrix[newj].y() << " " << SystemComponents.Tmmc[i].CMatrix[newj].z() << " " << SystemComponents.Tmmc[i].WLBias[newj] << " " << SystemComponents.Tmmc[i].ln_g[newj] << " " << SystemComponents.Tmmc[i].TMBias[newj] << " " << SystemComponents.Tmmc[i].lnpi[newj] << " " << SystemComponents.Tmmc[i].forward_lnpi[newj] << " " << SystemComponents.Tmmc[i].reverse_lnpi[newj] << " " << SystemComponents.Tmmc[i].Histogram[newj] << '\n';
      }
    }
  }
}
