#include <stdio.h>
#include <math.h>
#include <cstdlib>
#include <array>
#include <omp.h>

#include <complex>

#include "axpy.h"
#include "read_data.h"
#include "write_data.h"
#include "fxn_main.h"
int main(void)
{
  sycl::queue &que = *sycl_get_queue();

  size_t  Allocate_space_Adsorbate = 0; //Variable for recording allocate_space on the device for adsorbates //

  bool RunOneByOne = false;
  bool RunTogether = true;

  // SETUP NUMBER OF SIMULATIONS //
  size_t  NumberOfSimulations = 2;
  size_t  SelectedSim = 0; //Zhao's note: Selected simulation for testing //
  bool    RunSingleSim = false;
  bool    ReadRestart = false; //Whether we read data from restart file or not//
  read_number_of_sims_from_input(&NumberOfSimulations, &RunSingleSim);
  if(RunSingleSim)
  {
    RunOneByOne = true;
    RunTogether = false;
  }
  ////////////////////////////////////////////////////////////
  // DECLARE BASIC VARIABLES, READ FORCEFIELD and FRAMEWORK //
  ////////////////////////////////////////////////////////////
  size_t NumberOfComponents = 1; //0: Framework;
  //std::vector<Atoms*> HostSystem(NumberOfSimulations);
  std::vector<Boxsize> Box(NumberOfSimulations); //(Boxsize*) malloc(NumberOfSimulations * sizeof(Boxsize)); //Boxsize device_Box;
  PseudoAtomDefinitions PseudoAtom;
  ForceField FF; ForceField device_FF;
  ForceFieldParser(FF, PseudoAtom);
  PseudoAtomParser(FF, PseudoAtom);

  double EwaldPrecision = 1e-6;
  read_FFParams_from_input(FF, EwaldPrecision);

  // FORMULATE WIDOM VARIABLE, and STATISTICS FOR EVERY MOVE //
  WidomStruct Widom; 
  int NumberOfInitializationCycles;
  int NumberOfEquilibrationCycles;
  int NumberOfProductionCycles;
  int RANDOMSEED;
  
  bool SameFrameworkEverySimulation = true; //Use the same framework (box) setup for every simulation?//

  double PRESSURE = 0.0; double TEMPERATURE = 0.0; 

  printf("------------------GENERAL SIMULATION SETUP-------------\n");
  read_simulation_input(&Widom.UseGPUReduction, &Widom.Useflag, &FF.noCharges, &NumberOfInitializationCycles, &NumberOfEquilibrationCycles, &NumberOfProductionCycles, &Widom.NumberWidomTrials, &Widom.NumberWidomTrialsOrientations, &PRESSURE, &TEMPERATURE, &Allocate_space_Adsorbate, &ReadRestart, &RANDOMSEED, &SameFrameworkEverySimulation, NumberOfComponents);
 
  //Zhao's note: if we need to setup DNN models running just on the CPU, we set env variables before the first cuda call//
  //Then set the env variable back//
  //Test reading Tensorflow model//

 
  // UNIFIED MEMORY FOR DIFFERENT SIMULATIONS //
  //Simulations *Sims; cudaMallocManaged(&Sims, NumberOfSimulations*sizeof(Simulations));
  std::vector<Simulations> Sims(NumberOfSimulations);
  Gibbs  GibbsStatistics;
  bool   SetMaxStep = false;
  size_t MaxStepPerCycle = 1;
  read_Gibbs_Stats(GibbsStatistics, SetMaxStep, MaxStepPerCycle);
  printf("-------------------------------------------------------\n");
  // PREPARE VALUES FOR THE FORCEFIELD STRUCT //
  Prepare_ForceField(FF, device_FF, PseudoAtom);

  ////////////////////////
  // SETUP RANDOM SEEDS //
  ////////////////////////
  RandomNumber Random;
  std::srand(RANDOMSEED);
  Setup_RandomNumber(Random, 333334);

  //Physical constants for the simulation//
  Units Constants;
  //if true, then we can simulate the same framework at different T/P//
  //If false, then we can do Gibbs (where a box is empty)//
  std::vector<Components> SystemComponents;
  std::vector<WidomStruct> WidomArray;

  bool samePressure = true; bool sameTemperature = true;

  //Ewald Energy Arrays//
  //Setup Initial Ewald//
  std::vector<SystemEnergies>Energy(NumberOfSimulations);

  for(size_t a = 0; a < NumberOfSimulations; a++)
  {
    //Zhao's note: Hard-coded flag, put it in read_data.cpp later//
    Box[a].ExcludeHostGuestEwald = true;
    if(samePressure)    Box[a].Pressure = PRESSURE;
    //if(sameTemperature) 
    if(!samePressure || !sameTemperature) throw std::runtime_error("Currently not allowing the different Pressure/Temperature!");
    printf("==========================================\n");
    printf("====== Preparing Simulation box %zu ======\n", a);
    printf("==========================================\n");
    //Allocate data on the host for each simulation//
    Components TempComponents; //Temporary component variable//
    TempComponents.HostSystem  = (Atoms*) malloc(NumberOfComponents * sizeof(Atoms));
    TempComponents.ReadRestart = ReadRestart;
    TempComponents.Temperature = TEMPERATURE; 
    Move_Statistics MoveStats; 
    Initialize_Move_Statistics(MoveStats);
    /////////////////////////////////////
    // Read and process framework data //
    /////////////////////////////////////
    if(!SameFrameworkEverySimulation || a == 0)
    {
      TempComponents.NumberOfPseudoAtoms.resize(PseudoAtom.Name.size());
      std::fill(TempComponents.NumberOfPseudoAtoms.begin(), TempComponents.NumberOfPseudoAtoms.end(), 0);
      OverWriteFFTerms(TempComponents, FF, PseudoAtom);
      if(a > 0 && !SameFrameworkEverySimulation) printf("Processing %zu, new framework\n", a);
      //Read framework data from cif/poscar file//
      ReadFramework(Box[a], TempComponents.HostSystem[0], PseudoAtom, a, TempComponents);
      read_Ewald_Parameters_from_input(sqrt(FF.CutOffCoul), Box[a], EwaldPrecision);
      Update_Components_for_framework(NumberOfComponents, TempComponents, TempComponents.HostSystem);

      //Initialize Energy Averages//
      double2 tempdou ={0.0, 0.0};
      RosenbluthWeight tempRosen;
      for(size_t i = 0; i < TempComponents.Nblock; i++)
      {  
        TempComponents.EnergyAverage.push_back(tempdou);
        MoveStats.MolAverage.push_back(tempdou);
        MoveStats.Rosen.push_back(tempRosen);
      }
    }
    /////////////////////////////////////
    // Read and process adsorbate data //
    /////////////////////////////////////
    //Zhao's note: different systems will share the SAME components (for adsorbate), only read it once//
    for(size_t comp = 0; comp < TempComponents.Total_Components; comp++)
    {
      if(comp > 0) //0: framework//
      {
        //skip reading the first component, which is the framework
        read_component_values_from_simulation_input(TempComponents, MoveStats, comp-1, TempComponents.HostSystem[comp], PseudoAtom, Allocate_space_Adsorbate);
      }
      TempComponents.Moves.push_back(MoveStats);
    }
    //Int3 for the number of components, consider replace Total_Components with this int3 variable//
    TempComponents.NComponents.x() = TempComponents.Total_Components;
    TempComponents.NComponents.y() = 1;
    TempComponents.NComponents.z() = TempComponents.Total_Components - TempComponents.NComponents.y();

    SystemComponents.push_back(TempComponents);

    Setup_Box_Temperature_Pressure(Constants, SystemComponents[a], Box[a]);
    Sims[a].Box.Pressure = Box[a].Pressure; Sims[a].Box.Volume = Box[a].Volume;
    Sims[a].Box.Cubic    = Box[a].Cubic;    Sims[a].Box.ReciprocalCutOff = Box[a].ReciprocalCutOff;
    Sims[a].Box.Alpha    = Box[a].Alpha;    Sims[a].Box.Prefactor        = Box[a].Prefactor;
    Sims[a].Box.tol1     = Box[a].tol1;     Sims[a].Box.ExcludeHostGuestEwald = Box[a].ExcludeHostGuestEwald;

    Sims[a].Box.Cell = sycl::malloc_device<double>(9, que);
    Sims[a].Box.InverseCell = sycl::malloc_device<double>(9, que);
    que.memcpy(Sims[a].Box.Cell, Box[a].Cell, 9 * sizeof(double)).wait();
    que.memcpy(Sims[a].Box.InverseCell, Box[a].InverseCell, 9 * sizeof(double)).wait();

    Sims[a].Box.kmax = Box[a].kmax;
    // PREPARE VALUES FOR THE WIDOM STRUCT, DECLARE THE RESULT POINTERS IN WIDOM //
    WidomArray.push_back(Widom);
    Prepare_Widom(WidomArray[a], Box[a], Sims[a], SystemComponents[a], SystemComponents[a].HostSystem);
    ///////////////////////////////////
    // Read Restart file for Systems //
    ///////////////////////////////////
    //Zhao's note: here think about multiple simulations where we are doing indepedent simulations, each has a fractional molecule, need to be careful//
    bool AlreadyHasFractionalMolecule = false;
    Atoms device_System[NumberOfComponents];
    Sims[a].d_a = sycl::malloc_device<Atoms>(NumberOfComponents, que);
    if(RunSingleSim)
    {
      if(a == SelectedSim && ReadRestart) {RestartFileParser(Sims[a], SystemComponents[a].HostSystem, SystemComponents[a]); AlreadyHasFractionalMolecule = true;}
    }
    Copy_Atom_data_to_device(NumberOfComponents, device_System, SystemComponents[a].HostSystem);
    Prepare_TempSystem_On_Host(SystemComponents[a].TempSystem);

    que.memcpy(Sims[a].d_a, device_System, sizeof(Atoms)*NumberOfComponents).wait();

    // SET UP TEMPORARY ARRAYS //
    Setup_Temporary_Atoms_Structure(Sims[a].Old, SystemComponents[a].HostSystem);
    Setup_Temporary_Atoms_Structure(Sims[a].New, SystemComponents[a].HostSystem);


    //Prepare detailed Identity Swap statistics if there are more than 1 component//
    for(size_t i = 0; i < SystemComponents.size(); i++)
    if(SystemComponents[i].Total_Components > 2) //Including framework (component 0)
    {
      prepare_MixtureStats(SystemComponents[i]);
    }
    ///////////////////////////////////////////////////////////////////
    // Calculate & Initialize Ewald for the Initial state of the box //
    ///////////////////////////////////////////////////////////////////
    Check_Simulation_Energy(Box[a], SystemComponents[a].HostSystem, FF, device_FF, SystemComponents[a], INITIAL, a, Sims[a], true);
    //////////////////////////////////////////////////////////
    // CREATE MOLECULES IN THE BOX BEFORE SIMULAITON STARTS //
    //////////////////////////////////////////////////////////
  }

 
  //////////////////
  // RUN CYCLES //
  ////////////////

  bool RunSerialSimulations = true;
  double start = omp_get_wtime(); 
    ///////////////////////////
  // INITIALIZATION CYCLES //
  ///////////////////////////
  if(RunOneByOne)
  {
    for(size_t i = 0; i < NumberOfSimulations; i++)
    {
      if(i > 0 && RunSingleSim) continue;
      printf("Running Simulation Boxes in SERIAL, currently [%zu] box; pres: %.5f, temp: %.5f\n", i, Sims[i].Box.Pressure,  SystemComponents[i].Temperature);
      Energy[i].running_energy += Run_Simulation_ForOneBox(NumberOfInitializationCycles, SystemComponents[i], Sims[i], device_FF, Random, WidomArray[i], Energy[i].InitialVDW, INITIALIZATION, SetMaxStep, MaxStepPerCycle, Constants);
    }
  }

  //////////////////////////
  // EQUILIBRATION CYCLES //
  //////////////////////////
  if(RunOneByOne)
  {
    for(size_t i = 0; i < NumberOfSimulations; i++)
    {
      if(i > 0 && RunSingleSim) continue;
      printf("Running Simulation Boxes in SERIAL, currently [%zu] box; pres: %.5f, temp: %.5f\n", i, Sims[i].Box.Pressure,  SystemComponents[i].Temperature);
      Energy[i].running_energy += Run_Simulation_ForOneBox(NumberOfEquilibrationCycles, SystemComponents[i], Sims[i], device_FF, Random, WidomArray[i], Energy[i].InitialVDW, EQUILIBRATION, SetMaxStep, MaxStepPerCycle, Constants);
    }
  }

  ///////////////////////
  // PRODUCTION CYCLES //
  ///////////////////////
  if(RunOneByOne)
  {
    for(size_t i = 0; i < NumberOfSimulations; i++)
    {
      if(i > 0 && RunSingleSim) continue;
      printf("Running Simulation Boxes in SERIAL, currently [%zu] box; pres: %.5f, temp: %.5f\n", i, Sims[i].Box.Pressure,  SystemComponents[i].Temperature);
      Energy[i].running_energy += Run_Simulation_ForOneBox(NumberOfProductionCycles, SystemComponents[i], Sims[i], device_FF, Random, WidomArray[i], Energy[i].InitialVDW, PRODUCTION, SetMaxStep, MaxStepPerCycle, Constants);
    }
  }
  double end = omp_get_wtime();
  ///////////////////////////////////
  // PRINT FINAL ENERGIES AND TIME //
  ///////////////////////////////////
  printf("Work took %f seconds\n", end - start);

  // CALCULATE THE FINAL ENERGY (VDW + Real) //
  for(size_t i = 0; i < NumberOfSimulations; i++)
  { 
    if(RunSingleSim){if(i != SelectedSim) continue;}
    printf("======================================\n");
    printf("CHECKING FINAL ENERGY FOR SYSTEM [%zu]\n", i);
    printf("======================================\n");
    Check_Simulation_Energy(Box[i], SystemComponents[i].HostSystem, FF, device_FF, SystemComponents[i], FINAL, i, Sims[i], true);
    printf("======================================\n");
  }
  /////////////////////////////////////////////////////////
  // Check if the Ewald Diff and running Diff make sense //
  /////////////////////////////////////////////////////////
  ENERGY_SUMMARY(SystemComponents, Constants);
  printf("Random Numbers Regenerated %zu times, offset: %zu, randomsize: %zu\n", Random.Rounds, Random.offset, Random.randomsize);

  printf("DNN Feature Preparation Time: %.5f, DNN Prediction Time: %.5f\n", SystemComponents[0].DNNFeatureTime, SystemComponents[0].DNNPredictTime);
   printf("DNN GPU Time: %.5f, DNN Sort Time: %.5f, std::sort Time: %.5f, Featurization Time: %.5f\n", SystemComponents[0].DNNGPUTime, SystemComponents[0].DNNSortTime, SystemComponents[0].DNNstdsortTime, SystemComponents[0].DNNFeaturizationTime);

  //////////////////////
  // PRINT MOVIE FILE //
  //////////////////////
  GenerateRestartMovies(0, SystemComponents, Sims, FF, Box, PseudoAtom);
  /*
  if(SystemComponents[a].UseDNNforHostGuest)
  {
    Free_DNN_Model(SystemComponents[0]);
  }
  */
  return 0;
}
