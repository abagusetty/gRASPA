//#include <complex>
#include <filesystem>
#include <fstream>
#include "VDW_Coulomb.dp.hpp"
#include "maths.dp.hpp"
#include "Ewald_Energy_Functions.h"
#include "TailCorrection_Energy_Functions.h"
#include <omp.h>

//Zhao's note: There were a few variants of the same Setup_threadblock function, some of them are slightly different//
//This might be a point where debugging is needed//
void Setup_threadblock(size_t arraysize, size_t *Nblock, size_t *Nthread)
{
  //Zhao's note: Default thread should always be 64, 128, 256, 512, ...
  // This is because we are using partial sums, if arraysize is smaller than defaultthread, we need to make sure that
  //while Nthread is dividing by 2, it does not generate ODD NUMBER (for example, 5/2 = 2, then element 5 will be ignored)//
  if(arraysize == 0)  return;
  *Nthread = DEFAULTTHREAD;
  *Nblock = arraysize/(*Nthread);
  if(arraysize % (*Nthread) > 0) (*Nblock)++;
}

void VDWReal_Total_CPU(Boxsize Box, Atoms* Host_System, Atoms* System, ForceField FF, Components SystemComponents, MoveEnergy& E)
{
  sycl::queue &que = *sycl_get_queue();
  printf("****** Calculating VDW + Real Energy (CPU) ******\n");
  ///////////////////////////////////////////////////////
  //All variables passed here should be device pointers//
  ///////////////////////////////////////////////////////
  //Copy Adsorbate to host//
  for(size_t ijk=1; ijk < SystemComponents.Total_Components; ijk++) //Skip the first one(framework)
  {
    //if(Host_System[ijk].Allocate_size != System[ijk].Allocate_size)
    //{
      // if the host allocate_size is different from the device, allocate more space on the host
      Host_System[ijk].pos       = (double3*) malloc(System[ijk].Allocate_size*sizeof(double3));
      Host_System[ijk].scale     = (double*)  malloc(System[ijk].Allocate_size*sizeof(double));
      Host_System[ijk].charge    = (double*)  malloc(System[ijk].Allocate_size*sizeof(double));
      Host_System[ijk].scaleCoul = (double*)  malloc(System[ijk].Allocate_size*sizeof(double));
      Host_System[ijk].Type      = (size_t*)  malloc(System[ijk].Allocate_size*sizeof(size_t));
      Host_System[ijk].MolID     = (size_t*)  malloc(System[ijk].Allocate_size*sizeof(size_t));
      Host_System[ijk].size      = System[ijk].size; 
      Host_System[ijk].Allocate_size = System[ijk].Allocate_size;
    //}
  
    //if(Host_System[ijk].Allocate_size = System[ijk].Allocate_size) //means there is no more space allocated on the device than host, otherwise, allocate more on host
    //{
      que.memcpy(Host_System[ijk].pos,    System[ijk].pos, sizeof(double3) * System[ijk].Allocate_size);
      que.memcpy(Host_System[ijk].scale,  System[ijk].scale, sizeof(double) * System[ijk].Allocate_size);
      que.memcpy(Host_System[ijk].charge, System[ijk].charge, sizeof(double) * System[ijk].Allocate_size);
      que.memcpy(Host_System[ijk].scaleCoul, System[ijk].scaleCoul, sizeof(double) * System[ijk].Allocate_size);
      que.memcpy(Host_System[ijk].Type,  System[ijk].Type, sizeof(size_t) * System[ijk].Allocate_size);
      que.memcpy(Host_System[ijk].MolID, System[ijk].MolID, sizeof(size_t) * System[ijk].Allocate_size);
      que.wait();
      Host_System[ijk].size = System[ijk].size;
      //printf("CPU CHECK: comp: %zu, Host Allocate_size: %zu, Allocate_size: %zu\n", ijk, Host_System[ijk].Allocate_size, System[ijk].Allocate_size);
    //}
  }
  //Write to a file for checking//
  std::ofstream textrestartFile{};
  std::string dirname="FirstBead/";
  std::string fname  = dirname + "/" + "Energy.data";
  std::filesystem::path cwd = std::filesystem::current_path();

  std::filesystem::path directoryName = cwd /dirname;
  std::filesystem::path fileName = cwd /fname;
  std::filesystem::create_directories(directoryName);

  textrestartFile = std::ofstream(fileName, std::ios::out);
  textrestartFile << "PosA PosB TypeA TypeB E" <<"\n";

  double Total_HG_VDW = 0.0; double Total_HG_Real = 0.0;
  double Total_GG_VDW = 0.0; double Total_GG_Real = 0.0;
  size_t count = 0; size_t cutoff_count=0;
  double VDW_energy   = 0.0; double Coul_energy = 0.0;
  //FOR DEBUGGING ENERGY//
  MoveEnergy FirstBeadE; MoveEnergy ChainE; size_t FBHGPairs=0; size_t FBGGPairs = 0;
                                            size_t CHGPairs=0;  size_t CGGPairs = 0;
  size_t selectedComp= 4;
  size_t selectedMol = 44;
  //size_t selectedMol = SystemComponents.NumberOfMolecule_for_Component[selectedComp] - 1;
  std::vector<double> FBES; std::vector<double> CHAINES;
  for(size_t compi=0; compi < SystemComponents.Total_Components; compi++) 
  {
    const Atoms Component=Host_System[compi];
    for(size_t i=0; i<Component.size; i++)
    {
      //printf("comp: %zu, i: %zu, x: %.10f\n", compi, i, Component.pos[i].x());
      const double scaleA = Component.scale[i];
      const double chargeA = Component.charge[i];
      const double scalingCoulombA = Component.scaleCoul[i];
      const size_t typeA = Component.Type[i];
      const size_t MoleculeID = Component.MolID[i];
      for(size_t compj=0; compj < SystemComponents.Total_Components; compj++)
      {
        //if(SystemComponents.UseDNNforHostGuest && (compi == 0 || compj == 0)) continue; //Ignore host-guest if DNN is used for host-guest//
        if(!((compi == 0) && (compj == 0))) //ignore fraemwrok-framework interaction
        {
          const Atoms Componentj=Host_System[compj];
          for(size_t j=0; j<Componentj.size; j++)
          {
            const double scaleB = Componentj.scale[j];
            const double chargeB = Componentj.charge[j];
            const double scalingCoulombB = Componentj.scaleCoul[j];
            const size_t typeB = Componentj.Type[j];
            const size_t MoleculeIDB = Componentj.MolID[j];
            if(!((MoleculeID == MoleculeIDB) &&(compi == compj)))
            {
              count++;
              //double3 posvec = Component.pos[i] - Componentj.pos[j];
              double3 pos1 = static_cast<double3>(Component.pos[i]);
              double3 pos2 = static_cast<double3>(Componentj.pos[j]);
              double3 posvec = pos1 - pos2;
              PBC(posvec, Box.Cell, Box.InverseCell, Box.Cubic);
              const double rr_dot = dot(posvec, posvec);
              //printf("i: %zu, j: %zu, rr_dot: %.10f\n", i,j,rr_dot);
              //if((compi > 0) && (compj > 0)) printf("CHECK_DIST: Compi: %zu Mol[%zu], compj: %zu Mol[%zu], rr_dot: %.5f\n", compi, MoleculeID, compj, MoleculeIDB, rr_dot);
              double result[2] = {0.0, 0.0};
              if(rr_dot < FF.CutOffVDW)
              {
                cutoff_count++;
                const double scaling = scaleA * scaleB;
                const size_t row = typeA*FF.size+typeB;
                const double FFarg[4] = {FF.epsilon[row], FF.sigma[row], FF.z[row], FF.shift[row]};
                VDW(FFarg, rr_dot, scaling, result);
                if((compi == 0) || (compj == 0)) 
                {
                  Total_HG_VDW += 0.5*result[0];
                }
                else
                {
                  Total_GG_VDW += 0.5*result[0];
                }
                //if((compi > 0) && (compj > 0)) printf("Compi: %zu Mol[%zu], compj: %zu Mol[%zu], GG_E: %.5f\n", compi, MoleculeID, compj, MoleculeIDB, result[0]);
                VDW_energy   += 0.5*result[0];
                if(std::abs(result[0]) > 10000) printf("Very High Energy (VDW), comps: %zu, %zu, MolID: %zu %zu, Atom: %zu %zu, E: %.5f\n", compi, compj, MoleculeID, MoleculeIDB, i, j, result[0]);
                //DEBUG//
                if(MoleculeID == selectedMol && (compi == selectedComp)) 
                {
                  if(i%Component.Molsize == 0)//First Bead//
                  {
                    if(compj == 0){FirstBeadE.HGVDW += result[0]; FBHGPairs ++;}
                    else
                    {
                      //printf("FB_GG: posi: %zu, typeA: %zu, comp: %zu, ENERGY: %.5f\n", j, typeB, compj, result[0]);
                      FirstBeadE.GGVDW += result[0]; FBGGPairs ++;
                    }
                  }
                  else
                  { if(compj == 0){ChainE.HGVDW += result[0]; CHGPairs ++;}
                    else
                    {
                      //printf("Chain_GG: posi: %zu, typeA: %zu, comp: %zu, ENERGY: %.5f\n", j, typeB, compj, result[0]);
                      ChainE.GGVDW += result[0]; CGGPairs ++;
                    } 
                  }
                }
              }
              if (!FF.noCharges && rr_dot < FF.CutOffCoul)
              {
                const double r = sqrt(rr_dot);
                const double scalingCoul = scalingCoulombA * scalingCoulombB;
                double resultCoul[2] = {0.0, 0.0};
                CoulombReal(chargeA, chargeB, r, scalingCoul, resultCoul, Box.Prefactor, Box.Alpha);
                if((compi == 0) || (compj == 0))
                {
                  Total_HG_Real += 0.5*resultCoul[0];
                }
                else
                {
                  Total_GG_Real += 0.5*resultCoul[0];
                }
                Coul_energy  += 0.5*resultCoul[0];
                if(std::abs(result[0]) > 10000) printf("Very High Energy (Coul), comps: %zu, %zu, MolID: %zu %zu, Atom: %zu %zu, E: %.5f\n", compi, compj, MoleculeID, MoleculeIDB, i, j, resultCoul[0]);
                //DEBUG//
                if(MoleculeID == selectedMol && (compi == selectedComp))
                {
                  if(i%Component.Molsize == 0)//First Bead//
                  {
                    if(compj == 0){FirstBeadE.HGReal += resultCoul[0]; FBHGPairs ++;}
                    else
                    {
                      FirstBeadE.GGReal += resultCoul[0]; FBGGPairs ++;
                    }
                  }
                  else
                  { if(compj == 0){ChainE.HGReal += resultCoul[0]; CHGPairs ++;}
                    else{ChainE.GGReal += resultCoul[0]; CGGPairs ++;}
                  }
                }
              }
            }
          }
        }
      }
    }  
  }
  //printf("%zu interactions, within cutoff: %zu, energy: %.10f\n", count, Total_energy, cutoff_count);
  E.HGVDW = Total_HG_VDW;  E.GGVDW = Total_GG_VDW;
  E.HGReal= Total_HG_Real; E.GGReal= Total_GG_Real;
}

////////////////////////////// GPU CODE //////////////////////////

void one_thread_GPU_test(Boxsize Box, Atoms* System, ForceField FF, double* xxx)
{
  bool DEBUG=false;
  //Zhao's note: added temp_xxx values for checking individual energy for each molecule//
  double temp_energy = 0.0; double temp_firstbead = 0.0; double temp_chain = 0.0; int temp_count = -1;
  double Total_energy = 0.0; size_t count = 0; size_t cutoff_count=0;
  double VDW_energy = 0.0; double Coul_energy = 0.0;
  for(size_t compi=0; compi < 2; compi++) //Zhao's note: hard coded component, change
  {
    const Atoms Component=System[compi];
    //printf("GPU CHECK: Comp: %lu, Comp size: %lu, Allocate size: %lu\n", compi, Component.size, Component.Allocate_size);
    for(size_t i=0; i<Component.size; i++)
    {
      //printf("comp: %lu, i: %lu, x: %.10f\n", compi, i, Component.pos[i].x);
      const double scaleA = Component.scale[i];
      const double chargeA = Component.charge[i];
      const double scalingCoulombA = Component.scaleCoul[i];
      const size_t typeA = Component.Type[i];
      const size_t MoleculeID = Component.MolID[i];
      if(DEBUG){if(MoleculeID == 5) temp_count++;} //For testing individual molecule energy//
      for(size_t compj=0; compj < 2; compj++) //Zhao's note: hard coded component, change
      {
        if(!((compi == 0) && (compj == 0))) //ignore fraemwrok-framework interaction
        {
          const Atoms Componentj=System[compj];
          for(size_t j=0; j<Componentj.size; j++)
          {
            const double scaleB = Componentj.scale[j];
            const double chargeB = Componentj.charge[j];
            const double scalingCoulombB = Componentj.scaleCoul[j];
            const size_t typeB = Componentj.Type[j];
            const size_t MoleculeIDB = Componentj.MolID[j];
            if(!((MoleculeID == MoleculeIDB) &&(compi == compj)))
            {
              count++;
              double3 posvec = Component.pos[i] - Componentj.pos[j];
              PBC(posvec, Box.Cell, Box.InverseCell, Box.Cubic);

              const double rr_dot = dot(posvec, posvec);
              double result[2] = {0.0, 0.0};
              if(rr_dot < FF.CutOffVDW)
              {
                cutoff_count++;
                const double scaling = scaleA * scaleB;
                const size_t row = typeA*FF.size+typeB;
                const double FFarg[4] = {FF.epsilon[row], FF.sigma[row], FF.z[row], FF.shift[row]};
                VDW(FFarg, rr_dot, scaling, result);
                Total_energy += 0.5*result[0];
                VDW_energy   += 0.5*result[0];
                if(DEBUG){if(MoleculeID == 5)
                { 
                  temp_energy += result[0];
                  if(temp_count == 0){temp_firstbead += result[0];}
                  else {temp_chain += result[0];}
                } 
              }}
              //  printf("SPECIEL CHECK: compi: %lu, i: %lu, compj: %lu, j: %lu, pos: %.5f, %.5f, %.5f, rr_dot: %.10f, energy: %.10f\n", compi,i,compj,j,Component.pos[i].x, Component.pos[i].y, Component.pos[i].z, rr_dot, result[0]);
              if (!FF.noCharges && rr_dot < FF.CutOffCoul)
              {
                const double r = sqrt(rr_dot);
                const double scalingCoul = scalingCoulombA * scalingCoulombB;
                double resultCoul[2] = {0.0, 0.0};
                CoulombReal(chargeA, chargeB, r, scalingCoul, resultCoul, Box.Prefactor, Box.Alpha);
                Total_energy += 0.5*resultCoul[0]; //prefactor merged in the CoulombReal function
                Coul_energy  += 0.5*resultCoul[0];
              }
            }
          }
        }
      }
    }
  }
  if(DEBUG) printf("For Molecule 5, energy: %.10f, firstbead: %.10f, chain: %.10f\n", temp_energy, temp_firstbead, temp_chain);
  xxx[0] = Total_energy;
  printf("GPU (one Thread) Total Energy: %.5f, VDW Energy: %.5f, Coulomb Energy: %.5f\n", Total_energy, VDW_energy, Coul_energy);
  //printf("xxx: %.10f\n", Total_energy);
}

__attribute__((always_inline))
void Calculate_Single_Body_Energy_SEPARATE_HostGuest_VDWReal(Boxsize Box, Atoms* System, Atoms Old, Atoms New, ForceField FF, double* BlockEnergy, size_t ComponentID, size_t totalAtoms, size_t chainsize, bool* flag, size_t HG_Nblock, size_t GG_Nblock, bool Do_New, bool Do_Old, int3 NComps, const sycl::nd_item<1> &item, sycl::decorated_local_ptr<double> sdata)
{
  ///////////////////////////////////////////////////////
  //All variables passed here should be device pointers//
  ///////////////////////////////////////////////////////
  //auto sdata = (double *)dpct_local;
  size_t blockIdx  = item.get_group(0);
  size_t blockDim  = item.get_local_range(0);
  size_t threadIdx = item.get_local_id(0);
  int cache_id = threadIdx;
  size_t total_ij = blockIdx * blockDim + threadIdx;

  size_t ij_within_block = total_ij - blockIdx * blockDim;

  sdata[ij_within_block] = 0.0; sdata[ij_within_block + blockDim] = 0.0;
  //Initialize Blocky and BlockdUdlambda//
  BlockEnergy[blockIdx] = 0.0; BlockEnergy[blockIdx + HG_Nblock + GG_Nblock] = 0.0; 
  //BlockdUdlambda[blockIdx] = 0.0;

  bool Blockflag = false;

  const size_t NTotalComp = NComps.x(); //Zhao's note: need to change here for multicomponent (Nguest comp > 1)
  const size_t NHostComp  = NComps.y();
  //const size_t NGuestComp = NComps.z;

  size_t ij = total_ij;
  if(blockIdx >= HG_Nblock)
  {
    ij -= HG_Nblock * blockDim; //It belongs to the Guest-Guest Interaction//
  }
  // Manually fusing/collapsing the loop //
  size_t i = ij/chainsize;
  size_t j = ij%chainsize; //+ ij/totalAtoms; // position in Old and New

  //printf("ij: %lu, i: %lu, j: %lu, trial: %lu, totalAtoms: %lu, totalthreads: %lu\n", ij,i,j,k,totalAtoms, totalthreads);
  size_t posi = i; size_t totalsize= 0;
  size_t startComp = 0; size_t endComp = 0;
  if(blockIdx < HG_Nblock) //Host-Guest Interaction//
  {
    startComp = 0; endComp = NHostComp;
  }
  else //Guest-Guest Interaction//
  {
    startComp = NHostComp; endComp = NTotalComp;
  }
  size_t comp = startComp;
  for(size_t ijk = startComp; ijk < endComp; ijk++)
  {
    totalsize += System[ijk].size;
    if(posi >= System[ijk].size)
    {
      comp++;
      posi -= System[ijk].size;
    }
    else
    {break;}
  }
  //Also need to check the range of the components//
  //if host-guest, then comp need to fall into the framework components//
  bool CompCheck = false;
  if(blockIdx < HG_Nblock)
  {
    if(comp < NHostComp) CompCheck = true;
  }
  else //Guest-Guest interaction//
  {
    if((comp >= NHostComp) && (comp < NTotalComp)) CompCheck = true;
  }


  if(CompCheck)
  if(posi < System[comp].size)
  {
  const Atoms Component=System[comp];
  const double scaleA = Component.scale[posi];
  const double chargeA = Component.charge[posi];
  const double scalingCoulombA = Component.scaleCoul[posi];
  const size_t typeA = Component.Type[posi];
  const size_t MoleculeID = Component.MolID[posi];
  double2 tempy = {0.0, 0.0}; double tempdU = 0.0;
  if(!((MoleculeID == New.MolID[0]) &&(comp == ComponentID)) && Do_New) //ComponentID: Component ID for the molecule being translated
  {
    ///////////
    //  NEW  //
    ///////////
    double3 posvec = Component.pos[posi] - New.pos[j];
    PBC(posvec, Box.Cell, Box.InverseCell, Box.Cubic);
    double rr_dot = dot(posvec, posvec);
    double result[2] = {0.0, 0.0};
    if(rr_dot < FF.CutOffVDW)
    {
      const size_t typeB = New.Type[j];
      const double scaleB = New.scale[j];
      const double scaling = scaleA * scaleB;
      const size_t row = typeA*FF.size+typeB;
      const double FFarg[4] = {FF.epsilon[row], FF.sigma[row], FF.z[row], FF.shift[row]};
      VDW(FFarg, rr_dot, scaling, result);
      if(result[0] > FF.OverlapCriteria) { Blockflag = true; flag[0] = true; }
      if(rr_dot < 0.01)                  { Blockflag = true; flag[0] = true; } //DistanceCheck//
      tempy.x()+= result[0];
      tempdU   += result[1];
    }

    if (!FF.noCharges && rr_dot < FF.CutOffCoul)
    {
      const double chargeB = New.charge[j];
      const double scalingCoulombB = New.scaleCoul[j];
      const double r = sqrt(rr_dot);
      const double scalingCoul = scalingCoulombA * scalingCoulombB;
      CoulombReal(chargeA, chargeB, r, scalingCoul, result, Box.Prefactor, Box.Alpha);
      tempy.y() += result[0]; //prefactor merged in the CoulombReal function
    }
  }
  if(!((MoleculeID == Old.MolID[0]) &&(comp == ComponentID)) && Do_Old) //ComponentID: Component ID for the molecule being translated
  {
    ///////////
    //  OLD  //
    ///////////
    double3 posvec = Component.pos[posi] - Old.pos[j];
    PBC(posvec, Box.Cell, Box.InverseCell, Box.Cubic);
    double rr_dot = dot(posvec, posvec);
    double result[2] = {0.0, 0.0};
    if(rr_dot < FF.CutOffVDW)
    {
      const size_t typeB = Old.Type[j];
      const double scaleB = Old.scale[j];
      const double scaling = scaleA * scaleB;
      const size_t row = typeA*FF.size+typeB;
      const double FFarg[4] = {FF.epsilon[row], FF.sigma[row], FF.z[row], FF.shift[row]};
      VDW(FFarg, rr_dot, scaling, result);
      tempy.x()-= result[0];
      tempdU   -= result[1];
    }
    if (!FF.noCharges && rr_dot < FF.CutOffCoul)
    {
      const double chargeB = Old.charge[j];
      const double scalingCoulombB = Old.scaleCoul[j];
      const double r = sqrt(rr_dot);
      const double scalingCoul = scalingCoulombA * scalingCoulombB;
      CoulombReal(chargeA, chargeB, r, scalingCoul, result, Box.Prefactor, Box.Alpha);
      tempy.y() -= result[0]; //prefactor merged in the CoulombReal function
    }
    //printf("BlockID: %i, threadID: %i, VDW: %.5f, Real: %.5f\n", blockIdx.x, threadIdx.x, tempy.x, tempy.y);
  }
  sdata[ij_within_block] = tempy.x(); //sdata[ij_within_block].y = tempdU;
  sdata[ij_within_block + blockDim] = tempy.y();
  }
  item.barrier(sycl::access::fence_space::local_space);
  //Partial block sum//
  if(!Blockflag)
  {
    int i=blockDim / 2;
    while(i != 0)
    {
      if(cache_id < i)
      {
        sdata[cache_id] += sdata[cache_id + i]; 
        sdata[cache_id + blockDim] += sdata[cache_id + i + blockDim];
      }
      item.barrier();
      i /= 2;
    }
    if(cache_id == 0)
    {
      BlockEnergy[blockIdx]   = sdata[0]; //BlockdUdlambda[blockIdx] = sdata[0].y();
      BlockEnergy[blockIdx + HG_Nblock + GG_Nblock] = sdata[blockDim]; //Shift it//
    }
  }
}

__attribute__((always_inline))
void Calculate_Multiple_Trial_Energy_SEPARATE_HostGuest_VDWReal(Boxsize Box, Atoms* System, Atoms NewMol, ForceField FF, double* Blocksum, size_t ComponentID, size_t totalAtoms, bool* flag, size_t totalthreads, size_t chainsize, size_t NblockForTrial, size_t HG_Nblock, int3 NComps, int2* ExcludeList, const sycl::nd_item<1> &item, sycl::decorated_local_ptr<double> sdata)
{
  //Dividing Nblocks into Nblocks for host-guest and for guest-guest//
  //NblockForTrial = HG_Nblock + GG_Nblock;
  //Separating VDW + Real, if a trial needs 5 block (cuda blocks)
  //In the Blocksum array, it will use the first 5 doubles for VDW, the later 5 doubles for Real//
  //This is slightly confusing, don't be fooled, elements in Blocksum != cuda blocks!!!! //
  ///////////////////////////////////////////////////////
  //All variables passed here should be device pointers//
  ///////////////////////////////////////////////////////

  const size_t blockIdx  = item.get_group(0);
  const size_t blockDim  = item.get_local_range(0);
  const size_t threadIdx = item.get_local_id(0);

  int cache_id = threadIdx;
  size_t trial = blockIdx/NblockForTrial;
  size_t total_ij = blockIdx * blockDim + threadIdx;
  size_t ij = total_ij - trial * NblockForTrial * blockDim;

  size_t trial_blockID = blockIdx - NblockForTrial * trial;

  sdata[threadIdx] = 0.0; sdata[threadIdx + blockDim] = 0.0;
  //Initialize Blocksum//
  size_t StoreId = blockIdx + trial * NblockForTrial;
  if(cache_id == 0) { Blocksum[StoreId] = 0.0; Blocksum[StoreId + NblockForTrial] = 0.0; }

  //__shared__ bool Blockflag = false;

  if(trial_blockID >= HG_Nblock)
    ij -= HG_Nblock * blockDim; //It belongs to the Guest-Guest Interaction//
  // Manually fusing/collapsing the loop //
  size_t i = ij/chainsize; //ij is the thread id within the trial, just divide by chainsize to get the true i (atom id)
  size_t j = trial*chainsize + ij%chainsize; // position in NewMol
  //printf("ij: %lu, i: %lu, j: %lu, trial: %lu, totalAtoms: %lu, totalthreads: %lu\n", ij,i,j,k,totalAtoms, totalthreads);
  const size_t NTotalComp = NComps.x(); 
  const size_t NHostComp  = NComps.y();
  //const size_t NGuestComp = NComps.z();
 
  size_t posi = i; size_t totalsize= 0;
  size_t startComp = 0; size_t endComp = 0;

  //if posi exceeds the number of atoms in their components, stop//
  size_t NFrameworkAtoms = 0; size_t NAdsorbateAtoms = 0;
  for(size_t ijk = 0;         ijk < NHostComp;  ijk++) NFrameworkAtoms += System[ijk].size;
  for(size_t ijk = NHostComp; ijk < NTotalComp; ijk++) NAdsorbateAtoms += System[ijk].size;
  //Skip calculation if the block is for Host-Guest, and the posi is greater than or equal to N_FrameworkAtoms//
  //It is equal to 
  //if((posi >= NFrameworkAtoms) && (trial_blockID < HG_Nblock)) continue;
  if((posi < NFrameworkAtoms) || !(trial_blockID < HG_Nblock))
  {
  if(trial_blockID < HG_Nblock) //Host-Guest Interaction//
  {
    startComp = 0; endComp = NHostComp;
  }
  else //Guest-Guest Interaction//
  {
    startComp = NHostComp; endComp = NTotalComp;
  }
  size_t comp = startComp;
  //printf("%lu, %lu\n", System[0].size, System[1].size);

  for(size_t ijk = startComp; ijk < endComp; ijk++)
  {
    totalsize += System[ijk].size;
    if(posi >= System[ijk].size)
    {
      comp++;
      posi -= System[ijk].size;
    }
    else
    {break;}
  }

  //Also need to check the range of the components//
  //if host-guest, then comp need to fall into the framework components//
  bool CompCheck = false;
  if(trial_blockID < HG_Nblock)
  {
    if(comp < NHostComp) CompCheck = true;
  }
  else //Guest-Guest interaction//
  {
    if((comp >= NHostComp) && (comp < NTotalComp)) CompCheck = true;
  }

  if(CompCheck)
  if(posi < System[comp].size)
  {
    const Atoms Component = System[comp];
    const double scaleA = Component.scale[posi];
    const double chargeA = Component.charge[posi];
    const double scalingCoulombA = Component.scaleCoul[posi];
    const size_t typeA = Component.Type[posi];
    const size_t MoleculeID = System[comp].MolID[posi];

    //printf("i: %lu, posi: %lu, size1: %lu, size2: %lu\n", i, posi, System[0].size, System[1].size);

    double2 tempy = {0.0, 0.0};
    double ConsiderThisMolecule = true;
    //Checking the first element of the ExcludeList to Ignore specific component/molecule//
    if(comp == ExcludeList[0].x() && MoleculeID == ExcludeList[0].y()) ConsiderThisMolecule = false;
    if((MoleculeID == NewMol.MolID[0]) &&(comp == ComponentID))    ConsiderThisMolecule = false;

    if(ConsiderThisMolecule)
    {
      double3 posvec = Component.pos[posi] - NewMol.pos[j];
      // double3 pos1 = static_cast<double3>(Component.pos[posi] );
      // double3 pos2 =  static_cast<double3>( NewMol.pos[j]);
      // double3 posvec = pos1 - pos2;
      PBC(posvec, Box.Cell, Box.InverseCell, Box.Cubic);
      const double rr_dot = dot(posvec, posvec);
      if(rr_dot < FF.CutOffVDW)
      {
        double result[2] = {0.0, 0.0};
        const size_t typeB = NewMol.Type[j];
        const double scaleB = NewMol.scale[j];
        const double scaling = scaleA * scaleB;
        const size_t row = typeA*FF.size+typeB;
        //printf("typeA: %lu, typeB: %lu, FF.size: %lu, row: %lu\n", typeA, typeB, FF.size, row);
        const double FFarg[4] = {FF.epsilon[row], FF.sigma[row], FF.z[row], FF.shift[row]};
        VDW(FFarg, rr_dot, scaling, result); 
        if(result[0] > FF.OverlapCriteria){ flag[trial]=true; }
        if(rr_dot < 0.01) { flag[trial]=true; } //DistanceCheck//
        tempy.x() += result[0];
        //DEBUG//
        /*
        if(CYCLE == 28981 && comp != 0 && trial == 0)
        {
          printf("GG PAIR: total_ij: %lu, ij: %lu, posi: %lu, typeA: %lu, comp: %lu, ENERGY: %.5f\n", total_ij, ij, posi, typeA, comp, result[0]);
        }
        */
      }

      if (FF.VDWRealBias && !FF.noCharges && rr_dot < FF.CutOffCoul)
      {
        const double chargeB = NewMol.charge[j];
        const double scalingCoulombB = NewMol.scaleCoul[j];
        const double r = sqrt(rr_dot);
        const double scalingCoul = scalingCoulombA * scalingCoulombB;
        double resultCoul[2] = {0.0, 0.0};
        CoulombReal(chargeA, chargeB, r, scalingCoul, resultCoul, Box.Prefactor, Box.Alpha);
        tempy.y() += resultCoul[0]; //prefactor merged in the CoulombReal function
      }
    }
    //if((trial_blockID >= HG_Nblock) && (tempy > 1e-10))
    //  printf("Guest-Guest, comp = %lu, trial: %lu, posi = %lu, data: %.5f\n", comp, j, posi, tempy);
    sdata[threadIdx] = tempy.x(); sdata[threadIdx + blockDim] = tempy.y();
  }
  }
  item.barrier(sycl::access::fence_space::local_space);
  //Partial block sum//
  //if(!Blockflag)
  //{
    int reductionsize=blockDim / 2;
    while(reductionsize != 0) 
    {
      if(cache_id < reductionsize) 
      {
        sdata[cache_id] += sdata[cache_id + reductionsize];
        sdata[cache_id + blockDim] += sdata[cache_id + reductionsize + blockDim];
      }
      item.barrier();
      reductionsize /= 2;
    }
    if(cache_id == 0) 
    {
     Blocksum[StoreId] = sdata[0];
     Blocksum[StoreId + NblockForTrial] = sdata[blockDim];
     //if(trial_blockID >= HG_Nblock) 
    //printf("GG, trial: %lu, BlockID: %lu, data: %.5f\n", trial, blockIdx, sdata[0]);
    }
  //}
}


// ABB: THis needs edits to look similar like CUDA's version
__attribute__((always_inline))
void VDWCoulEnergy_Total(Boxsize Box, Atoms ComponentA, Atoms ComponentB, ForceField FF, bool* flag, double2& tempy, size_t posA, size_t posB, size_t compA, size_t compB, bool UseOffset)
//void VDWCoulEnergy_Total(Boxsize Box, Atoms ComponentA, Atoms ComponentB, size_t Aij, size_t Bij, ForceField FF, bool* flag, bool& Blockflag, double& tempy, size_t NA, size_t NB, bool UseOffset)
{
  //compA may go beyond the bound, if this happens, check if compA + 1 goes beyond endcompA//
  size_t OffsetA         = 0;
  if(UseOffset) OffsetA  = ComponentA.Allocate_size / 2; //Read the positions shifted to the later half of the storage//
  const double scaleA          = ComponentA.scale[posA];
  const double chargeA         = ComponentA.charge[posA];
  const double scalingCoulombA = ComponentA.scaleCoul[posA];
  const size_t typeA           = ComponentA.Type[posA];

  const double3 PosA = ComponentA.pos[posA + OffsetA];

  size_t OffsetB         = 0;
  if(UseOffset) OffsetB  = ComponentB.Allocate_size / 2; //Read the positions shifted to the later half of the storage//
  const double scaleB          = ComponentB.scale[posB];
  const double chargeB         = ComponentB.charge[posB];
  const double scalingCoulombB = ComponentB.scaleCoul[posB];
  const size_t typeB           = ComponentB.Type[posB];
  const double3 PosB = ComponentB.pos[posB + OffsetB];

  //printf("thread: %lu, i:%lu, j:%lu, comp: %lu, posi: %lu\n", ij,i,j,comp, posi);
  double3 posvec = PosA - PosB;
  PBC(posvec, Box.Cell, Box.InverseCell, Box.Cubic);
  const double rr_dot = dot(posvec, posvec);
  if(rr_dot < FF.CutOffVDW)
  {
    double result[2] = {0.0, 0.0};
    const double scaling = scaleA * scaleB;
    const size_t row = typeA*FF.size+typeB;
    const double FFarg[4] = {FF.epsilon[row], FF.sigma[row], FF.z[row], FF.shift[row]};
    VDW(FFarg, rr_dot, scaling, result);
    if(result[0] > FF.OverlapCriteria){ flag[0]=true;}
    if(rr_dot < 0.01) { flag[0]=true; } //DistanceCheck//
    //if(result[0] > FF.OverlapCriteria || rr_dot < 0.01) printf("OVERLAP IN KERNEL!\n");
    tempy.x() += result[0];
  }
  //Coulombic (REAL)//
  if (!FF.noCharges && rr_dot < FF.CutOffCoul)
  {
    const double r = sqrt(rr_dot);
    const double scalingCoul = scalingCoulombA * scalingCoulombB;
    double resultCoul[2] = {0.0, 0.0};
    CoulombReal(chargeA, chargeB, r, scalingCoul, resultCoul, Box.Prefactor, Box.Alpha);
    tempy.y() += resultCoul[0]; //prefactor merged in the CoulombReal function
  }

  
  // for(size_t i = 0; i < NA; i++)
  // {
  //         size_t OffsetA         = 0;
  //         size_t posi            = i + Aij;
  //         if(UseOffset) OffsetA  = ComponentA.Allocate_size / 2; //Read the positions shifted to the later half of the storage//
  //   //Zhao's note: add protection here//
  //   if(posi >= ComponentA.size) continue;
  //   const double scaleA          = ComponentA.scale[posi];
  //   const double chargeA         = ComponentA.charge[posi];
  //   const double scalingCoulombA = ComponentA.scaleCoul[posi];
  //   const size_t typeA           = ComponentA.Type[posi];

  //   const double3 PosA = ComponentA.pos[posi + OffsetA];
  //   for(size_t j = 0; j < NB; j++)
  //   {
  //           size_t OffsetB         = 0;
  //           size_t posj            = j + Bij;
  //           if(UseOffset) OffsetB  = ComponentB.Allocate_size / 2; //Read the positions shifted to the later half of the storage//
  //     //Zhao's note: add protection here//
  //     //if(posj >= ComponentB.size) continue;
  //     const double scaleB          = ComponentB.scale[posj];
  //     const double chargeB         = ComponentB.charge[posj];
  //     const double scalingCoulombB = ComponentB.scaleCoul[posj];
  //     const size_t typeB           = ComponentB.Type[posj];
  //     //if(j == 6) printf("PAIR CHECK: i: %lu, j: %lu, MoleculeID: %lu, NewMol.MolID: %lu\n", i,j,MoleculeID, NewMol.MolID[0]);
  //     const double3 PosB = ComponentB.pos[posj + OffsetB];
  //     double3 posvec = static_cast<double3>(PosA) - static_cast<double3>(PosB);
  //     //printf("thread: %lu, i:%lu, j:%lu, comp: %lu, posi: %lu\n", ij,i,j,comp, posi);
  //     PBC(posvec, Box.Cell, Box.InverseCell, Box.Cubic);
  //     const double rr_dot = dot(posvec, posvec);
  //     if(rr_dot < FF.CutOffVDW)
  //     {
  //       double result[2] = {0.0, 0.0};
  //       const double scaling = scaleA * scaleB;
  //       const size_t row = typeA*FF.size+typeB;
  //       //printf("typeA: %lu, typeB: %lu, FF.size: %lu, row: %lu\n", typeA, typeB, FF.size, row);
  //       const double FFarg[4] = {FF.epsilon[row], FF.sigma[row], FF.z[row], FF.shift[row]};
  //       VDW(FFarg, rr_dot, scaling, result);
  //       if(result[0] > FF.OverlapCriteria){ flag[0]=true; Blockflag = true;}
  //       if(rr_dot < 0.01) { flag[0]=true; Blockflag = true; } //DistanceCheck//
  //       if(result[0] > FF.OverlapCriteria || rr_dot < 0.01) printf("OVERLAP IN KERNEL!\n");
  //       tempy += result[0];
  //     }
  //     //Coulombic (REAL)//
  //     if (!FF.noCharges && rr_dot < FF.CutOffCoul)
  //     {
  //       const double r = sqrt(rr_dot);
  //       const double scalingCoul = scalingCoulombA * scalingCoulombB;
  //       double resultCoul[2] = {0.0, 0.0};
  //       CoulombReal(chargeA, chargeB, r, scalingCoul, resultCoul, Box.Prefactor, Box.Alpha);
  //       tempy += resultCoul[0]; //prefactor merged in the CoulombReal function
  //     }
  //   }
  // }
}

// void TotalVDWCoul(Boxsize Box, Atoms* System, ForceField FF, double* Blocksum, bool* flag, size_t totalthreads, size_t Host_threads, size_t NAds, size_t NFrameworkAtomsPerThread, bool HostHost, bool UseOffset)
// {
//   extern double sdata[]; //shared memory for partial sum//
//   int cache_id = threadIdx.x; 
//   size_t total_ij = blockIdx.x * blockDim.x + threadIdx.x;
 
//   size_t ij_within_block = total_ij - blockIdx.x * blockDim.x;
//   sdata[ij_within_block] = 0.0;

//   //Initialize Blocksum//
//   if(cache_id == 0) Blocksum[blockIdx.x] = 0.0;
//   bool Blockflag = false;
 
//   if(total_ij < totalthreads)
//   {
//     size_t totalsize = 0;
//     const size_t NumberComp = 2; //Zhao's note: need to change here for multicomponent
//     //Aij and Bij indicate the starting positions for the objects in the pairwise interaction//
//     size_t Aij   = 0; size_t Bij   = 0;
//     size_t MolA  = 0; size_t MolB  = 0;
//     size_t compA = 0; size_t compB = 0;
//     size_t NA    = 0; size_t NB    = 0;
//     if(total_ij < Host_threads) //This thread belongs to the Host_threads//
//     {
//       MolA = 0;
//       Aij  = total_ij / NAds * NFrameworkAtomsPerThread;
//       MolB = total_ij % NAds;
//       NA  = NFrameworkAtomsPerThread;
//       if(total_ij == Host_threads - 1) 
//         if(Host_threads % NFrameworkAtomsPerThread != 0)
//           NA = Host_threads % NFrameworkAtomsPerThread; 
//       if(!HostHost) compB = 1; //If we do not consider host-host, start with host-guest
//       //Here we need to determine the Molecule ID and which component the molecule belongs to//
      
//       for(size_t ijk = compB; ijk < NumberComp; ijk++)
//       {
//         size_t Mol_ijk = System[ijk].size / System[ijk].Molsize;
//         totalsize     += Mol_ijk;
//         if(MolB >= totalsize)
//         {
//           compB++;
//           MolB -= Mol_ijk;
//         }
//       }
//       NB = System[compB].Molsize;
//       Bij = MolB * NB;
//     }
//     else //Adsorbate-Adsorbate//
//     { 
//       compA = 1; compB = 1;
//       size_t Ads_i = total_ij - Host_threads;
//       //https://stackoverflow.com/questions/27086195/linear-index-upper-triangular-matrix
//       MolA = NAds - 2 - std::floor(std::sqrt(-8*Ads_i + 4*NAds*(NAds-1)-7)/2.0 - 0.5);
//       MolB = Ads_i + MolA + 1 - NAds*(NAds-1)/2 + (NAds-MolA)*((NAds- MolA)-1)/2;
//       totalsize = 0;
//       //Determine the Molecule IDs and the component of MolA and MolB//
//       for(size_t ijk = 1; ijk < NumberComp; ijk++)
//       {
//         size_t Mol_ijk = System[ijk].size / System[ijk].Molsize;
//         totalsize     += Mol_ijk;
//         if(MolA >= totalsize)
//         {
//           compA++;
//           MolA -= Mol_ijk;
//         }
//       }
//       NA = System[compA].Molsize;
//       totalsize = 0;
//       for(size_t ijk = 1; ijk < NumberComp; ijk++)
//       {
//         size_t Mol_ijk = System[ijk].size / System[ijk].Molsize;
//         totalsize     += Mol_ijk;
//         if(MolB >= totalsize)
//         {
//           compB++;
//           MolB -= Mol_ijk;
//         }
//       }
//       NB = System[compB].Molsize;
//       Aij = MolA * NA; Bij = MolB * NB;
//     }
//     //printf("Thread: %lu, compA: %lu, compB: %lu, MolA: %lu, MolB: %lu, Aij: %lu, Bij: %lu, Molsizes: %lu %lu\n", total_ij, compA, compB, MolA, MolB, Aij, Bij, System[0].Molsize, System[1].Molsize);

//     sdata[ij_within_block] = 0.0;
//     //Initialize Blocksum//
//     if(cache_id == 0) Blocksum[blockIdx.x] = 0.0;

//     // Manually fusing/collapsing the loop //

//     const Atoms ComponentA=System[compA];
//     const Atoms ComponentB=System[compB];
//     double tempy = 0.0;
//     VDWCoulEnergy_Total(Box, ComponentA, ComponentB, Aij, Bij, FF, flag, Blockflag, tempy, NA, NB, UseOffset);
//     sdata[ij_within_block] = tempy;
//     //printf("ThreadID: %lu, HostThread: %lu, compA: %lu, compB: %lu, Aij: %lu, Bij: %lu, NA: %lu, NB: %lu, tempy: %.5f\n", total_ij, Host_threads, compA, compB, Aij, Bij, NA, NB, tempy);
//   }
//   __syncthreads();
//   //Partial block sum//
//   if(!Blockflag)
//   {
//     int i=blockDim.x / 2;
//     while(i != 0) 
//     {
//       if(cache_id < i) {sdata[cache_id] += sdata[cache_id + i];}
//       __syncthreads();
//       i /= 2;
//     }
//     if(cache_id == 0) {Blocksum[blockIdx.x] = sdata[0];}
//   }
//   else
//     flag[0] = true;
// }

__attribute__((always_inline))
void determine_comp_and_Atomindex_from_thread(Atoms* System, size_t& Atom, size_t& comp, size_t startComponent, size_t endComponent)
{
  /*
  //size_t count = Atom;
  comp = startComponent;
  for(size_t ijk = startComponent; ijk < endComponent; ijk++)
  {
    //totalsize += d_a[ijk].size;
    if(Atom >= System[ijk].size)
    {
      Atom -= System[ijk].size;
      comp ++;
    }
    else
    {
      //Atom = count;
      //comp = ijk;
      break;
    }
  }
  */
  size_t count = Atom;
  comp = startComponent;
  for(size_t ijk = startComponent; ijk < endComponent; ijk++)
  {
    //totalsize += d_a[ijk].size;
    if(count >= System[ijk].size)
    {
      count -= System[ijk].size;
      //comp ++;
    }
    else
    {
      Atom = count;
      comp = ijk;
      break;
    }
  }
  //return {(int) Atom, (int) comp};
}

__attribute__((always_inline))
void TotalVDWRealCoulomb(Boxsize Box, Atoms* System, ForceField FF, double* Blocksum, bool* flag, size_t InteractionPerThread, bool UseOffset, int3 BLOCK, int3 NComponents, size_t NFrameworkAtoms, size_t NAdsorbateAtoms, size_t NFrameworkZero_ExtraFramework, bool ConsiderIntra, const sycl::nd_item<1>& item, sycl::decorated_local_ptr<double> sdata)
{
  size_t Nblock = BLOCK.x() + BLOCK.y() + BLOCK.z();
  size_t HH_Nblock = BLOCK.x();
  size_t HG_Nblock = BLOCK.y();
  //size_t GG_Nblock = BLOCK.z(); //currently not used//

  size_t blockIdx  = item.get_group(0);
  size_t blockDim  = item.get_local_range(0);
  size_t threadIdx = item.get_local_id(0);
  
  int cache_id = threadIdx;

  size_t THREADIdx = item.get_global_id(0); if(THREADIdx > NFrameworkZero_ExtraFramework)
                                              
  sdata[threadIdx] = 0.0;
  sdata[threadIdx + blockDim] = 0.0;

  //Initialize Blocksum//
  if(cache_id == 0)
  {Blocksum[blockIdx] = 0.0; Blocksum[blockIdx + Nblock] = 0.0; }
 
    //Aij and Bij indicate the starting positions for the objects in the pairwise interaction//
  size_t AtomA = 0; size_t AtomB = 0;
  size_t MolA  = 0; size_t MolB  = 0;
  size_t compA = 0; size_t compB = 0;
  double2 tempy = {0.0, 0.0};

  size_t HH_Threads = HH_Nblock * blockDim;
  size_t HG_Threads = HG_Nblock * blockDim;
  //size_t GG_Threads = GG_Nblock * blockDim; //currently not used//

    for(size_t i = 0; i != InteractionPerThread; i++)
    {
      if(blockIdx < HH_Nblock)
      {
        size_t InteractionIdx = THREADIdx * InteractionPerThread + i;
        if(ConsiderIntra) //All Framework atom vs. All Framework atom//
        {
          AtomA = NFrameworkAtoms - 2 - sycl::floor(sycl::sqrt(-8*(int) InteractionIdx + 4*NFrameworkAtoms*(NFrameworkAtoms-1)-7)/2.0 - 0.5);
          AtomB = InteractionIdx + AtomA + 1 - NFrameworkAtoms*(NFrameworkAtoms-1)/2 + (NFrameworkAtoms-AtomA)*((NFrameworkAtoms-AtomA)-1)/2;
        determine_comp_and_Atomindex_from_thread(System, AtomA, compA, 0, NComponents.y());
        determine_comp_and_Atomindex_from_thread(System, AtomB, compB, 0, NComponents.y());
        }
        //If you have more than 1 framework components, then the following two conditions will always be there//
        else 
        {  
          if(InteractionIdx < NFrameworkZero_ExtraFramework) //Framework comp 0 vs. other framework atoms//
          {
            AtomA  = InteractionIdx % System[0].size; //Framework Component 0 Atom//
            AtomB  = InteractionIdx / System[0].size; //Framework Component >0 Atom//
            determine_comp_and_Atomindex_from_thread(System, AtomA, compA, 0, NComponents.y());
            determine_comp_and_Atomindex_from_thread(System, AtomB, compB, 1, NComponents.y());
          }
          else 
          {
            size_t NExtraFrameworkAtoms = NFrameworkAtoms - System[0].size;
            size_t InteractionIdx = THREADIdx * InteractionPerThread + i - NFrameworkZero_ExtraFramework;
            AtomA = NExtraFrameworkAtoms - 2 - sycl::floor(sycl::sqrt(-8*(int) InteractionIdx + 4*NExtraFrameworkAtoms*(NExtraFrameworkAtoms-1)-7)/2.0 - 0.5);
            AtomB = InteractionIdx + AtomA + 1 - NExtraFrameworkAtoms*(NExtraFrameworkAtoms-1)/2 + (NExtraFrameworkAtoms-AtomA)*((NExtraFrameworkAtoms-AtomA)-1)/2;
            determine_comp_and_Atomindex_from_thread(System, AtomA, compA, 1, NComponents.y());
            determine_comp_and_Atomindex_from_thread(System, AtomB, compB, 1, NComponents.y());
            //DEBUG//printf("THERE IS NExtraFrameworkAtoms x NExtraFrameworkAtoms INTERACTIONS, THREADIdx: %lu, blockIdx: %lu, threadIdx: %lu\n", THREADIdx, blockIdx, threadIdx);
          }
        }
      }
      else if(blockIdx < (HG_Nblock + HH_Nblock)) //Host-Guest//
      {
        //This thread belongs to the Host-Guest_threads//
        size_t InteractionIdx = (THREADIdx - HH_Threads) * InteractionPerThread + i;
        AtomA  = InteractionIdx / NAdsorbateAtoms; //Framework Atom//
        AtomB  = InteractionIdx % NAdsorbateAtoms; //Adsorbate Atom//

        determine_comp_and_Atomindex_from_thread(System, AtomA, compA, 0, NComponents.y());
        determine_comp_and_Atomindex_from_thread(System, AtomB, compB, NComponents.y, NComponents.x());
      }
      else //Guest-Guest//
      {
        size_t InteractionIdx = (THREADIdx - HH_Threads - HG_Threads) * InteractionPerThread + i;
        AtomA = NAdsorbateAtoms - 2 - sycl::floor(sycl::sqrt(-8*(int) InteractionIdx + 4*NAdsorbateAtoms*(NAdsorbateAtoms-1)-7)/2.0 - 0.5);
        AtomB = InteractionIdx + AtomA + 1 - NAdsorbateAtoms*(NAdsorbateAtoms-1)/2 + (NAdsorbateAtoms-AtomA)*((NAdsorbateAtoms-AtomA)-1)/2;

        determine_comp_and_Atomindex_from_thread(System, AtomA, compA, NComponents.y, NComponents.x());
        determine_comp_and_Atomindex_from_thread(System, AtomB, compB, NComponents.y, NComponents.x());
      }
      MolA = System[compA].MolID[AtomA];
      MolB = System[compB].MolID[AtomB];
      if(AtomA >= System[compA].size || AtomB >= System[compB].size) continue;
      if(!ConsiderIntra && (MolA == MolB) && (compA == compB)) continue;

      const Atoms ComponentA = System[compA];
      const Atoms ComponentB = System[compB];
      VDWCoulEnergy_Total(Box, ComponentA, ComponentB, FF, flag, tempy, AtomA, AtomB, compA, compB, UseOffset);
    }

  
  sdata[threadIdx] = tempy.x(); 
  sdata[threadIdx + blockDim] = tempy.y();

  item.barrier();
  //Partial block sum//
  int reductionsize=blockDim / 2;
  while(reductionsize != 0)
  {
    if(cache_id < reductionsize)
    {
      sdata[cache_id] += sdata[cache_id + reductionsize];
      sdata[cache_id + blockDim] += sdata[cache_id + reductionsize + blockDim];
    }
    item.barrier();
    reductionsize /= 2;
  }
  if(cache_id == 0)
  {
    Blocksum[blockIdx] = sdata[0];
    Blocksum[blockIdx + Nblock] = sdata[blockDim];
  }
  
}

//Zhao's note: here the totMol does not consider framework atoms, ONLY Adsorbates//
MoveEnergy Total_VDW_Coulomb_Energy(Simulations& Sim, Components& SystemComponents, ForceField FF, bool UseOffset)
//MoveEnergy Total_VDW_Coulomb_Energy(Simulations& Sim, ForceField FF, size_t totMol, size_t Host_threads, size_t Guest_threads, size_t NFrameworkAtomsPerThread, bool ConsiderHostHost, bool UseOffset)
{
  // MoveEnergy E;
  // if(Host_threads + Guest_threads == 0) return E;
  // double VDWRealE = 0.0;
  // size_t Nblock = 0; size_t Nthread = 0;
  // Setup_threadblock(Host_threads + Guest_threads, &Nblock, &Nthread);
  // if(Nblock > Sim.Nblocks)
  // {
  //   printf("More blocks for block sum is needed\n");
  //   //cudaMalloc(&Sim.Blocksum, Nblock * sizeof(double));
  // }

  // //Calculate the energy of the new systems//
  // //TotalVDWCoul<<<Nblock, Nthread, Nthread * sizeof(double)>>>(Sim.Box, Sim.d_a, FF, Sim.Blocksum, Sim.device_flag, Host_threads + Guest_threads, Host_threads, totMol, NFrameworkAtomsPerThread, ConsiderHostHost, UseOffset);
  // //printf("Total VDW + Real, Nblock = %zu, Nthread = %zu, Host: %zu, Guest: %zu, Allocated size: %zu\n", Nblock, Nthread, Host_threads, Guest_threads, Sim.Nblocks);
  // //Zhao's note: consider using the flag to check for overlap here//
  // //printf("Total Thread: %zu, Nblock: %zu, Nthread: %zu\n", Host_threads + Guest_threads, Nblock, Nthread);
  // //double BlockE[Nblock]; 
  // //que.memcpy(BlockE, Sim.Blocksum, sizeof(double) * Nblock).wait();
  // //for(size_t id = 0; id < Nblock; id++) E.GGVDW += BlockE[id];
  // return E;





  size_t NHostAtom = 0; size_t NGuestAtom = 0;
  for(size_t i = 0; i < SystemComponents.NComponents.y(); i++)
    NHostAtom += SystemComponents.Moleculesize[i] * SystemComponents.NumberOfMolecule_for_Component[i];
  for(size_t i = SystemComponents.NComponents.y(); i < SystemComponents.NComponents.x(); i++)
    NGuestAtom+= SystemComponents.Moleculesize[i] * SystemComponents.NumberOfMolecule_for_Component[i];

  bool ConsiderIntra = false;
  size_t HH_TotalThreads = 0; //if(ConsiderHostHost) HH_TotalThreads = NHostAtom * NHostAtom / 2;
  size_t NFrameworkZero_ExtraFramework = 0;
  if(ConsiderIntra) //Atom-Atom//
  {
    HH_TotalThreads = NHostAtom * (NHostAtom - 1) / 2;
  }
  else //if molecule-intra interactions are not considered, do component 0 x component 1-n_framework + component 1-n_framework x component 1-n_framework//
  {
    //printf("THERE IS MORE THAN 1 FRAMEWORK COMPONENTS\n");
    size_t NFrameworkComponentZeroAtoms = SystemComponents.Moleculesize[0] * SystemComponents.NumberOfMolecule_for_Component[0];
    size_t NExtraFrameworkAtoms = NHostAtom - NFrameworkComponentZeroAtoms;
    NFrameworkZero_ExtraFramework = NFrameworkComponentZeroAtoms * NExtraFrameworkAtoms;
    HH_TotalThreads = NFrameworkZero_ExtraFramework; //component 0 x component 1-n_framework
    
    HH_TotalThreads+= NExtraFrameworkAtoms * (NExtraFrameworkAtoms - 1) / 2; //component 1-n_framework x component 1-n_framework//
    //printf("Framework Comp Zero Atoms: %zu, Other Comp Atoms: %zu\n", NFrameworkComponentZeroAtoms, NExtraFrameworkAtoms);
    //printf("NFrameworkZero_ExtraFramework interactions: %zu, NExtraFrameworkAtoms * (NExtraFrameworkAtoms - 1) / 2: %zu\n", NFrameworkZero_ExtraFramework, NExtraFrameworkAtoms * (NExtraFrameworkAtoms - 1) / 2);
  }

  size_t HG_TotalThreads = NHostAtom * NGuestAtom; 
  size_t GG_TotalThreads = NGuestAtom * (NGuestAtom - 1) / 2; //GG_TotalThreads =  * NGuestAtom / 2;
  size_t InteractionPerThread = 100;

  size_t HHThreadsNeeded = HH_TotalThreads / InteractionPerThread + (HH_TotalThreads % InteractionPerThread == 0 ? 0 : 1);
  size_t HGThreadsNeeded = HG_TotalThreads / InteractionPerThread + (HG_TotalThreads % InteractionPerThread == 0 ? 0 : 1);
  size_t GGThreadsNeeded = GG_TotalThreads / InteractionPerThread + (GG_TotalThreads % InteractionPerThread == 0 ? 0 : 1);

  size_t HH_Nthread=0; size_t HH_Nblock=0; Setup_threadblock(HHThreadsNeeded, HH_Nblock, HH_Nthread);
  size_t HG_Nthread=0; size_t HG_Nblock=0; Setup_threadblock(HGThreadsNeeded, HG_Nblock, HG_Nthread);
  size_t GG_Nthread=0; size_t GG_Nblock=0; Setup_threadblock(GGThreadsNeeded, GG_Nblock, GG_Nthread);
  MoveEnergy E;
  
  if((HH_Nblock + HG_Nblock + GG_Nblock) == 0) return E;
  size_t Nblock = 0; size_t Nthread = 0;
  //Setup_threadblock(Host_threads + Guest_threads, Nblock, Nthread);
  Nthread = std::max(GG_Nthread, HG_Nthread);
  if(HH_Nthread > Nthread) Nthread = HH_Nthread;

  if(Nblock*2 > Sim.Nblocks)
  {
    printf("More blocks for block sum is needed\n");
    cudaMalloc(&Sim.Blocksum, 2*Nblock * sizeof(double));
    Sim.BlockSum = sycl::malloc<double>(2*Nblock, que);
  }

  int3 BLOCKS = {HH_Nblock, HG_Nblock, GG_Nblock};

  //Calculate the energy of the new systems//
  //Host-Guest + Guest-Guest//
  Nblock = HH_Nblock + HG_Nblock + GG_Nblock;
  //printf("Atoms: %zu %zu\n", NHostAtom, NGuestAtom);
  //printf("Interactions: %zu %zu %zu\n", HH_TotalThreads, HG_TotalThreads, GG_TotalThreads);
  //printf("Nblock %zu, blocks: %zu %zu %zu, threads needed: %zu %zu %zu, Nthread: %zu\n", Nblock, HH_Nblock, HG_Nblock, GG_Nblock, HHThreadsNeeded, HGThreadsNeeded, GGThreadsNeeded, Nthread);

  //Set Overlap Flag//
  que.memset(Sim.device_flag, false, sizeof(bool));
 
  que.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<double, 1> local_mem(sycl::range<1>(2 * Nthread), cgh);
      
    cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(Nblock * Nthread, Nthread)), [=](sycl::nd_item<1> item) {
      TotalVDWRealCoulomb(Sim.Box, Sim.d_a, FF, Sim.Blocksum, Sim.device_flag, InteractionPerThread, UseOffset, BLOCKS, SystemComponents.NComponents, NHostAtom, NGuestAtom, NFrameworkZero_ExtraFramework, ConsiderIntra, item, local_mem.get_multi_ptr<sycl::access::decorated::yes>());
    });
  });
  
  que.wait();

  //printf("Total VDW + Real, Nblock = %zu, Nthread = %zu, Host: %zu, Guest: %zu, Allocated size: %zu\n", Nblock, Nthread, Host_threads, Guest_threads, Sim.Nblocks);
  //Zhao's note: consider using the flag to check for overlap here//
  //printf("Total Thread: %zu, Nblock: %zu, Nthread: %zu\n", Host_threads + Guest_threads, Nblock, Nthread);
  double BlockE[Nblock*2];
  que.memcpy(BlockE, Sim.Blocksum, Nblock*2 * sizeof(double)).wait();

  for(size_t id = 0; id < HH_Nblock; id++) E.HHVDW += BlockE[id];
  for(size_t id = HH_Nblock; id < HH_Nblock + HG_Nblock; id++) E.HGVDW += BlockE[id];
  for(size_t id = HH_Nblock + HG_Nblock; id < Nblock; id++) E.GGVDW += BlockE[id];

  for(size_t id = Nblock; id < Nblock + HH_Nblock; id++) E.HHReal += BlockE[id];
  for(size_t id = Nblock + HH_Nblock; id < Nblock + HH_Nblock + HG_Nblock; id++) E.HGReal += BlockE[id];
  for(size_t id = Nblock + HH_Nblock + HG_Nblock; id < Nblock+Nblock; id++) E.GGReal += BlockE[id];
  //printf("GPU VDW REAL ENERGY:\n"); E.print();

  return E;  
}
