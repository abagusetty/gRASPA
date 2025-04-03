#pragma once

#include "mc_widom.h"
#include "mc_swap_utilities.h"
#include "lambda.h"
#include <cmath>

__attribute__((always_inline)) void StoreNewLocation_Reinsertion(Atoms Mol, Atoms NewMol, double3* temp, size_t SelectedTrial, size_t Moleculesize, const sycl::nd_item<1> &item)
{
  size_t i = item.get_global_id(0);
  if(i == 0)
  {
    if(Moleculesize == 1) temp[0] = NewMol.pos[SelectedTrial];
    else temp[0] = Mol.pos[0];
  }
  else
  {
    size_t chainsize = Moleculesize - 1; // FOr trial orientations //
    size_t selectsize = SelectedTrial*chainsize+(i-1);
    temp[i] = NewMol.pos[selectsize];
  }
}

__attribute__((always_inline))
void Update_Reinsertion_data(Atoms *d_a, double3 *temp,
                             size_t SelectedComponent, size_t UpdateLocation,
                             const sycl::nd_item<1> &item)
{
  size_t i = item.get_global_id(0);
  size_t realLocation = UpdateLocation + i;
  d_a[SelectedComponent].pos[realLocation] = temp[i];
}


//Zhao's note: added feature for creating fractional molecules//
static inline MoveEnergy
CreateMolecule(Components &SystemComponents, Simulations &Sims, ForceField &FF,
               RandomNumber &Random, WidomStruct &Widom,
               size_t SelectedMolInComponent, size_t SelectedComponent,
               double2 newScale)
{
  bool SuccessConstruction = false;
  double Rosenbluth = 0.0;
  size_t SelectedTrial = 0;
  double preFactor = 0.0;
  
  //Zhao's note: For creating the fractional molecule, there is no previous step, so set the flag to false//
  MoveEnergy energy = Insertion_Body(SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, SelectedComponent, Rosenbluth, SuccessConstruction, SelectedTrial, preFactor, false, newScale); 
  if(!SuccessConstruction) 
  {
    energy.zero();
    return energy;
  }
  double IdealRosen = SystemComponents.IdealRosenbluthWeight[SelectedComponent];
  double RANDOM = 1e-100;
  if(RANDOM < preFactor * Rosenbluth / IdealRosen)
  { // accept the move
    size_t UpdateLocation = SystemComponents.Moleculesize[SelectedComponent] * SystemComponents.NumberOfMolecule_for_Component[SelectedComponent];
    //Zhao's note: here needs more consideration: need to update after implementing polyatomic molecule
    auto SystemComponents_Moleculesize_SelectedComponent_ct6 =
        (int)SystemComponents.Moleculesize[SelectedComponent];

    sycl_get_queue()->parallel_for(sycl::nd_range<1>(1,1), [=](sycl::nd_item<1> item) {
        Update_insertion_data(
                              Sims.d_a, Sims.Old, Sims.New, SelectedTrial, SelectedComponent,
                              UpdateLocation,
                              SystemComponents_Moleculesize_SelectedComponent_ct6, item);
      });

    if(!FF.noCharges && SystemComponents.hasPartialCharge[SelectedComponent])
    {
      Update_Vector_Ewald(Sims.Box, false, SystemComponents);
    }
    Update_NumberOfMolecules(SystemComponents, Sims.d_a, SelectedComponent, INSERTION);
    return energy;
  }
  energy.zero();
  return energy;
}
//Zhao's note: This insertion only takes care of the full (not fractional) molecules//
static inline MoveEnergy Insertion(Components& SystemComponents, Simulations& Sims, ForceField& FF, RandomNumber& Random, WidomStruct& Widom, size_t SelectedMolInComponent, size_t SelectedComponent)
{
  //Get Number of Molecules for this component (For updating TMMC)//
  //This is the OLD STATE//
  double NMol = SystemComponents.NumberOfMolecule_for_Component[SelectedComponent];
  if(SystemComponents.hasfractionalMolecule[SelectedComponent]) NMol--;
  double TMMCPacc = 0.0;

  SystemComponents.Moves[SelectedComponent].InsertionTotal ++;
  bool SuccessConstruction = false;
  double Rosenbluth = 0.0;
  size_t SelectedTrial = 0;
  double preFactor = 0.0;

  double2 newScale = SystemComponents.Lambda[SelectedComponent].SET_SCALE(1.0); // Set scale for full molecule (lambda = 1.0)//
  MoveEnergy energy = Insertion_Body(SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, SelectedComponent, Rosenbluth, SuccessConstruction, SelectedTrial, preFactor, false, newScale); 
  if(!SuccessConstruction) 
  {
    //If unsuccessful move (Overlap), Pacc = 0//
    SystemComponents.Tmmc[SelectedComponent].Update(TMMCPacc, NMol, INSERTION);
    SystemComponents.Moves[SelectedComponent].RecordRosen(0.0, INSERTION);
    energy.zero();
    return energy;
  }
  double IdealRosen = SystemComponents.IdealRosenbluthWeight[SelectedComponent];
  double RANDOM = Get_Uniform_Random();
  TMMCPacc = preFactor * Rosenbluth / IdealRosen; //Unbiased Acceptance//
  //Apply the bias according to the macrostate//
  SystemComponents.Tmmc[SelectedComponent].ApplyWLBias(preFactor, SystemComponents.Beta, NMol, INSERTION);
  SystemComponents.Tmmc[SelectedComponent].ApplyTMBias(preFactor, SystemComponents.Beta, NMol, INSERTION);

  bool Accept = false;
  if(RANDOM < preFactor * Rosenbluth / IdealRosen) Accept = true;
  SystemComponents.Tmmc[SelectedComponent].TreatAccOutofBound(Accept, NMol, INSERTION);
  SystemComponents.Moves[SelectedComponent].RecordRosen(Rosenbluth, INSERTION);

  if(Accept)
  { // accept the move
    SystemComponents.Moves[SelectedComponent].InsertionAccepted ++;
    AcceptInsertion(SystemComponents, Sims, SelectedComponent, SelectedTrial, FF.noCharges, INSERTION);
    SystemComponents.Tmmc[SelectedComponent].Update(TMMCPacc, NMol, INSERTION);
    return energy;
  }
  //else
  //Zhao's note: Even if the move is rejected by acceptance rule, still record the Pacc//
  SystemComponents.Tmmc[SelectedComponent].Update(TMMCPacc, NMol, INSERTION);
  energy.zero();
  return energy;
}

static inline MoveEnergy Deletion(Components& SystemComponents, Simulations& Sims, ForceField& FF,  RandomNumber& Random, WidomStruct& Widom, size_t SelectedMolInComponent, size_t SelectedComponent)
{
  //Get Number of Molecules for this component (For updating TMMC)//
  double NMol = SystemComponents.NumberOfMolecule_for_Component[SelectedComponent];
  if(SystemComponents.hasfractionalMolecule[SelectedComponent]) NMol--;
  double TMMCPacc = 0.0;

  SystemComponents.Moves[SelectedComponent].DeletionTotal ++;
 
  double preFactor = 0.0;
  bool SuccessConstruction = false;
  MoveEnergy energy;
  double Rosenbluth = 0.0;
  size_t UpdateLocation = 0;
  int CBMCType = CBMC_DELETION; //Deletion//
  double2 Scale = SystemComponents.Lambda[SelectedComponent].SET_SCALE(
      1.0); // Set scale for full molecule (lambda = 1.0), Zhao's note: This is
            // not used in deletion, just set to 1//
  //Wrapper for the deletion move//
  energy = Deletion_Body(SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, SelectedComponent, UpdateLocation, Rosenbluth, SuccessConstruction, preFactor, Scale);
  if(!SuccessConstruction)
  {
    //If unsuccessful move (Overlap), Pacc = 0//
    SystemComponents.Tmmc[SelectedComponent].Update(TMMCPacc, NMol, DELETION);
    SystemComponents.Moves[SelectedComponent].RecordRosen(0.0, DELETION);
    energy.zero();
    return energy;
  }
  //DEBUG//
  /*
  if(SystemComponents.CURRENTCYCLE == 28981)
  { printf("Selected Molecule: %zu\n", SelectedMolInComponent);
    printf("DELETION MOVE ENERGY: "); energy.print();
  }
  */
  double IdealRosen = SystemComponents.IdealRosenbluthWeight[SelectedComponent];
  double RANDOM = Get_Uniform_Random();
  TMMCPacc = preFactor * IdealRosen / Rosenbluth; //Unbiased Acceptance//
  //Apply the bias according to the macrostate//
  SystemComponents.Tmmc[SelectedComponent].ApplyWLBias(preFactor, SystemComponents.Beta, NMol, DELETION);
  SystemComponents.Tmmc[SelectedComponent].ApplyTMBias(preFactor, SystemComponents.Beta, NMol, DELETION);

  bool Accept = false;
  if(RANDOM < preFactor * IdealRosen / Rosenbluth) Accept = true;
  SystemComponents.Tmmc[SelectedComponent].TreatAccOutofBound(Accept, NMol, DELETION);

  if(Accept)
  { // accept the move
    SystemComponents.Moves[SelectedComponent].DeletionAccepted ++;
    AcceptDeletion(SystemComponents, Sims, SelectedComponent, UpdateLocation, SelectedMolInComponent, FF.noCharges);
    //If unsuccessful move (Overlap), Pacc = 0//
    SystemComponents.Tmmc[SelectedComponent].Update(TMMCPacc, NMol, DELETION);
    energy.take_negative();
    return energy;
  }
  //Zhao's note: Even if the move is rejected by acceptance rule, still record the Pacc//
  SystemComponents.Tmmc[SelectedComponent].Update(TMMCPacc, NMol, DELETION);
  energy.zero();
  return energy;
}

static inline void GibbsParticleTransfer(std::vector<Components>& SystemComponents, Simulations*& Sims, ForceField FF, RandomNumber Random, std::vector<WidomStruct>& Widom, std::vector<SystemEnergies>& Energy, size_t SelectedComponent, Gibbs& GibbsStatistics)
{
  size_t NBox = SystemComponents.size();
  size_t SelectedBox = 0;
  size_t OtherBox    = 1;
  GibbsStatistics.GibbsXferStats.x() += 1.0;

  if(Get_Uniform_Random() > 0.5)
  {
    SelectedBox = 1;
    OtherBox    = 0;
  }
  
  //printf("Performing Gibbs Particle Transfer Move! on Box[%zu]\n", SelectedBox);
  double2 Scale = {1.0, 1.0};
  bool TransferFractionalMolecule = false;
  //Randomly Select a Molecule from the OtherBox in the SelectedComponent//
  if(SystemComponents[OtherBox].NumberOfMolecule_for_Component[SelectedComponent] == 0) return;
  size_t DeletionSelectedMol = (size_t) (Get_Uniform_Random()*(SystemComponents[OtherBox].NumberOfMolecule_for_Component[SelectedComponent]));
  //Special treatment for transfering the fractional molecule//
  if(SystemComponents[OtherBox].hasfractionalMolecule[SelectedComponent] && SystemComponents[OtherBox].Lambda[SelectedComponent].FractionalMoleculeID == DeletionSelectedMol)
  {
    double oldBin    = SystemComponents[OtherBox].Lambda[SelectedComponent].currentBin;
    double delta     = SystemComponents[OtherBox].Lambda[SelectedComponent].delta;
    double oldLambda = delta * static_cast<double>(oldBin);
    Scale            = SystemComponents[OtherBox].Lambda[SelectedComponent].SET_SCALE(oldLambda);
    TransferFractionalMolecule = true;
  }
  //Perform Insertion on the selected System, then deletion on the other system//

  /////////////////////////////////////////////////
  // PERFORMING INSERTION ON THE SELECTED SYSTEM //
  /////////////////////////////////////////////////
  size_t InsertionSelectedTrial = 0;
  double InsertionPrefactor     = 0.0;
  double InsertionRosen         = 0.0;
  bool   InsertionSuccess       = false;
  size_t InsertionSelectedMol   = 0; //It is safer to ALWAYS choose the first atom as the template for CBMC_INSERTION//
  MoveEnergy InsertionEnergy = Insertion_Body(SystemComponents[SelectedBox], Sims[SelectedBox], FF, Random, Widom[SelectedBox], InsertionSelectedMol, SelectedComponent, InsertionRosen, InsertionSuccess, InsertionSelectedTrial, InsertionPrefactor, false, Scale);
  printf("Gibbs Particle Insertion energy: "); InsertionEnergy.print();
  if(!InsertionSuccess) return;

  /////////////////////////////////////////////
  // PERFORMING DELETION ON THE OTHER SYSTEM //
  /////////////////////////////////////////////
  size_t DeletionUpdateLocation = 0;
  double DeletionPrefactor      = 0.0;
  double DeletionRosen          = 0.0;
  bool   DeletionSuccess        = false;
  MoveEnergy DeletionEnergy = Deletion_Body(SystemComponents[OtherBox], Sims[OtherBox], FF, Random, Widom[OtherBox], DeletionSelectedMol, SelectedComponent, DeletionUpdateLocation, DeletionRosen, DeletionSuccess, DeletionPrefactor, Scale);
  printf("Gibbs Particle Deletion energy: "); DeletionEnergy.print();
  if(!DeletionSuccess) return;

  bool Accept = false;

  double NMolA= static_cast<double>(SystemComponents[SelectedBox].TotalNumberOfMolecules - SystemComponents[SelectedBox].NumberOfFrameworks);
  double NMolB= static_cast<double>(SystemComponents[OtherBox].TotalNumberOfMolecules - SystemComponents[OtherBox].NumberOfFrameworks);
  //Minus the fractional molecule//
  for(size_t comp = 0; comp < SystemComponents[SelectedBox].Total_Components; comp++)
  {
    if(SystemComponents[SelectedBox].hasfractionalMolecule[comp])
    {
      NMolA-=1.0;
    }
  }
  for(size_t comp = 0; comp < SystemComponents[OtherBox].Total_Components; comp++)
  {
    if(SystemComponents[OtherBox].hasfractionalMolecule[comp])
    {
      NMolB-=1.0;
    }
  }

  //This assumes that the two boxes share the same temperature, it might not be true//
  if(Get_Uniform_Random()< (InsertionRosen * NMolB * Sims[SelectedBox].Box.Volume) / (DeletionRosen * (NMolA + 1) * Sims[OtherBox].Box.Volume)) Accept = true;

  //printf("SelectedBox: %zu, OtherBox: %zu, InsertionEnergy: %.5f(%.5f %.5f), DeletionEnergy: %.5f(%.5f %.5f)\n", SelectedBox, OtherBox, InsertionEnergy, SystemComponents[SelectedBox].tempdeltaVDWReal, SystemComponents[SelectedBox].tempdeltaEwald, DeletionEnergy, SystemComponents[OtherBox].tempdeltaVDWReal, SystemComponents[OtherBox].tempdeltaEwald);

  if(Accept)
  {
    GibbsStatistics.GibbsXferStats.y() += 1.0;
    // Zhao's note: the assumption for the below implementation is that the component index are the same for both systems //
    // for example, methane in box A is component 1, it has to be component 1 in box B //
    //For the box that is deleting the molecule, update the recorded fractional molecule ID//
    // Update System information regarding the fractional molecule, if the fractional molecule is being transfered //
    if(TransferFractionalMolecule)
    {
      SystemComponents[SelectedBox].hasfractionalMolecule[SelectedComponent] = true;
      SystemComponents[OtherBox].hasfractionalMolecule[SelectedComponent]    = false;
      SystemComponents[SelectedBox].Lambda[SelectedComponent].currentBin = SystemComponents[OtherBox].Lambda[SelectedComponent].currentBin;
    }
    ///////////////////////////////////////////
    // UPDATE INSERTION FOR THE SELECTED BOX //
    ///////////////////////////////////////////
    AcceptInsertion(SystemComponents[SelectedBox], Sims[SelectedBox], SelectedComponent, InsertionSelectedTrial, FF.noCharges, INSERTION);

    SystemComponents[SelectedBox].deltaE += InsertionEnergy;
    //Energy[SelectedBox].running_energy += InsertionEnergy.total();
    //printf("Insert Box: %zu, Insertion Energy: %.5f\n", SelectedBox, InsertionEnergy);

    ///////////////////////////////////////
    // UPDATE DELETION FOR THE OTHER BOX //
    ///////////////////////////////////////
    AcceptDeletion(SystemComponents[OtherBox], Sims[OtherBox], SelectedComponent, DeletionUpdateLocation, DeletionSelectedMol, FF.noCharges);
    Energy[OtherBox].running_energy -= DeletionEnergy.total();
    SystemComponents[OtherBox].deltaE -= DeletionEnergy;
    //printf("Delete Box: %zu, Insertion Energy: %.5f\n", OtherBox, DeletionEnergy);
  }
}

void copy_firstbead_to_new(Atoms NEW, Atoms* d_a, size_t comp, size_t position)
{
  NEW.pos[0] = d_a[comp].pos[position];
}

void Update_IdentitySwap_Insertion_data(Atoms* d_a, double3* temp, size_t NEWComponent, size_t UpdateLocation, size_t MolID, size_t Molsize)
{
  //Zhao's note: assuming not working with fractional molecule for Identity swap//
  for(size_t i = 0; i < Molsize; i++)
  {
    size_t realLocation = UpdateLocation + i;
    d_a[NEWComponent].pos[realLocation]   = temp[i];
    d_a[NEWComponent].scale[realLocation] = 1.0;
    d_a[NEWComponent].charge[realLocation]= d_a[NEWComponent].charge[i];
    d_a[NEWComponent].scaleCoul[realLocation]=1.0;
    d_a[NEWComponent].Type[realLocation] = d_a[NEWComponent].Type[i];
    d_a[NEWComponent].MolID[realLocation] = MolID;
  }
  d_a[NEWComponent].size += Molsize;
}

static inline MoveEnergy IdentitySwapMove(Components& SystemComponents, Simulations& Sims, WidomStruct& Widom, ForceField& FF, RandomNumber& Random, size_t OLDMolInComponent, size_t OLDComponent)
{
  sycl::queue &que = *sycl_get_queue();

  //Identity Swap is sort-of Reinsertion//
  //The difference is that the component of the molecule are different//
  //Selected Molecule is the molecule that is being identity-swapped//
  
  size_t NEWComponent = OLDComponent;
  while(NEWComponent == OLDComponent || NEWComponent == 0 || NEWComponent >= SystemComponents.Total_Components)
  {
    NEWComponent = (size_t) (Get_Uniform_Random()*(SystemComponents.Total_Components - 1)) + 1;
  }
  //printf("OLDComp: %zu, NEWComp: %zu\n", OLDComponent, NEWComponent);

  double NNEWMol = static_cast<double>(SystemComponents.NumberOfMolecule_for_Component[NEWComponent]);
  double NOLDMol = static_cast<double>(SystemComponents.NumberOfMolecule_for_Component[OLDComponent]);
  if(SystemComponents.hasfractionalMolecule[NEWComponent]) NNEWMol--;
  if(SystemComponents.hasfractionalMolecule[OLDComponent]) NOLDMol--;

  size_t NEWMolInComponent = SystemComponents.NumberOfMolecule_for_Component[NEWComponent];

  SystemComponents.Moves[OLDComponent].IdentitySwapRemoveTotal ++;
  SystemComponents.Moves[NEWComponent].IdentitySwapAddTotal ++;
  SystemComponents.Moves[OLDComponent].IdentitySwap_Total_TO[NEWComponent] ++;

  bool SuccessConstruction = false;
  size_t SelectedTrial = 0;
  MoveEnergy energy; MoveEnergy old_energy; double StoredR = 0.0;

  ///////////////
  // INSERTION //
  ///////////////
  int CBMCType = IDENTITY_SWAP_NEW; //Reinsertion-Insertion//
  //Take care of the frist bead location HERE//
  size_t OLD_LOCATION  = OLDMolInComponent * SystemComponents.Moleculesize[OLDComponent];
  que.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
        [=](sycl::nd_item<3> item) {
          copy_firstbead_to_new(Sims.New, Sims.d_a, OLDComponent, OLD_LOCATION);
  });
  //WRITE THE component and molecule ID THAT IS GOING TO BE IGNORED DURING THE ENERGY CALC//
  Sims.ExcludeList[0] = {OLDComponent, OLDMolInComponent};
  double2 newScale  = SystemComponents.Lambda[NEWComponent].SET_SCALE(1.0); //Zhao's note: not used in reinsertion, just set to 1.0//
  double Rosenbluth=Widom_Move_FirstBead_PARTIAL(SystemComponents, Sims, FF, Random, Widom, NEWMolInComponent, NEWComponent, CBMCType, StoredR, &SelectedTrial, &SuccessConstruction, &energy, newScale);

  if(Rosenbluth <= 1e-150) SuccessConstruction = false; //Zhao's note: added this protection bc of weird error when testing GibbsParticleXfer

  if(!SuccessConstruction)
  {
    energy.zero(); Sims.ExcludeList[0] = {-1, -1}; //Set to negative so that excludelist is ignored
    return energy;
  }

  //printf("NEW First-Bead ENERGY:"); energy.print();

  if(SystemComponents.Moleculesize[NEWComponent] > 1 && Rosenbluth > 1e-150)
  {
    size_t SelectedFirstBeadTrial = SelectedTrial;
    MoveEnergy temp_energy = energy;
    Rosenbluth*=Widom_Move_Chain_PARTIAL(SystemComponents, Sims, FF, Random, Widom, NEWMolInComponent, NEWComponent, CBMCType, &SelectedTrial, &SuccessConstruction, &energy, SelectedFirstBeadTrial, newScale); //True for doing insertion for reinsertion, different in MoleculeID//
    if(Rosenbluth <= 1e-150) SuccessConstruction = false; //Zhao's note: added this protection bc of weird error when testing GibbsParticleXfer
    if(!SuccessConstruction)
    {
      energy.zero(); Sims.ExcludeList[0] = {-1, -1}; //Set to negative so that excludelist is ignored
      return energy;
    }
    energy += temp_energy;
  }
  //printf("NEW MOLECULE ENERGY:"); energy.print();

  // Store The New Locations //

  size_t NEWCOMP_MOLSIZE = SystemComponents.Moleculesize[NEWComponent];
  que.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
      [=](sycl::nd_item<3> item) {
        StoreNewLocation_Reinsertion(Sims.Old, Sims.New, Sims.temp, SelectedTrial, NEWCOMP_MOLSIZE);
  });

  /////////////
  // RETRACE //
  /////////////
  
  Sims.ExcludeList[0] = {-1, -1}; //Set to negative so that excludelist is ignored

  CBMCType = REINSERTION_RETRACE; //Identity_Swap for the new configuration//
  double Old_Rosen=Widom_Move_FirstBead_PARTIAL(SystemComponents, Sims, FF, Random, Widom, OLDMolInComponent, OLDComponent, CBMCType, StoredR, &SelectedTrial, &SuccessConstruction, &old_energy, newScale);
  if(SystemComponents.Moleculesize[OLDComponent] > 1)
  {
    size_t SelectedFirstBeadTrial = SelectedTrial;
    MoveEnergy temp_energy = old_energy;
    Old_Rosen*=Widom_Move_Chain_PARTIAL(SystemComponents, Sims, FF, Random, Widom, OLDMolInComponent, OLDComponent, CBMCType, &SelectedTrial, &SuccessConstruction, &old_energy, SelectedFirstBeadTrial, newScale);
    old_energy += temp_energy;
  }

  //printf("OLD MOLECULE ENERGY:"); old_energy.print();
  energy -= old_energy;

  //Calculate Ewald//
  size_t UpdateLocation = SystemComponents.Moleculesize[OLDComponent] * OLDMolInComponent;
  if(!FF.noCharges)
  {
    double2 EwaldE = GPU_EwaldDifference_IdentitySwap(Sims.Box, Sims.d_a, Sims.Old, Sims.temp, FF, Sims.Blocksum, SystemComponents, OLDComponent, NEWComponent, UpdateLocation);
    energy.GGEwaldE = EwaldE.x();
    energy.HGEwaldE = EwaldE.y();
    Rosenbluth *= std::exp(-SystemComponents.Beta * (EwaldE.x() + EwaldE.y()));
  }

  energy.TailE = TailCorrectionIdentitySwap(SystemComponents, NEWComponent, OLDComponent, FF.size, Sims.Box.Volume);
  Rosenbluth *= std::exp(-SystemComponents.Beta * energy.TailE);

  //NO DEEP POTENTIAL FOR THIS MOVE!//
  if(SystemComponents.UseDNNforHostGuest) throw std::runtime_error("NO DEEP POTENTIAL FOR IDENTITY SWAP!");

  /////////////////////////////
  // PREPARE ACCEPTANCE RULE //
  /////////////////////////////
  //(P_New * N_Old) / (P_Old * (N_New + 1))
  double preFactor = (SystemComponents.MolFraction[NEWComponent] * SystemComponents.FugacityCoeff[NEWComponent]) * NOLDMol / ((SystemComponents.MolFraction[OLDComponent] * SystemComponents.FugacityCoeff[OLDComponent]) * (NNEWMol + 1));
  double NEWIdealRosen = SystemComponents.IdealRosenbluthWeight[NEWComponent];
  double OLDIdealRosen = SystemComponents.IdealRosenbluthWeight[OLDComponent];

  double Pacc = preFactor * (Rosenbluth / NEWIdealRosen) / (Old_Rosen / OLDIdealRosen);
  //Determine whether to accept or reject the insertion
  double RANDOM = Get_Uniform_Random();
  bool Accept = false;
  if(RANDOM < Pacc) Accept = true;

  if(Accept)
  { // accept the move
    SystemComponents.Moves[NEWComponent].IdentitySwapAddAccepted ++;
    SystemComponents.Moves[OLDComponent].IdentitySwapRemoveAccepted ++;

    SystemComponents.Moves[OLDComponent].IdentitySwap_Acc_TO[NEWComponent] ++;

    size_t LastMolecule = SystemComponents.NumberOfMolecule_for_Component[OLDComponent]-1;
    size_t LastLocation = LastMolecule*SystemComponents.Moleculesize[OLDComponent];

    int OldComponentMolsize = (int)SystemComponents.Moleculesize[OLDComponent];
    que.parallel_for(sycl::nd_range<1>(1,1), [=](sycl::nd_item<1> item) {
        Update_deletion_data(Sims.d_a, OLDComponent, UpdateLocation,
                             OldComponentMolsize, LastLocation,
                             item);
      });
   
    //The function below will only be processed if the system has a fractional molecule and the transfered molecule is NOT the fractional one //
    if((SystemComponents.hasfractionalMolecule[OLDComponent])&&(LastMolecule == SystemComponents.Lambda[OLDComponent].FractionalMoleculeID))
    {
      //Since the fractional molecule is moved to the place of the selected deleted molecule, update fractional molecule ID on host
      SystemComponents.Lambda[OLDComponent].FractionalMoleculeID = OLDMolInComponent;
    }
 
    UpdateLocation = SystemComponents.Moleculesize[NEWComponent] * NEWMolInComponent;

    size_t NewComponentMolsize = SystemComponents.Moleculesize[NEWComponent];
    que.single_task([=](){
        Update_IdentitySwap_Insertion_data(Sims.d_a, Sims.temp, NEWComponent, UpdateLocation, NEWMolInComponent, NewComponentMolsize);
      });

    Update_NumberOfMolecules(SystemComponents, Sims.d_a, NEWComponent, INSERTION);
    Update_NumberOfMolecules(SystemComponents, Sims.d_a, OLDComponent, DELETION);

    if(!FF.noCharges && ((SystemComponents.hasPartialCharge[NEWComponent]) ||(SystemComponents.hasPartialCharge[OLDComponent])))
      Update_Vector_Ewald(Sims.Box, false, SystemComponents);
    //energy.print();
    return energy;
  }
  else
  {
    energy.zero();
    return energy;
  }
}
