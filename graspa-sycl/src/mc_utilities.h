#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
////////////////////////////////////////////////////////////
// FUNCATIONS RELATED TO ALLOCATING MORE SPACE ON THE GPU //
////////////////////////////////////////////////////////////

void AllocateMoreSpace_CopyToTemp(Atoms* d_a, Atoms temp, size_t Space, size_t SelectedComponent)
{
  for(size_t i = 0; i < Space; i++)
  {
    temp.pos[i]       = d_a[SelectedComponent].pos[i];
    temp.scale[i]     = d_a[SelectedComponent].scale[i];
    temp.charge[i]    = d_a[SelectedComponent].charge[i];
    temp.scaleCoul[i] = d_a[SelectedComponent].scaleCoul[i];
    temp.Type[i]      = d_a[SelectedComponent].Type[i];
    temp.MolID[i]     = d_a[SelectedComponent].MolID[i];
  }
}

void AllocateMoreSpace_CopyBack(Atoms* d_a, Atoms temp, size_t Space, size_t Newspace, size_t SelectedComponent)
{
  d_a[SelectedComponent].Allocate_size = Newspace;
  for(size_t i = 0; i < Space; i++)
  {
    d_a[SelectedComponent].pos[i]       = temp.pos[i];
    d_a[SelectedComponent].scale[i]     = temp.scale[i];
    d_a[SelectedComponent].charge[i]    = temp.charge[i];
    d_a[SelectedComponent].scaleCoul[i] = temp.scaleCoul[i];
    d_a[SelectedComponent].Type[i]      = temp.Type[i];
    d_a[SelectedComponent].MolID[i]     = temp.MolID[i];
  /*
    System_comp.pos[i]         = temp.pos[i];
    System_comp.scale[i]     = temp.scale[i];
    System_comp.charge[i]    = temp.charge[i];
    System_comp.scaleCoul[i] = temp.scaleCoul[i];
    System_comp.Type[i]      = temp.Type[i];
    System_comp.MolID[i]     = temp.MolID[i];
  */
  }
  //test the new allocation//
  //d_a[SelectedComponent].pos[Newspace-1].x = 0.0;
  /*if(d_a[SelectedComponent].pos[Newspace-1].x = 0.0)
  {
  }*/
}

void AllocateMoreSpace(Atoms*& d_a, size_t SelectedComponent, Components& SystemComponents)
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  printf("Allocating more space on device\n");
  Atoms temp; // allocate a struct on the device for copying data.
  //Atoms tempSystem[SystemComponents.Total_Components];
  size_t Copysize=SystemComponents.Allocate_size[SelectedComponent];
  size_t Morespace = 1024;
  size_t Newspace = Copysize+Morespace;
  //Allocate space on the temporary struct
  /*
  DPCT1083:38: The size of double3 in the migrated code may be different from
  the original code. Check that the allocated memory size in the migrated code
  is correct.
  */
  (temp.pos) = (typename std::remove_reference<decltype(temp.pos)>::type)
      sycl::malloc_device(Copysize * sizeof(double3), q_ct1);
  (temp.scale) = (typename std::remove_reference<decltype(temp.scale)>::type)
      sycl::malloc_device(Copysize * sizeof(double), q_ct1);
  (temp.charge) = (typename std::remove_reference<decltype(temp.charge)>::type)
      sycl::malloc_device(Copysize * sizeof(double), q_ct1);
  (temp.scaleCoul) =
      (typename std::remove_reference<decltype(temp.scaleCoul)>::type)
          sycl::malloc_device(Copysize * sizeof(double), q_ct1);
  (temp.Type) = (typename std::remove_reference<decltype(temp.Type)>::type)
      sycl::malloc_device(Copysize * sizeof(size_t), q_ct1);
  (temp.MolID) = (typename std::remove_reference<decltype(temp.MolID)>::type)
      sycl::malloc_device(Copysize * sizeof(size_t), q_ct1);
  // Copy data to temp
  /*
  DPCT1069:65: The argument 'temp' of the kernel function contains virtual
  pointer(s), which cannot be dereferenced. Try to migrate the code with
  "usm-level=restricted".
  */
  q_ct1.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
      [=](sycl::nd_item<3> item_ct1) {
        AllocateMoreSpace_CopyToTemp(d_a, temp, Copysize, SelectedComponent);
      });
  // Allocate more space on the device pointers

  Atoms System[SystemComponents.Total_Components];
      q_ct1
          .memcpy(System, d_a,
                  SystemComponents.Total_Components * sizeof(Atoms))
          .wait();

  /*
  DPCT1083:39: The size of double3 in the migrated code may be different from
  the original code. Check that the allocated memory size in the migrated code
  is correct.
  */
  (System[SelectedComponent].pos) =
      (typename std::remove_reference<
          decltype(System[SelectedComponent].pos)>::type)
          sycl::malloc_device(Newspace * sizeof(double3), q_ct1);
  (System[SelectedComponent].scale) =
      (typename std::remove_reference<
          decltype(System[SelectedComponent].scale)>::type)
          sycl::malloc_device(Newspace * sizeof(double), q_ct1);
  (System[SelectedComponent].charge) =
      (typename std::remove_reference<
          decltype(System[SelectedComponent].charge)>::type)
          sycl::malloc_device(Newspace * sizeof(double), q_ct1);
  (System[SelectedComponent].scaleCoul) =
      (typename std::remove_reference<
          decltype(System[SelectedComponent].scaleCoul)>::type)
          sycl::malloc_device(Newspace * sizeof(double), q_ct1);
  (System[SelectedComponent].Type) =
      (typename std::remove_reference<
          decltype(System[SelectedComponent].Type)>::type)
          sycl::malloc_device(Newspace * sizeof(size_t), q_ct1);
  (System[SelectedComponent].MolID) =
      (typename std::remove_reference<
          decltype(System[SelectedComponent].MolID)>::type)
          sycl::malloc_device(Newspace * sizeof(size_t), q_ct1);
  // Copy data from temp back to the new pointers
  /*
  DPCT1069:66: The argument 'temp' of the kernel function contains virtual
  pointer(s), which cannot be dereferenced. Try to migrate the code with
  "usm-level=restricted".
  */
  q_ct1.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
      [=](sycl::nd_item<3> item_ct1) {
        AllocateMoreSpace_CopyBack(d_a, temp, Copysize, Newspace,
                                   SelectedComponent);
      });
  //System[SelectedComponent].Allocate_size = Newspace;
}

///////////////////////////////////////////
// FUNCTIONS RELATED TO ACCEPTING A MOVE //
///////////////////////////////////////////

static inline void Update_NumberOfMolecules(Components& SystemComponents, Atoms*& d_a, size_t SelectedComponent, int MoveType)
{
  size_t Molsize = SystemComponents.Moleculesize[SelectedComponent]; //change in atom number counts
  int NumMol = -1; //default: deletion; Insertion: +1, Deletion: -1, size_t is never negative

  switch(MoveType)
  {
    case INSERTION: case SINGLE_INSERTION: case CBCF_INSERTION:
    {
      NumMol = 1;  break;
    }
    case DELETION: case SINGLE_DELETION: case CBCF_DELETION:
    {
      NumMol = -1; break;
    }
  }
  //Update Components
  SystemComponents.NumberOfMolecule_for_Component[SelectedComponent] += NumMol;
  SystemComponents.TotalNumberOfMolecules += NumMol;

  // check System size //
  size_t size = SystemComponents.Moleculesize[SelectedComponent] * SystemComponents.NumberOfMolecule_for_Component[SelectedComponent];

  SystemComponents.UpdatePseudoAtoms(MoveType,  SelectedComponent);

  if(size > SystemComponents.Allocate_size[SelectedComponent])
  {
    AllocateMoreSpace(d_a, SelectedComponent, SystemComponents);
    throw std::runtime_error("Need to allocate more space, not implemented\n");
  }
}

void Update_SINGLE_INSERTION_data(Atoms* d_a, Atoms New, size_t SelectedComponent)
{
  //Assuming single thread//
  size_t Molsize = d_a[SelectedComponent].Molsize;
  size_t UpdateLocation = d_a[SelectedComponent].size;
  for(size_t j = 0; j < Molsize; j++)
  {
    d_a[SelectedComponent].pos[UpdateLocation+j]       = New.pos[j];
    d_a[SelectedComponent].scale[UpdateLocation+j]     = New.scale[j];
    d_a[SelectedComponent].charge[UpdateLocation+j]    = New.charge[j];
    d_a[SelectedComponent].scaleCoul[UpdateLocation+j] = New.scaleCoul[j];
    d_a[SelectedComponent].Type[UpdateLocation+j]      = New.Type[j];
    d_a[SelectedComponent].MolID[UpdateLocation+j]     = New.MolID[j];
  }
  d_a[SelectedComponent].size  += Molsize;
}

void Update_deletion_data(Atoms* d_a, size_t SelectedComponent, size_t UpdateLocation, int Moleculesize, size_t LastLocation,
                          const sycl::nd_item<3> &item_ct1)
{
  size_t i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2);

  //UpdateLocation should be the molecule that needs to be deleted
  //Then move the atom at the last position to the location of the deleted molecule
  //**Zhao's note** MolID of the deleted molecule should not be changed
  //**Zhao's note** if Molecule deleted is the last molecule, then nothing is copied, just change the size later.
  if(UpdateLocation != LastLocation)
  {
    for(size_t i = 0; i < Moleculesize; i++)
    {
      d_a[SelectedComponent].pos[UpdateLocation+i]       = d_a[SelectedComponent].pos[LastLocation+i];
      d_a[SelectedComponent].scale[UpdateLocation+i]     = d_a[SelectedComponent].scale[LastLocation+i];
      d_a[SelectedComponent].charge[UpdateLocation+i]    = d_a[SelectedComponent].charge[LastLocation+i];
      d_a[SelectedComponent].scaleCoul[UpdateLocation+i] = d_a[SelectedComponent].scaleCoul[LastLocation+i];
      d_a[SelectedComponent].Type[UpdateLocation+i]      = d_a[SelectedComponent].Type[LastLocation+i];
    }
  }
  //Zhao's note: the single values in d_a and System are pointing to different locations//
  //d_a is just device (cannot access on host), while System is shared (accessible on host), need to update d_a values here
  //there are two of these values: size and Allocate_size
  if(i==0)
  {
    d_a[SelectedComponent].size  -= Moleculesize; //Zhao's special note: AllData.size doesn't work... So single values are painful, need to consider pointers for single values
  }
}

void Update_insertion_data(Atoms* d_a, Atoms Mol, Atoms NewMol, size_t SelectedTrial, size_t SelectedComponent, size_t UpdateLocation, int Moleculesize,
                           const sycl::nd_item<3> &item_ct1)
{
  size_t i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2);
  //UpdateLocation should be the last position of the dataset
  //Need to check if Allocate_size is smaller than size
  if(Moleculesize == 1) //Only first bead is inserted, first bead data is stored in NewMol
  {
    d_a[SelectedComponent].pos[UpdateLocation]       = NewMol.pos[SelectedTrial];
    d_a[SelectedComponent].scale[UpdateLocation]     = NewMol.scale[SelectedTrial];
    d_a[SelectedComponent].charge[UpdateLocation]    = NewMol.charge[SelectedTrial];
    d_a[SelectedComponent].scaleCoul[UpdateLocation] = NewMol.scaleCoul[SelectedTrial];
    d_a[SelectedComponent].Type[UpdateLocation]      = NewMol.Type[SelectedTrial];
    d_a[SelectedComponent].MolID[UpdateLocation]     = NewMol.MolID[SelectedTrial];
  }
  else //Multiple beads: first bead + trial orientations
  {
    //Update the first bead, first bead data stored in position 0 of Mol //
    d_a[SelectedComponent].pos[UpdateLocation]       = Mol.pos[0];
    d_a[SelectedComponent].scale[UpdateLocation]     = Mol.scale[0];
    d_a[SelectedComponent].charge[UpdateLocation]    = Mol.charge[0];
    d_a[SelectedComponent].scaleCoul[UpdateLocation] = Mol.scaleCoul[0];
    d_a[SelectedComponent].Type[UpdateLocation]      = Mol.Type[0];
    d_a[SelectedComponent].MolID[UpdateLocation]     = Mol.MolID[0];

    size_t chainsize = Moleculesize - 1; // For trial orientations //
    for(size_t j = 0; j < chainsize; j++) //Update the selected orientations//
    {
      size_t selectsize = SelectedTrial*chainsize+j;
      d_a[SelectedComponent].pos[UpdateLocation+j+1]       = NewMol.pos[selectsize];
      d_a[SelectedComponent].scale[UpdateLocation+j+1]     = NewMol.scale[selectsize];
      d_a[SelectedComponent].charge[UpdateLocation+j+1]    = NewMol.charge[selectsize];
      d_a[SelectedComponent].scaleCoul[UpdateLocation+j+1] = NewMol.scaleCoul[selectsize];
      d_a[SelectedComponent].Type[UpdateLocation+j+1]      = NewMol.Type[selectsize];
      d_a[SelectedComponent].MolID[UpdateLocation+j+1]     = NewMol.MolID[selectsize];
    }
  }
  //Zhao's note: the single values in d_a and System are pointing to different locations//
  //d_a is just device (cannot access on host), while System is shared (accessible on host), need to update d_a values here
  //there are two of these values: size and Allocate_size
  if(i==0)
  {
    d_a[SelectedComponent].size  += Moleculesize;
    /*
    for(size_t j = 0; j < Moleculesize; j++)
      printf("xyz: %.5f %.5f %.5f, scale/charge/scaleCoul: %.5f %.5f %.5f, Type: %lu, MolID: %lu\n", d_a[SelectedComponent].pos[UpdateLocation+j].x, d_a[SelectedComponent].pos[UpdateLocation+j].y, d_a[SelectedComponent].pos[UpdateLocation+j].z, d_a[SelectedComponent].scale[UpdateLocation+j], d_a[SelectedComponent].charge[UpdateLocation+j], d_a[SelectedComponent].scaleCoul[UpdateLocation+j], d_a[SelectedComponent].Type[UpdateLocation+j], d_a[SelectedComponent].MolID[UpdateLocation+j]);
    */
  }
}

void update_translation_position(Atoms* d_a, Atoms NewMol, size_t start_position, size_t SelectedComponent,
                                 const sycl::nd_item<3> &item_ct1)
{
  size_t i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2);
  d_a[SelectedComponent].pos[start_position+i] = NewMol.pos[i];
  d_a[SelectedComponent].scale[start_position+i] = NewMol.scale[i];
  d_a[SelectedComponent].charge[start_position+i] = NewMol.charge[i];
  d_a[SelectedComponent].scaleCoul[start_position+i] = NewMol.scaleCoul[i];
}

////////////////////////////////////////////////
// GET PREFACTOR FOR INSERTION/DELETION MOVES //
////////////////////////////////////////////////

static inline double GetPrefactor(Components& SystemComponents, Simulations& Sims, size_t SelectedComponent, int MoveType)
{
  double MolFraction = SystemComponents.MolFraction[SelectedComponent];
  double FugacityCoefficient = SystemComponents.FugacityCoeff[SelectedComponent];
  double NumberOfMolecules = static_cast<double>(SystemComponents.NumberOfMolecule_for_Component[SelectedComponent]);

  //If component has fractional molecule, subtract the number of molecules by 1.//
  if(SystemComponents.hasfractionalMolecule[SelectedComponent]){NumberOfMolecules-=1.0;}
  if(NumberOfMolecules < 0.0) NumberOfMolecules = 0.0;

  double preFactor = 0.0;
  //Hard to generalize the prefactors, when you consider identity swap
  //Certainly you can use insertion/deletion for identity swap, but you may lose accuracies
  //since you are multiplying then dividing by the same variables
  switch(MoveType)
  {
    case INSERTION: case SINGLE_INSERTION:
    {
      preFactor = SystemComponents.Beta * MolFraction * Sims.Box.Pressure * FugacityCoefficient * Sims.Box.Volume / (1.0+NumberOfMolecules);
      break;
    }
    case DELETION: case SINGLE_DELETION:
    {
      preFactor = (NumberOfMolecules) / (SystemComponents.Beta * MolFraction * Sims.Box.Pressure * FugacityCoefficient * Sims.Box.Volume);
      break;
    }
    case IDENTITY_SWAP:
    {
      throw std::runtime_error("Sorry, but no IDENTITY_SWAP OPTION for now. That move uses its own function\n");
    }
    case TRANSLATION: case ROTATION:
    {
      preFactor = 1.0;
    }
  }
  return preFactor;
}

////////////////////////
// ACCEPTION OF MOVES //
////////////////////////

static inline void AcceptInsertion(Components& SystemComponents, Simulations& Sims, size_t SelectedComponent, size_t SelectedTrial, bool noCharges, int MoveType)
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  size_t UpdateLocation = SystemComponents.Moleculesize[SelectedComponent] * SystemComponents.NumberOfMolecule_for_Component[SelectedComponent];
  //printf("AccInsertion, SelectedTrial: %zu, UpdateLocation: %zu\n", SelectedTrial, UpdateLocation);
  //Zhao's note: here needs more consideration: need to update after implementing polyatomic molecule
  if(MoveType == INSERTION)
  {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto SystemComponents_Moleculesize_SelectedComponent_ct6 =
          (int)SystemComponents.Moleculesize[SelectedComponent];

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            Update_insertion_data(
                Sims.d_a, Sims.Old, Sims.New, SelectedTrial, SelectedComponent,
                UpdateLocation,
                SystemComponents_Moleculesize_SelectedComponent_ct6, item_ct1);
          });
    });
  }
  else if(MoveType == SINGLE_INSERTION)
  {
    q_ct1.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
        [=](sycl::nd_item<3> item_ct1) {
          Update_SINGLE_INSERTION_data(Sims.d_a, Sims.New, SelectedComponent);
        });
  }
  Update_NumberOfMolecules(SystemComponents, Sims.d_a, SelectedComponent, INSERTION); //true = Insertion//
  if(!noCharges && SystemComponents.hasPartialCharge[SelectedComponent])
  {
    Update_Ewald_Vector(Sims.Box, false, SystemComponents);
  }
}

static inline void AcceptDeletion(Components& SystemComponents, Simulations& Sims, size_t SelectedComponent, size_t UpdateLocation, size_t SelectedMol, bool noCharges)
{
  size_t LastMolecule = SystemComponents.NumberOfMolecule_for_Component[SelectedComponent]-1;
  size_t LastLocation = LastMolecule*SystemComponents.Moleculesize[SelectedComponent];
  dpct::get_default_queue().submit([&](sycl::handler &cgh) {
    auto SystemComponents_Moleculesize_SelectedComponent_ct3 =
        (int)SystemComponents.Moleculesize[SelectedComponent];

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
        [=](sycl::nd_item<3> item_ct1) {
          Update_deletion_data(
              Sims.d_a, SelectedComponent, UpdateLocation,
              SystemComponents_Moleculesize_SelectedComponent_ct3, LastLocation,
              item_ct1);
        });
  });

  Update_NumberOfMolecules(SystemComponents, Sims.d_a, SelectedComponent, DELETION); //false = Deletion//
  if(!noCharges && SystemComponents.hasPartialCharge[SelectedComponent])
  {
    Update_Ewald_Vector(Sims.Box, false, SystemComponents);
  }

  //Zhao's note: the last molecule can be the fractional molecule, (fractional molecule ID is stored on the host), we need to update it as well (at least check it)//
  //The function below will only be processed if the system has a fractional molecule and the transfered molecule is NOT the fractional one //
  if((SystemComponents.hasfractionalMolecule[SelectedComponent])&&(LastMolecule == SystemComponents.Lambda[SelectedComponent].FractionalMoleculeID))
  {
    //Since the fractional molecule is moved to the place of the selected deleted molecule, update fractional molecule ID on host
    SystemComponents.Lambda[SelectedComponent].FractionalMoleculeID = SelectedMol;
  }
}

//////////////////////////////////////////////////
// PREPARING NEW (TRIAL) LOCATIONS/ORIENTATIONS //
//////////////////////////////////////////////////

void Rotate_Quaternions(double3 &Vec, double3 RANDOM)
{
  //https://stackoverflow.com/questions/31600717/how-to-generate-a-random-quaternion-quickly
  //https://stackoverflow.com/questions/38978441/creating-uniform-random-quaternion-and-multiplication-of-two-quaternions
  //James J. Kuffner 2004//
  //Zhao's note: this one needs three random numbers//
  const double u = RANDOM.x();
  const double v = RANDOM.y();
  const double w = RANDOM.z();
  const double pi=3.14159265358979323846;//Zhao's note: consider using M_PI
  //Sadly, no double4 type available//
  const double q0 = sycl::sqrt(1 - u) * sycl::sin(2 * pi * v);
  const double q1 = sycl::sqrt(1 - u) * sycl::cos(2 * pi * v);
  const double q2 = sycl::sqrt((double)u) * sycl::sin(2 * pi * w);
  const double q3 = sycl::sqrt((double)u) * sycl::cos(2 * pi * w);

  double rot[3*3];
  const double a01=q0*q1; const double a02=q0*q2; const double a03=q0*q3;
  const double a11=q1*q1; const double a12=q1*q2; const double a13=q1*q3;
  const double a22=q2*q2; const double a23=q2*q3; const double a33=q3*q3;

  rot[0]=1.0-2.0*(a22+a33);
  rot[1]=2.0*(a12-a03);
  rot[2]=2.0*(a13+a02);
  rot[3]=2.0*(a12+a03);
  rot[4]=1.0-2.0*(a11+a33);
  rot[5]=2.0*(a23-a01);
  rot[6]=2.0*(a13-a02);
  rot[7]=2.0*(a23+a01);
  rot[8]=1.0-2.0*(a11+a22);
  const double r = Vec.x() * rot[0 * 3 + 0] + Vec.y() * rot[0 * 3 + 1] +
                   Vec.z() * rot[0 * 3 + 2];
  const double s = Vec.x() * rot[1 * 3 + 0] + Vec.y() * rot[1 * 3 + 1] +
                   Vec.z() * rot[1 * 3 + 2];
  const double c = Vec.x() * rot[2 * 3 + 0] + Vec.y() * rot[2 * 3 + 1] +
                   Vec.z() * rot[2 * 3 + 2];
  Vec={r, s, c};
}

void RotationAroundAxis(Atoms Mol,size_t i,double theta, int Axis)
{
  double w,s,c,rot[3*3];

  c = sycl::cos(theta);
  s = sycl::sin(theta);
  switch(Axis)
  {
    case X:
    {
      rot[0*3+0]=1.0; rot[1*3+0]=0.0;  rot[2*3+0]=0.0;
      rot[0*3+1]=0.0; rot[1*3+1]=c;    rot[2*3+1]=-s;
      rot[0*3+2]=0.0; rot[1*3+2]=s;    rot[2*3+2]=c;
      break;
    }
    case Y:
    {
      rot[0*3+0]=c;   rot[1*3+0]=0;    rot[2*3+0]=s;
      rot[0*3+1]=0;   rot[1*3+1]=1.0;  rot[2*3+1]=0;
      rot[0*3+2]=-s;  rot[1*3+2]=0;    rot[2*3+2]=c;
      break;
    }
    case Z:
    {
      rot[0*3+0]=c;   rot[1*3+0]=-s;   rot[2*3+0]=0;
      rot[0*3+1]=s;   rot[1*3+1]=c;    rot[2*3+1]=0;
      rot[0*3+2]=0;   rot[1*3+2]=0;    rot[2*3+2]=1.0;
      break;
    }
    case SELF_DEFINED:  //Zhao's note: not implemented yet//
    {
      break;
    }
  }
  w = Mol.pos[i].x() * rot[0 * 3 + 0] + Mol.pos[i].y() * rot[0 * 3 + 1] +
      Mol.pos[i].z() * rot[0 * 3 + 2];
  s = Mol.pos[i].x() * rot[1 * 3 + 0] + Mol.pos[i].y() * rot[1 * 3 + 1] +
      Mol.pos[i].z() * rot[1 * 3 + 2];
  c = Mol.pos[i].x() * rot[2 * 3 + 0] + Mol.pos[i].y() * rot[2 * 3 + 1] +
      Mol.pos[i].z() * rot[2 * 3 + 2];
  Mol.pos[i].x() = w;
  Mol.pos[i].y() = s;
  Mol.pos[i].z() = c;
}

void get_new_position(Simulations Sim, ForceField FF, size_t start_position,
                      size_t SelectedComponent, double3 *RANDOM, double3 Max,
                      size_t index, int MoveType,
                      const sycl::nd_item<3> &item_ct1)
{
  const size_t i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                   item_ct1.get_local_id(2);
  const size_t real_pos = start_position + i;

  const double3 pos = Sim.d_a[SelectedComponent].pos[real_pos];
  const double  scale     = Sim.d_a[SelectedComponent].scale[real_pos];
  const double  charge    = Sim.d_a[SelectedComponent].charge[real_pos];
  const double  scaleCoul = Sim.d_a[SelectedComponent].scaleCoul[real_pos];
  const size_t  Type      = Sim.d_a[SelectedComponent].Type[real_pos];
  const size_t  MolID     = Sim.d_a[SelectedComponent].MolID[real_pos];

  switch(MoveType)
  {
    case TRANSLATION://TRANSLATION:
    {
      Sim.New.pos[i] = pos + Max * 2.0 * (RANDOM[index] - 0.5);
      Sim.New.scale[i] = scale; Sim.New.charge[i] = charge; Sim.New.scaleCoul[i] = scaleCoul; Sim.New.Type[i] = Type; Sim.New.MolID[i] = MolID;

      Sim.Old.pos[i] = pos;
      Sim.Old.scale[i] = scale; Sim.Old.charge[i] = charge; Sim.Old.scaleCoul[i] = scaleCoul; Sim.Old.Type[i] = Type; Sim.Old.MolID[i] = MolID;
      break;
    }
    case ROTATION://ROTATION:
    {
      Sim.New.pos[i] = pos - Sim.d_a[SelectedComponent].pos[start_position];
      const double3 Angle = Max * 2.0 * (RANDOM[index] - 0.5);

      RotationAroundAxis(Sim.New, i, Angle.x(), X);
      RotationAroundAxis(Sim.New, i, Angle.y(), Y);
      RotationAroundAxis(Sim.New, i, Angle.z(), Z);
      Sim.New.pos[i] += Sim.d_a[SelectedComponent].pos[start_position];

      Sim.New.scale[i] = scale; Sim.New.charge[i] = charge; Sim.New.scaleCoul[i] = scaleCoul; Sim.New.Type[i] = Type; Sim.New.MolID[i] = MolID;

      Sim.Old.pos[i] = pos;
      Sim.Old.scale[i] = scale; Sim.Old.charge[i] = charge; Sim.Old.scaleCoul[i] = scaleCoul; Sim.Old.Type[i] = Type; Sim.Old.MolID[i] = MolID;
      break;
    }
    case SINGLE_INSERTION:
    { 
      //First ROTATION using QUATERNIONS//
      //Then TRANSLATION//
      double3 BoxLength = {Sim.Box.Cell[0], Sim.Box.Cell[4],
                                 Sim.Box.Cell[8]};
      double3 NEW_COM   = BoxLength * RANDOM[index];
      if(i == 0) Sim.New.pos[0] = NEW_COM;
      if(i > 0)
      {
        double3 Vec = pos - Sim.d_a[SelectedComponent].pos[start_position];
        Rotate_Quaternions(Vec, RANDOM[index + 1]);
        Sim.New.pos[i] = Vec + NEW_COM;
      }
      Sim.New.scale[i] = scale;
      Sim.New.charge[i] = charge;
      Sim.New.scaleCoul[i] = scaleCoul;
      Sim.New.Type[i] = Type;
      Sim.New.MolID[i] = Sim.d_a[SelectedComponent].size / Sim.d_a[SelectedComponent].Molsize;
      break;
    }
    case SINGLE_DELETION: //Just Copy the old positions//
    {
      Sim.Old.pos[i] = pos;
      Sim.Old.scale[i] = scale; Sim.Old.charge[i] = charge; Sim.Old.scaleCoul[i] = scaleCoul; Sim.Old.Type[i] = Type; Sim.Old.MolID[i] = MolID;
    }
  }
  Sim.device_flag[i] = false;
}

////////////////////////////////////////////////////////////////////////////////////
// OPTIMIZING THE ACCEPTANCE (MAINTAIN AROUND 0.5) FOR TRANSLATION/ROTATION MOVES //
////////////////////////////////////////////////////////////////////////////////////

static inline void Update_Max_Translation(Move_Statistics MoveStats, Simulations& Sims)
{
  MoveStats.TranslationAccRatio = static_cast<double>(MoveStats.TranslationAccepted)/MoveStats.TranslationTotal;
  //printf("AccRatio is %.10f\n", MoveStats.TranslationAccRatio);
  if(MoveStats.TranslationAccRatio > 0.5)
  {
    Sims.MaxTranslation.x() *= 1.05; Sims.MaxTranslation.y() *= 1.05;
        Sims.MaxTranslation.z() *= 1.05;
  }
  else
  {
    Sims.MaxTranslation.x() *= 0.95; Sims.MaxTranslation.y() *= 0.95;
        Sims.MaxTranslation.z() *= 0.95;
  }
  if (Sims.MaxTranslation.x() < 0.01) Sims.MaxTranslation.x() = 0.01;
  if (Sims.MaxTranslation.y() < 0.01) Sims.MaxTranslation.y() = 0.01;
  if (Sims.MaxTranslation.z() < 0.01) Sims.MaxTranslation.z() = 0.01;

  if (Sims.MaxTranslation.x() > 5.0) Sims.MaxTranslation.x() = 5.0;
  if (Sims.MaxTranslation.y() > 5.0) Sims.MaxTranslation.y() = 5.0;
  if (Sims.MaxTranslation.z() > 5.0) Sims.MaxTranslation.z() = 5.0;
  MoveStats.TranslationAccepted = 0; MoveStats.TranslationTotal=0;
}

static inline void Update_Max_Rotation(Move_Statistics MoveStats, Simulations& Sims)
{
  MoveStats.RotationAccRatio = static_cast<double>(MoveStats.RotationAccepted)/MoveStats.RotationTotal;
  //printf("AccRatio is %.10f\n", MoveStats.TranslationAccRatio);
  if(MoveStats.RotationAccRatio > 0.5)
  {
    Sims.MaxRotation.x() *= 1.05; Sims.MaxRotation.y() *= 1.05;
        Sims.MaxRotation.z() *= 1.05;
  }
  else
  {
    Sims.MaxRotation.x() *= 0.95; Sims.MaxRotation.y() *= 0.95;
        Sims.MaxRotation.z() *= 0.95;
  }
  if (Sims.MaxRotation.x() < 0.01) Sims.MaxRotation.x() = 0.01;
  if (Sims.MaxRotation.y() < 0.01) Sims.MaxRotation.y() = 0.01;
  if (Sims.MaxRotation.z() < 0.01) Sims.MaxRotation.z() = 0.01;

  if (Sims.MaxRotation.x() > 3.14) Sims.MaxRotation.x() = 3.14;
  if (Sims.MaxRotation.y() > 3.14) Sims.MaxRotation.y() = 3.14;
  if (Sims.MaxRotation.z() > 3.14) Sims.MaxRotation.z() = 3.14;
  MoveStats.RotationAccepted = 0; MoveStats.RotationTotal=0;
}
