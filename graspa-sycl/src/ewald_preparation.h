#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <complex>
#include <vector>
//#define M_PI           3.14159265358979323846e0

void Ewald_Total(Boxsize& Box, Atoms*& Host_System, ForceField& FF, Components& SystemComponents, MoveEnergy& E)
{
  printf("****** Calculating Ewald Energy (CPU) ******\n");
  double recip_cutoff = Box.ReciprocalCutOff;
  int kx_max = Box.kmax.x();
  int ky_max = Box.kmax.y();
  int kz_max = Box.kmax.z();
  if(FF.noCharges) return;
  double alpha = Box.Alpha; double alpha_squared = alpha * alpha;
  double prefactor = Box.Prefactor * (2.0 * M_PI / Box.Volume);

  double3 ax = {
      Box.InverseCell[0], Box.InverseCell[3],
      Box.InverseCell[6]}; // printf("ax: %.10f, %.10f, %.10f\n",
                           // Box.InverseCell[0], Box.InverseCell[3],
                           // Box.InverseCell[6]);
  double3 ay = {
      Box.InverseCell[1], Box.InverseCell[4],
      Box.InverseCell[7]}; // printf("ay: %.10f, %.10f, %.10f\n",
                           // Box.InverseCell[1], Box.InverseCell[4],
                           // Box.InverseCell[7]);
  double3 az = {
      Box.InverseCell[2], Box.InverseCell[5],
      Box.InverseCell[8]}; // printf("az: %.10f, %.10f, %.10f\n",
                           // Box.InverseCell[2], Box.InverseCell[5],
                           // Box.InverseCell[8]);

  size_t numberOfAtoms = 0;
  for(size_t i=0; i < SystemComponents.Total_Components; i++) //Skip the first one(framework)
  {
    numberOfAtoms  += SystemComponents.Moleculesize[i] * SystemComponents.NumberOfMolecule_for_Component[i];
  }
  //size_t numberOfWaveVectors = (kx_max + 1) * (2 * ky_max + 1) * (2 * kz_max + 1);
  //Zhao's note: if starting with an empty box, numberOfAtoms = 0, but to allocate space on the GPU, you cannot do zero space for an array//
  //Here, we use 2 * adsorbate_size, since this is the max size gonna be used in the Monte Carlo steps//
  size_t eik_atomsize = *std::max_element(std::begin(SystemComponents.Moleculesize), std::end(SystemComponents.Moleculesize));
  eik_atomsize *= 2;
  if(eik_atomsize < numberOfAtoms) eik_atomsize = numberOfAtoms;
  std::vector<std::complex<double>>eik_x(eik_atomsize * (kx_max + 1));
  std::vector<std::complex<double>>eik_y(eik_atomsize * (ky_max + 1));
  std::vector<std::complex<double>>eik_z(eik_atomsize * (kz_max + 1));
  std::vector<std::complex<double>>eik_xy(eik_atomsize);
  size_t numberOfWaveVectors =
      (Box.kmax.x() + 1) * (2 * Box.kmax.y() + 1) * (2 * Box.kmax.z() + 1);
  std::vector<std::complex<double>>storedEik(numberOfWaveVectors);
  std::vector<std::complex<double>>FrameworkEik(numberOfWaveVectors);
  // Construct exp(ik.r) for atoms and k-vectors kx, ky, kz = 0, 1 explicitly
  size_t count=0;
  for(size_t comp=0; comp < SystemComponents.Total_Components; comp++)
  {
    for(size_t posi=0; posi < SystemComponents.NumberOfMolecule_for_Component[comp] * SystemComponents.Moleculesize[comp]; posi++)
    {
      //determine the component for i
      double3 pos = Host_System[comp].pos[posi];
      eik_x[count + 0 * numberOfAtoms] = std::complex<double>(1.0, 0.0);
      eik_y[count + 0 * numberOfAtoms] = std::complex<double>(1.0, 0.0);
      eik_z[count + 0 * numberOfAtoms] = std::complex<double>(1.0, 0.0);
      double3 s; matrix_multiply_by_vector(Box.InverseCell, pos, s);
      s*=2 * M_PI;
      eik_x[count + 1 * numberOfAtoms] = std::complex<double>(std::cos(s.x()), std::sin(s.x()));
      eik_y[count + 1 * numberOfAtoms] = std::complex<double>(std::cos(s.y()), std::sin(s.y()));
      eik_z[count + 1 * numberOfAtoms] = std::complex<double>(std::cos(s.z()), std::sin(s.z()));
      count++;
    }
  }
  // Calculate remaining positive kx, ky and kz by recurrence
  for(size_t kx = 2; kx <= kx_max; ++kx)
  {
    for(size_t i = 0; i != numberOfAtoms; ++i)
    {
      eik_x[i + kx * numberOfAtoms] = eik_x[i + (kx - 1) * numberOfAtoms] * eik_x[i + 1 * numberOfAtoms];
    }
  }
  for(size_t ky = 2; ky <= ky_max; ++ky)
  {
    for(size_t i = 0; i != numberOfAtoms; ++i)
    {
      eik_y[i + ky * numberOfAtoms] = eik_y[i + (ky - 1) * numberOfAtoms] * eik_y[i + 1 * numberOfAtoms];
    }
  }
  for(size_t kz = 2; kz <= kz_max; ++kz)
  {
    for(size_t i = 0; i != numberOfAtoms; ++i)
    {
      eik_z[i + kz * numberOfAtoms] = eik_z[i + (kz - 1) * numberOfAtoms] * eik_z[i + 1 * numberOfAtoms];
    }
  }
  size_t nvec = 0;
  //for debugging
  double FrameworkEwald = 0.0; //Framework-Framework interaction//
  size_t kxcount = 0; size_t kycount = 0; size_t kzcount = 0; size_t kzinactive = 0;
  for(std::make_signed_t<std::size_t> kx = 0; kx <= kx_max; ++kx)
  {
    double3 kvec_x = ax * 2.0 * M_PI * static_cast<double>(kx);
    // Only positive kx are used, the negative kx are taken into account by the factor of two
    double factor = (kx == 0) ? (1.0 * prefactor) : (2.0 * prefactor);
    
    for(std::make_signed_t<std::size_t> ky = -ky_max; ky <= ky_max; ++ky)
    {
      double3 kvec_y = ay * 2.0 * M_PI * static_cast<double>(ky);
      // Precompute and store eik_x * eik_y outside the kz-loop
      for(size_t i = 0; i != numberOfAtoms; ++i)
      {
        std::complex<double> eiky_temp = eik_y[i + numberOfAtoms * static_cast<size_t>(std::abs(ky))];
        eiky_temp.imag(ky>=0 ? eiky_temp.imag() : -eiky_temp.imag());
        eik_xy[i] = eik_x[i + numberOfAtoms * static_cast<size_t>(kx)] * eiky_temp;
      }

      for(std::make_signed_t<std::size_t> kz = -kz_max; kz <= kz_max; ++kz)
      {
        // Ommit kvec==0
        size_t ksqr = kx * kx + ky * ky + kz * kz;
        std::complex<double> cksum(0.0, 0.0);
        std::complex<double> Frameworkck(0.0, 0.0);
        if((ksqr != 0) && (static_cast<double>(ksqr) < recip_cutoff))
        {
          double3 kvec_z = az * 2.0 * M_PI * static_cast<double>(kz);
          //std::complex<double> cksum(0.0, 0.0);
          count=0;
          for(size_t comp=0; comp<SystemComponents.Total_Components; comp++)
          {
            for(size_t posi=0; posi<SystemComponents.NumberOfMolecule_for_Component[comp]*SystemComponents.Moleculesize[comp]; posi++)
            {
              std::complex<double> eikz_temp = eik_z[count + numberOfAtoms * static_cast<size_t>(std::abs(kz))];
              eikz_temp.imag(kz>=0 ? eikz_temp.imag() : -eikz_temp.imag());
              double charge = Host_System[comp].charge[posi];
              double scaling = Host_System[comp].scaleCoul[posi];
              cksum += scaling * charge * (eik_xy[count] * eikz_temp);
              if(comp == 0 && SystemComponents.NumberOfFrameworks > 0) Frameworkck += scaling * charge * (eik_xy[count] * eikz_temp);
              count++;
            }
          }
          double3 tempkvec = kvec_x + kvec_y + kvec_z;
          double  rksq = dot(tempkvec, tempkvec);
          double  temp = factor * std::exp((-0.25 / alpha_squared) * rksq) / rksq;
          if(SystemComponents.NumberOfFrameworks > 0 && Box.ExcludeHostGuestEwald)
            cksum -= Frameworkck;
          double tempsum = temp * (cksum.real() * cksum.real() + cksum.imag() * cksum.imag());
          double tempFramework = 0.0; double tempFrameworkGuest = 0.0;

          tempFramework = temp * (Frameworkck.real()      * Frameworkck.real() + Frameworkck.imag() * Frameworkck.imag());
          tempFrameworkGuest = temp * (Frameworkck.real() * cksum.real()       + Frameworkck.imag() * cksum.imag()) * 2.0;
          E.EwaldE   += tempsum;
          FrameworkEwald  += tempFramework;
          E.HGEwaldE += tempFrameworkGuest;
          //storedEik[nvec] = cksum;
          //++nvec;
          kzcount++;
        }
        FrameworkEik[nvec] = Frameworkck;
        storedEik[nvec]    = cksum;
        /*
        if(Box.ExcludeHostGuestEwald) 
        {
          storedEik[nvec] -= FrameworkEik[nvec]; //exclude Framework contribution in Eik, this will also exclude it on the GPU and kernel functions
        }
        */
        ++nvec;
        kzinactive++;
      }
      kycount++;
    }
    kxcount++;
  }

  printf("Total ewaldE Fourier: %.10f, Framework Fourier: %.10f, Framework-Guest Fourier: %.10f\n", E.EwaldE, FrameworkEwald, E.HGEwaldE);
  if(Box.ExcludeHostGuestEwald) E.EwaldE += FrameworkEwald;

  // Subtract self-energy
  double prefactor_self = Box.Prefactor * alpha / std::sqrt(M_PI);
  count=0;
  for(size_t comp=0; comp<SystemComponents.Total_Components; comp++)
  {
    double SelfE = 0.0;
    for(size_t posi=0; posi<SystemComponents.NumberOfMolecule_for_Component[comp]*SystemComponents.Moleculesize[comp]; posi++)
    {
      double charge  = Host_System[comp].charge[posi];
      double scaling = Host_System[comp].scaleCoul[posi];
      E.EwaldE -= prefactor_self * scaling * charge * scaling * charge;
      SelfE         += prefactor_self * scaling * charge * scaling * charge;
      if(comp == 0 && SystemComponents.NumberOfFrameworks > 0) FrameworkEwald -= prefactor_self * scaling * charge * scaling * charge;
    }
    printf("Component: %zu, SelfAtomE: %.5f\n", comp, SelfE);
  }

  // Subtract exclusion-energy, Zhao's note: taking out the pairs of energies that belong to the same molecule
  size_t j_count = 0;
  for(size_t l = 0; l != SystemComponents.Total_Components; ++l)
  {
    double exclusionE = 0.0;  
    //printf("Exclusion on component %zu, size: %zu\n", l, Host_System[l].size);
    if(Host_System[l].size != 0)
    {
      for(size_t mol = 0; mol != SystemComponents.NumberOfMolecule_for_Component[l]; mol++)
      {
        size_t AtomID = mol * SystemComponents.Moleculesize[l];
        for(size_t i = AtomID; i != AtomID + SystemComponents.Moleculesize[l] - 1; i++)
        {
          double  factorA = Host_System[l].scaleCoul[i] * Host_System[l].charge[i];
          double3 posA = Host_System[l].pos[i];
          for(size_t j = i + 1; j != AtomID + SystemComponents.Moleculesize[l]; j++)
          {
            double  factorB = Host_System[l].scaleCoul[j] * Host_System[l].charge[j];
            double3 posB = Host_System[l].pos[j];

            double3 posvec = posA - posB;
            PBC(posvec, Box.Cell, Box.InverseCell, Box.Cubic);
            double rr_dot = dot(posvec, posvec);
            double r = std::sqrt(rr_dot);

            E.EwaldE -= Box.Prefactor * factorA * factorB * std::erf(alpha * r) / r;
            exclusionE    -= Box.Prefactor * factorA * factorB * std::erf(alpha * r) / r;
            if(l == 0 && SystemComponents.NumberOfFrameworks > 0) FrameworkEwald -= Box.Prefactor * factorA * factorB * std::erf(alpha * r) / r;
            j_count++;
          }
        }
      }
    }
    printf("Component: %zu, Intra-Molecular ExclusionE: %.5f\n", l, exclusionE);
  }
  printf("Framework Ewald: %.10f\n", FrameworkEwald);
  SystemComponents.FrameworkEwald = FrameworkEwald;

  /*
  if(Box.ExcludeHostGuestEwald)
    E -= FrameworkEwald;
  */
  //Record the values for the Ewald Vectors//
//  for(size_t i = 0; i < eik_xy.size(); i++)
    SystemComponents.eik_xy       = eik_xy;
//  for(size_t i = 0; i < eik_x.size(); i++)
    SystemComponents.eik_x        = eik_x;
//  for(size_t i = 0; i < eik_y.size(); i++)
    SystemComponents.eik_y        = eik_y;
//  for(size_t i = 0; i < eik_z.size(); i++)
    SystemComponents.eik_z        = eik_z;
//  for(size_t i = 0; i < storedEik.size(); i++)
    SystemComponents.storedEik    = storedEik;

    SystemComponents.FrameworkEik = FrameworkEik;
    //IF RUN ON CPU, check totalEik vs. storedEik//
    /*
    if(SystemComponents.totalEik.size() > 0)
    {
      for(size_t i = 0; i < storedEik.size(); i++)
      {
        if((SystemComponents.storedEik[i].real() != SystemComponents.totalEik[i].real()) || (SystemComponents.storedEik[i].imag() != SystemComponents.totalEik[i].imag()))
        {
          printf("element %zu: stored: %.15f %.15f, updated: %.15f %.15f\n", i, storedEik[i].real(), storedEik[i].imag(), SystemComponents.totalEik[i].real(), SystemComponents.totalEik[i].imag());
        }
      }
    }
    */
}

double Calculate_Intra_Molecule_Exclusion(Boxsize& Box, Atoms* System, double alpha, double Prefactor, Components& SystemComponents, size_t SelectedComponent)
{
  double E = 0.0;
  if(SystemComponents.Moleculesize[SelectedComponent] == 0) return 0.0;
  for(size_t i = 0; i != SystemComponents.Moleculesize[SelectedComponent] - 1; i++)
  {
    double  factorA = System[SelectedComponent].scaleCoul[i] * System[SelectedComponent].charge[i];
    double3 posA = System[SelectedComponent].pos[i];
    for(size_t j = i + 1; j != SystemComponents.Moleculesize[SelectedComponent]; j++)
    {
      double  factorB = System[SelectedComponent].scaleCoul[j] * System[SelectedComponent].charge[j];
      double3 posB = System[SelectedComponent].pos[j];

      double3 posvec = posA - posB;
      PBC(posvec, Box.Cell, Box.InverseCell, Box.Cubic);
      double rr_dot = dot(posvec, posvec);
      double r = std::sqrt(rr_dot);

      E += Prefactor * factorA * factorB * std::erf(alpha * r) / r;
    }
  }
  printf("Component %zu, Intra Exclusion Energy: %.5f\n", SelectedComponent, E);
  return E;
}

double Calculate_Self_Exclusion(Boxsize& Box, Atoms* System, double alpha, double Prefactor, Components& SystemComponents, size_t SelectedComponent)
{
  double E = 0.0; double prefactor_self = Prefactor * alpha / std::sqrt(M_PI);
  if(SystemComponents.Moleculesize[SelectedComponent] == 0) return 0.0;
  for(size_t i=0; i<SystemComponents.Moleculesize[SelectedComponent]; i++)
  {
    double charge = System[SelectedComponent].charge[i];
    double scaling = System[SelectedComponent].scaleCoul[i];
    E += prefactor_self * scaling * charge * scaling * charge;
  }
  printf("Component %zu, Atom Self Exclusion Energy: %.5f\n", SelectedComponent, E);
  return E;
}

void Check_WaveVector_CPUGPU(Boxsize& Box, Components& SystemComponents)
{
  printf(" ****** CHECKING WaveVectors Stored on CPU vs. GPU ****** \n");
  size_t numberOfWaveVectors =
      (Box.kmax.x() + 1) * (2 * Box.kmax.y() + 1) * (2 * Box.kmax.z() + 1);
  Complex GPUWV[numberOfWaveVectors];
  dpct::get_default_queue()
      .memcpy(GPUWV, Box.storedEik, numberOfWaveVectors * sizeof(Complex))
      .wait();
  size_t numWVCPU            = SystemComponents.storedEik.size();
  if(numberOfWaveVectors != numWVCPU) printf("ERROR: Number of CPU WaveVectors does NOT EQUAL to the GPU one!!!");
  size_t counter = 0;
  for(size_t i = 0; i < numberOfWaveVectors; i++)
  {
    double diff_real = abs(SystemComponents.storedEik[i].real() - GPUWV[i].real);
    double diff_imag = abs(SystemComponents.storedEik[i].imag() - GPUWV[i].imag);
    if(i < 10)
      printf("Wave Vector %zu, CPU: %.5f %.5f, GPU: %.5f %.5f\n", i, SystemComponents.storedEik[i].real(), SystemComponents.storedEik[i].imag(), GPUWV[i].real, GPUWV[i].imag);
    if(diff_real > 1e-10 || diff_imag > 1e-10)
    {
      counter++;
      if(counter < 10) printf("There is a difference in GPU/CPU WaveVector at position %zu: CPU: %.5f %.5f, GPU: %.5f %.5f\n", i, SystemComponents.storedEik[i].real(), SystemComponents.storedEik[i].imag(), GPUWV[i].real, GPUWV[i].imag);
    }
  }
  if(counter >= 10) printf("More than 10 WaveVectors mismatch.\n");
  //Also check Framework Eik vectors//
  printf(" ****** CHECKING Framework WaveVectors Stored on CPU ****** \n");
  for(size_t i = 0; i < numberOfWaveVectors; i++)
  {
    if(i < 10) printf("Framework Wave Vector %zu, real: %.5f imag: %.5f\n", i, SystemComponents.FrameworkEik[i].real(), SystemComponents.FrameworkEik[i].imag());
  }
}

void CPU_GPU_EwaldTotalEnergy(Boxsize& Box, Boxsize& device_Box, Atoms* System, Atoms* d_a, ForceField FF, ForceField device_FF, Components& SystemComponents, MoveEnergy& E)
{
  ///////////////////
  // Run CPU Ewald //
  ///////////////////
  double start = omp_get_wtime();
  Ewald_Total(Box, System, FF, SystemComponents, E);
  double end = omp_get_wtime(); double CPU_ewald_time = end-start;
  printf("HostEwald took %.5f sec\n", CPU_ewald_time);
}

void Calculate_Exclusion_Energy_Rigid(Boxsize& Box, Atoms* System, ForceField FF, Components& SystemComponents)
{
  for(size_t i = 0; i < SystemComponents.Total_Components; i++)
  {
    double IntraE = 0.0; double SelfE = 0.0;
    if(SystemComponents.rigid[i]) //Only Calculate this when the component is rigid//
    {
      IntraE = Calculate_Intra_Molecule_Exclusion(Box, System, Box.Alpha, Box.Prefactor, SystemComponents, i);
      SelfE  = Calculate_Self_Exclusion(Box, System, Box.Alpha, Box.Prefactor, SystemComponents, i);
    }
    SystemComponents.ExclusionIntra.push_back(IntraE);
    SystemComponents.ExclusionAtom.push_back(SelfE);
    printf("DEBUG: comp: %zu, IntraE: %.5f, SelfE: %.5f\n", i, SystemComponents.ExclusionIntra[i], SystemComponents.ExclusionAtom[i]);
  }

}
