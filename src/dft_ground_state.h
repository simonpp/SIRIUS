#ifndef __DFT_GROUND_STATE_H__
#define __DFT_GROUND_STATE_H__

/** \file dft_ground_state.h

    \brief Contains definition and partial implementation of sirius::DFT_ground_state class.
*/

/** \page DFT Spin-polarized DFT
    \section section1 Preliminary notes

    \note Here and below sybol \f$ \sigma \f$ is reserved for the Pauli matrices. Spin components are labeled with \f$ \alpha \f$ or \f$ \beta\f$.

    Wave-function of spin-1/2 particle is a two-component spinor:
    \f[
        {\bf \varphi}({\bf r})=\left( \begin{array}{c} \varphi_1({\bf r}) \\ \varphi_2({\bf r}) \end{array} \right)
    \f]
    Operator of spin:
    \f[
        {\bf \hat S}=\frac{\hbar}{2}{\bf \sigma},
    \f]
*/
namespace sirius
{

class DFT_ground_state
{
    private:

        Global& parameters_;

        Potential* potential_;

        Density* density_;

        K_set* kset_;

        double ewald_energy_;

        double ewald_energy()
        {
            Timer t("sirius::DFT_ground_state::ewald_energy");

            double alpha = 1.5;
            
            double ewald_g = 0;

            for (int ig = 0; ig < parameters_.reciprocal_lattice()->num_gvec(); ig++)
            {
                complex16 rho(0, 0);
                for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
                {
                    rho += parameters_.reciprocal_lattice()->gvec_phase_factor<global>(ig, ia) * 
                           double(parameters_.unit_cell()->atom(ia)->zn());
                }
                double g2 = pow(parameters_.reciprocal_lattice()->gvec_len(ig), 2);
                if (ig)
                {
                    ewald_g += pow(abs(rho), 2) * exp(-g2 / 4 / alpha) / g2;
                }
                else
                {
                    ewald_g -= pow(parameters_.unit_cell()->num_electrons(), 2) / alpha / 4; // constant term in QE comments
                }
            }
            ewald_g *= (twopi / parameters_.unit_cell()->omega());

            // remove self-interaction
            for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
            {
                ewald_g -= sqrt(alpha / pi) * pow(parameters_.unit_cell()->atom(ia)->zn(), 2);
            }

            double ewald_r = 0;
            for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
            {
                for (int i = 1; i < parameters_.unit_cell()->num_nearest_neighbours(ia); i++)
                {
                    int ja = parameters_.unit_cell()->nearest_neighbour(i, ia).atom_id;
                    double d = parameters_.unit_cell()->nearest_neighbour(i, ia).distance;
                    ewald_r += 0.5 * parameters_.unit_cell()->atom(ia)->zn() * parameters_.unit_cell()->atom(ja)->zn() * 
                               gsl_sf_erfc(sqrt(alpha) * d) / d;
                }
            }

            return (ewald_g + ewald_r);
        }

    public:

        DFT_ground_state(Global& parameters__, Potential* potential__, Density* density__, K_set* kset__) 
            : parameters_(parameters__), 
              potential_(potential__), 
              density_(density__), 
              kset_(kset__)
        {
            if (parameters_.potential_type() == ultrasoft_pseudopotential) ewald_energy_ = ewald_energy();
        }

        void move_atoms(int istep);

        void forces(mdarray<double, 2>& atom_force);

        void scf_loop(double potential_tol, double energy_tol, int num_dft_iter);

        void relax_atom_positions();

        void update();

        void print_info()
        {
            double evalsum1 = kset_->valence_eval_sum();
            double evalsum2 = core_eval_sum();
            double ekin = energy_kin();
            double evxc = energy_vxc();
            double eexc = energy_exc();
            double ebxc = energy_bxc();
            double evha = energy_vha();
            double etot = total_energy();
            double gap = kset_->band_gap() * ha2ev;
            double ef = kset_->energy_fermi();
            double core_leak = density_->core_leakage();
            
            if (Platform::mpi_rank() == 0)
            {
                printf("\n");
                printf("Energy\n");
                for (int i = 0; i < 80; i++) printf("-");
                printf("\n"); 

                printf("valence_eval_sum          : %18.8f\n", evalsum1);
                if (parameters_.potential_type() == full_potential)
                {
                    printf("core_eval_sum             : %18.8f\n", evalsum2);
                    printf("kinetic energy            : %18.8f\n", ekin);
                }
                printf("<rho|V^{XC}>              : %18.8f\n", evxc);
                printf("<rho|E^{XC}>              : %18.8f\n", eexc);
                printf("<mag|B^{XC}>              : %18.8f\n", ebxc);
                printf("<rho|V^{H}>               : %18.8f\n", evha);
                if (parameters_.potential_type() == ultrasoft_pseudopotential)
                {
                    printf("one-electron contribution : %18.8f\n", evalsum1 - (evxc + evha)); // eband + deband in QE
                    printf("hartree contribution      : %18.8f\n", 0.5 * evha);
                    printf("xc contribution           : %18.8f\n", eexc);
                    printf("ewald contribution        : %18.8f\n", ewald_energy_);
                }
                printf("Total energy              : %18.8f\n", etot);

                printf("\n");
                printf("band gap (eV) : %18.8f\n", gap);
                printf("Efermi        : %18.8f\n", ef);
                printf("\n");
                if (parameters_.potential_type() == full_potential) printf("core leakage : %18.8f\n", core_leak);
            }
        }

        /// Return nucleus energy in the electrostatic field.
        /** Compute energy of nucleus in the electrostatic potential generated by the total (electrons + nuclei) 
            charge density. Diverging self-interaction term z*z/|r=0| is excluded. */
        inline double energy_enuc();
        
        /// Return eigen-value sum of core states.
        inline double core_eval_sum();
        
        inline double energy_vha()
        {
            return inner(parameters_, density_->rho(), potential_->coulomb_potential());
        }
        
        inline double energy_vxc()
        {
            return inner(parameters_, density_->rho(), potential_->xc_potential());
        }
        
        inline double energy_exc()
        {
            double exc = inner(parameters_, density_->rho(), potential_->xc_energy_density());
            if (parameters_.potential_type() == ultrasoft_pseudopotential) 
                exc += inner(parameters_, density_->rho_pseudo_core(), potential_->xc_energy_density());
            return exc;
        }

        inline double energy_bxc()
        {
            double ebxc = 0.0;
            for (int j = 0; j < parameters_.num_mag_dims(); j++) 
                ebxc += inner(parameters_, density_->magnetization(j), potential_->effective_magnetic_field(j));
            return ebxc;
        }

        inline double energy_veff()
        {
            return inner(parameters_, density_->rho(), potential_->effective_potential());
        }

        /// Full eigen-value sum (core + valence)
        inline double eval_sum()
        {
            return (core_eval_sum() + kset_->valence_eval_sum());
        }
        
        /// Kinetic energy
        /** more doc here
        */
        inline double energy_kin()
        {
            return (eval_sum() - energy_veff() - energy_bxc());
        }

        /// Total energy of the electronic subsystem.
        /** From the definition of the density functional we have:
            
            \f[
                E[\rho] = T[\rho] + E^{H}[\rho] + E^{XC}[\rho] + E^{ext}[\rho]
            \f]
            where \f$ T[\rho] \f$ is the kinetic energy, \f$ E^{H}[\rho] \f$ - electrostatic energy of
            electron-electron density interaction, \f$ E^{XC}[\rho] \f$ - exchange-correlation energy
            and \f$ E^{ext}[\rho] \f$ - energy in the external field of nuclei.
            
            Electrostatic and external field energies are grouped in the following way:
            \f[
                \frac{1}{2} \int \int \frac{\rho({\bf r})\rho({\bf r'}) d{\bf r} d{\bf r'}}{|{\bf r} - {\bf r'}|} + 
                    \int \rho({\bf r}) V^{nuc}({\bf r}) d{\bf r} = \frac{1}{2} \int V^{H}({\bf r})\rho({\bf r})d{\bf r} + 
                    \frac{1}{2} \int \rho({\bf r}) V^{nuc}({\bf r}) d{\bf r}
            \f]
            Here \f$ V^{H}({\bf r}) \f$ is the total (electron + nuclei) electrostatic potential returned by the 
            poisson solver. Next we transform the remaining term:
            \f[
                \frac{1}{2} \int \rho({\bf r}) V^{nuc}({\bf r}) d{\bf r} = 
                \frac{1}{2} \int \int \frac{\rho({\bf r})\rho^{nuc}({\bf r'}) d{\bf r} d{\bf r'}}{|{\bf r} - {\bf r'}|} = 
                \frac{1}{2} \int V^{H,el}({\bf r}) \rho^{nuc}({\bf r}) d{\bf r}
            \f]
        */
        inline double total_energy()
        {
            switch (parameters_.potential_type())
            {
                case full_potential:
                {
                    return (energy_kin() + energy_exc() + 0.5 * energy_vha() + energy_enuc());
                }
                case ultrasoft_pseudopotential:
                {
                    return (kset_->valence_eval_sum() - (energy_vxc() + energy_vha()) + 0.5 * energy_vha() + 
                            energy_exc() + ewald_energy_);
                }
                default:
                {
                    stop_here
                }
            }
            return 0; // make compiler happy
        }
};

#include "dft_ground_state.hpp"

};

#endif // __DFT_GROUND_STATE_H__

