
namespace sirius {

// generated by the following Mathematica expression: 
//     Do[Print[Table[CForm[N[ Gamma[5/2+l+n]/ Gamma[3/2+l],20]],{n,0,10}],","],{l,0,30}]
/** \todo Think about the exact expression and how to avoid overflowing. */
const double gamma_factors[31][11] = {
{1.5,3.75,13.125,59.0625,324.84375,2111.484375,15836.1328125,134607.12890625,1.278767724609375e6,1.34270611083984375e7,1.5441120274658203125e8},
{2.5,8.75,39.375,216.5625,1407.65625,10557.421875,89738.0859375,852511.81640625,8.951374072265625e6,1.029408018310546875e8,1.2867600228881835938e9},
{3.5,15.75,86.625,563.0625,4222.96875,35895.234375,341004.7265625,3.58054962890625e6,4.1176320732421875e7,5.147040091552734375e8,6.9485041235961914062e9},
{4.5,24.75,160.875,1206.5625,10255.78125,97429.921875,1.0230141796875e6,1.176466306640625e7,1.47058288330078125e8,1.9852868924560546875e9,2.8786659940612792969e10},
{5.5,35.75,268.125,2279.0625,21651.09375,227336.484375,2.6143695703125e6,3.267961962890625e7,4.41174864990234375e8,6.3970355423583984375e9,9.9154050906555175781e10},
{6.5,48.75,414.375,3936.5625,41333.90625,475339.921875,5.9417490234375e6,8.021361181640625e7,1.163097371337890625e9,1.8028009255737304687e10,2.9746215271966552734e11},
{7.5,63.75,605.625,6359.0625,73129.21875,914115.234375,1.23405556640625e7,1.7893805712890625e8,2.773539885498046875e9,4.5763408110717773438e10,8.0085964193756103516e11},
{8.5,80.75,847.875,9750.5625,121882.03125,1.645407421875e6,2.38584076171875e7,3.6980531806640625e8,6.101787748095703125e9,1.0678128559167480469e11,1.9754537834459838867e12},
{9.5,99.75,1147.125,14339.0625,193577.34375,2.806871484375e6,4.35065080078125e7,7.1785738212890625e8,1.2562504187255859375e10,2.3240632746423339844e11,4.5319233855525512695e12},
{10.5,120.75,1509.375,20376.5625,295460.15625,4.579632421875e6,7.55639349609375e7,1.32236886181640625e9,2.4463823943603515625e10,4.7704456690026855469e11,9.7794136214555053711e12},
{11.5,143.75,1940.625,28139.0625,436155.46875,7.196565234375e6,1.259398916015625e8,2.32988799462890625e9,4.5432815895263671875e10,9.3137272585290527344e11,2.0024513605837463379e13},
{12.5,168.75,2446.875,37926.5625,625788.28125,1.0951294921875e7,2.025989560546875e8,3.95067964306640625e9,8.0988932682861328125e10,1.7412620526815185547e12,3.917839618533416748e13},
{13.5,195.75,3034.125,50063.0625,876103.59375,1.6207916484375e7,3.160543714453125e8,6.47911461462890625e9,1.3930096421452148438e11,3.1342716948267333984e12,7.3655384828428234863e13},
{14.5,224.75,3708.375,64896.5625,1.20058640625e6,2.3411434921875e7,4.799344158984375e8,1.031858994181640625e10,2.3216827369086914063e11,5.4559544317354248047e12,1.3367088357751790771e14},
{15.5,255.75,4475.625,82799.0625,1.61458171875e6,3.3098925234375e7,7.116268925390625e8,1.601160508212890625e10,3.7627271943002929687e11,9.2186816260357177734e12,2.3507638146391080322e14},
{16.5,288.75,5341.875,104166.5625,2.13541453125e6,4.5911412421875e7,1.0330067794921875e9,2.427565931806640625e10,5.9475365329262695312e11,1.5166218158961987305e13,4.0190478121249266357e14},
{17.5,323.75,6313.125,129419.0625,2.78250984375e6,6.2606471484375e7,1.4712520798828125e9,3.604567595712890625e10,9.1916473690678710938e11,2.4357865528029858398e13,6.6984130202082110596e14},
{18.5,360.75,7395.375,159000.5625,3.57751265625e6,8.4071547421875e7,2.0597529118359375e9,5.252369925181640625e10,1.3918780301731347656e12,3.8276645829761206055e13,1.0908844061481943726e15},
{19.5,399.75,8594.625,193379.0625,4.54440796875e6,1.11337995234375e8,2.8391188784765625e9,7.523665027962890625e10,2.0690078826897949219e12,5.8966724656659155273e13,1.7395183773714450806e15},
{20.5,440.75,9916.875,233046.5625,5.70964078125e6,1.45595839921875e8,3.8582897579296875e9,1.0610296834306640625e11,3.0239345977773925781e12,8.9206070634433081055e13,2.7207851543502089722e15},
{21.5,483.75,11368.125,278519.0625,7.10223609375e6,1.88209256484375e8,5.1757545533203125e9,1.4750900476962890625e11,4.3515156407040527344e12,1.327212270414736084e14,4.1807186518064186646e15},
{22.5,528.75,12954.375,330336.5625,8.75391890625e6,2.40732769921875e8,6.8608839427734375e9,2.0239607631181640625e11,6.1730803275104003906e12,1.944520303165776123e14,6.3196909852887723999e15},
{23.5,575.75,14681.625,389063.0625,1.069923421875e7,3.04928175234375e8,8.9953811694140625e9,2.7435912566712890625e11,8.6423124585145605469e12,2.8087515490172321777e14,9.4093176892077277954e15},
{24.5,624.75,16555.875,455286.5625,1.297566703125e7,3.82782177421875e8,1.16748564113671875e10,3.6775797695806640625e11,1.1952134251137158203e13,4.003964974130947998e14,1.3813679160751770593e16},
{25.5,675.75,18583.125,529619.0625,1.562376234375e7,4.76524751484375e8,1.50105296717578125e10,4.8784221433212890625e11,1.6342714180126318359e13,5.638236392143579834e14,2.0015739192109708411e16},
{26.5,728.75,20769.375,612696.5625,1.868724515625e7,5.88648222421875e8,1.91310672287109375e10,6.4089075216181640625e11,2.2110730949582666016e13,7.8493094871018464355e14,2.864997962792173949e16},
{27.5,783.75,23120.625,705179.0625,2.221314046875e7,7.21927065234375e8,2.41845566853515625e10,8.3436720564462890625e11,2.9620035800384326172e13,1.0811313067140279053e15,4.0542424001776046448e16},
{28.5,840.75,25642.875,807750.5625,2.625189328125e7,8.79438424921875e8,3.03406256598046875e10,1.0770922109230664062e12,3.9313865698691923828e13,1.4742699637009471436e15,5.6759393602486465027e16},
{29.5,899.75,28342.125,921119.0625,3.085748859375e7,1.064583356484375e9,3.77927091551953125e10,1.3794338841646289062e12,5.1728770656173583984e13,1.9915576702626829834e15,7.8666527975375977844e16},
{30.5,960.75,31224.375,1.0460165625e6,3.608757140625e7,1.281108784921875e9,4.67604706496484375e10,1.7535176493618164062e12,6.7510429500429931641e13,2.6666619652669822998e15,1.0799980959331278314e17},
{31.5,1023.75,34295.625,1.1831990625e6,4.200356671875e7,1.533130185234375e9,5.74923819462890625e10,2.2134567049321289062e12,8.7431539844819091797e13,3.5409773637151732178e15,1.4695056059417968854e17}};

/// Generate effective potential from charge density and magnetization
/** \note At some point we need to update the atomic potential with the new MT potential. This is simple if the 
          effective potential is a global function. Otherwise we need to pass the effective potential between MPI ranks.
          This is also simple, but requires some time. It is also easier to mix the global functions.
*/
class Potential 
{
    private:
        
        Global& parameters_;

        Periodic_function<double>* effective_potential_;
        
        Periodic_function<double>* effective_magnetic_field_[3];
 
        Periodic_function<double>* coulomb_potential_;
        Periodic_function<double>* xc_potential_;
        Periodic_function<double>* xc_energy_density_;
        
        mdarray<double, 3> sbessel_mom_;

        mdarray<double, 3> sbessel_mt_;

        int lmax_;
        
        SHT* sht_;

        int pseudo_density_order;

        std::vector<complex16> zil_;
        
        std::vector<complex16> zilm_;

        std::vector<int> l_by_lm_;

        /// Compute MT part of the potential and MT multipole moments
        void poisson_vmt(mdarray<Spheric_function<complex16>*, 1>& rho_ylm, mdarray<Spheric_function<complex16>*, 1>& vh, 
                         mdarray<complex16, 2>& qmt);

        /// Compute multipole momenst of the interstitial charge density
        /** Also, compute the MT boundary condition 
        */
        void poisson_sum_G(complex16* fpw, mdarray<double, 3>& fl, mdarray<complex16, 2>& flm);
        
        /// Compute contribution from the pseudocharge to the plane-wave expansion
        void poisson_pw(mdarray<complex16, 2>& qmt, mdarray<complex16, 2>& qit, complex16* pseudo_pw);

    public:

        /// Constructor
        Potential(Global& parameters__);

        ~Potential();

        void update();

        void set_effective_potential_ptr(double* veffmt, double* veffit);
        
        void set_effective_magnetic_field_ptr(double* beffmt, double* beffit);
         
        /// Zero effective potential and magnetic field.
        void zero();

        /// Poisson solver
        /** Plane wave expansion
            \f[
                e^{i{\bf g}{\bf r}}=4\pi e^{i{\bf g}{\bf r}_{\alpha}} \sum_{\ell m} i^\ell 
                    j_{\ell}(g|{\bf r}-{\bf r}_{\alpha}|)
                    Y_{\ell m}^{*}({\bf \hat g}) Y_{\ell m}(\widehat{{\bf r}-{\bf r}_{\alpha}})
            \f]

            Multipole moment:
            \f[
                q_{\ell m} = \int Y_{\ell m}^{*}(\hat {\bf r}) r^l \rho({\bf r}) d {\bf r}

            \f]

            Spherical Bessel function moments
            \f[
                \int_0^R j_{\ell}(a x)x^{2+\ell} dx = \frac{\sqrt{\frac{\pi }{2}} R^{\ell+\frac{3}{2}} 
                    J_{\ell+\frac{3}{2}}(a R)}{a^{3/2}}
            \f]
            for a = 0 the integral is \f$ \frac{R^3}{3} \delta_{\ell,0} \f$

            General solution to the Poisson equation with spherical boundary condition:
            \f[
                V({\bf x}) = \int \rho({\bf x'})G({\bf x},{\bf x'}) d{\bf x'} - \frac{1}{4 \pi} \int_{S} V({\bf x'}) 
                    \frac{\partial G}{\partial n'} d{\bf S'}
            \f]

            Green's function for a sphere
            \f[
                G({\bf x},{\bf x'}) = 4\pi \sum_{\ell m} \frac{Y_{\ell m}^{*}(\hat {\bf x'}) 
                    Y_{\ell m}(\hat {\bf x})}{2\ell + 1}
                    \frac{r_{<}^{\ell}}{r_{>}^{\ell+1}}\Biggl(1 - \Big( \frac{r_{>}}{R} \Big)^{2\ell + 1} \Biggr)
            \f]

            Pseudodensity radial functions:
            \f[
                p_{\ell}(r) = r^{\ell} \left(1-\frac{r^2}{R^2}\right)^n
            \f]
            where n is the order of pseudo density.

        */
        void poisson(Periodic_function<double>* rho, Periodic_function<double>* vh);
        
        /// Generate XC potential and energy density
        void xc(Periodic_function<double>* rho, Periodic_function<double>* magnetization[3], 
                Periodic_function<double>* vxc, Periodic_function<double>* bxc[3], Periodic_function<double>* exc);
        
        /// Generate effective potential and magnetic field from charge density and magnetization.
        void generate_effective_potential(Periodic_function<double>* rho, Periodic_function<double>* magnetization[3]);
        
        void hdf5_read();

        void save();
        
        void load();
        
        void update_atomic_potential();
        
        template <processing_unit_t pu> 
        void add_mt_contribution_to_pw();

        /// Generate plane-wave coefficients of the potential in the interstitial region
        void generate_pw_coefs();

        double value(double* vc);

        void check_potential_continuity_at_mt();

        void copy_to_global_ptr(double* fmt, double* fit, Periodic_function<double>* src);

        //void copy_xc_potential(double* vxcmt, double* vxcir);

        //void copy_effective_magnetic_field(double* beffmt, double* beffit);
        
        Periodic_function<double>* effective_potential()
        {
            return effective_potential_;
        }

        Spheric_function<double>& effective_potential_mt(int ialoc)
        {
            return effective_potential_->f_mt(ialoc);
        }

        Periodic_function<double>** effective_magnetic_field()
        {
            return effective_magnetic_field_;
        }
        
        Periodic_function<double>* effective_magnetic_field(int i)
        {
            return effective_magnetic_field_[i];
        }

        Periodic_function<double>* coulomb_potential()
        {
            return coulomb_potential_;
        }
        
        Spheric_function<double>& coulomb_potential_mt(int ialoc)
        {
            return coulomb_potential_->f_mt(ialoc);
        }
        
        Periodic_function<double>* xc_potential()
        {
            return xc_potential_;
        }

        Periodic_function<double>* xc_energy_density()
        {
            return xc_energy_density_;
        }
        
        void allocate()
        {
            effective_potential_->allocate(true, true);
            for (int j = 0; j < parameters_.num_mag_dims(); j++) effective_magnetic_field_[j]->allocate(true, true);
        }
};

#include "potential.hpp"

};


