#include "band.h"

namespace sirius {

template <typename T>
void Band::diag_pseudo_potential(K_point* kp__,
                                 Periodic_function<double>* effective_potential__,
                                 Periodic_function<double>* effective_magnetic_field__[3]) const
{
    PROFILE_WITH_TIMER("sirius::Band::diag_pseudo_potential");

    Hloc_operator hloc(ctx_.fft_coarse(), kp__->gkvec_vloc(), ctx_.mpi_grid_fft_vloc().communicator(1 << 1),
                       ctx_.num_mag_dims(), ctx_.gvec_coarse(), effective_potential__, effective_magnetic_field__);
    
    ctx_.fft_coarse().prepare(kp__->gkvec().partition());

    D_operator<T> d_op(ctx_, kp__->beta_projectors());
    Q_operator<T> q_op(ctx_, kp__->beta_projectors());

    auto& itso = ctx_.iterative_solver_input_section();
    if (itso.type_ == "exact") {
        if (ctx_.num_mag_dims() != 3) {
            for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                diag_pseudo_potential_exact(kp__, ispn, hloc, d_op, q_op);
            }
        } else {
            STOP();
        }
    } else if (itso.type_ == "davidson") {
        if (ctx_.num_mag_dims() != 3) {
            for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                diag_pseudo_potential_davidson(kp__, ispn, hloc, d_op, q_op);
            }
        } else {
            STOP();
        }
    } else if (itso.type_ == "rmm-diis") {
        if (ctx_.num_mag_dims() != 3) {
            for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                diag_pseudo_potential_rmm_diis(kp__, ispn, hloc, d_op, q_op);
            }
        } else {
            STOP();
        }
    } else if (itso.type_ == "chebyshev") {
        P_operator<T> p_op(ctx_, kp__->beta_projectors(), kp__->p_mtrx());
        if (ctx_.num_mag_dims() != 3) {
            for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                diag_pseudo_potential_chebyshev(kp__, ispn, hloc, d_op, q_op, p_op);

            }
        } else {
            STOP();
        }
    } else {
        TERMINATE("unknown iterative solver type");
    }

    ctx_.fft_coarse().dismiss();
}

template void Band::diag_pseudo_potential<double>(K_point* kp__,
                                                  Periodic_function<double>* effective_potential__,
                                                  Periodic_function<double>* effective_magnetic_field__[3]) const;

template void Band::diag_pseudo_potential<double_complex>(K_point* kp__,
                                                          Periodic_function<double>* effective_potential__,
                                                          Periodic_function<double>* effective_magnetic_field__[3]) const;
};
