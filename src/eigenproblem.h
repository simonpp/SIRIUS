// Copyright (c) 2013-2014 Anton Kozhevnikov, Thomas Schulthess
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that 
// the following conditions are met:
// 
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the 
//    following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions 
//    and the following disclaimer in the documentation and/or other materials provided with the distribution.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED 
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A 
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR 
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR 
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/** \file eigenproblem.h
 *
 *  \brief Contains definition and implementaiton of various eigenvalue solver interfaces.
 */

#ifndef __EIGENPROBLEM_H__
#define __EIGENPROBLEM_H__

#include "constants.h"
#include "blacs_grid.h"
#include "linalg.h"

/// Type of the solver to use for the standard or generalized eigen-value problem
enum ev_solver_t 
{
    /// use LAPACK
    ev_lapack, 

    /// use ScaLAPACK
    ev_scalapack,

    /// use ELPA1 solver
    ev_elpa1,

    /// use ELPA2 (2-stage) solver
    ev_elpa2,

    /// use MAGMA
    ev_magma,

    /// use PLASMA
    ev_plasma,

    ev_rs_gpu,

    ev_rs_cpu
};

/// Base class for generalized eigen-value problem
class Eigenproblem
{
    public:

        virtual ~Eigenproblem()
        {
        }
        
        /// Complex generalized Hermitian-definite eigenproblem.
        virtual int solve(int32_t matrix_size, int32_t nevec,
                          double_complex* A, int32_t lda,
                          double_complex* B, int32_t ldb,
                          double* eval,
                          double_complex* Z, int32_t ldz,
                          int32_t num_rows_loc = 0, int32_t num_cols_loc = 0)
        {
            TERMINATE("solver is not implemented");
            return -1;
        }

        /// Real generalized symmetric-definite eigenproblem.
        virtual int solve(int32_t matrix_size, int32_t nevec,
                          double* A, int32_t lda,
                          double* B, int32_t ldb,
                          double* eval,
                          double* Z, int32_t ldz,
                          int32_t num_rows_loc = 0, int32_t num_cols_loc = 0)
        {
            TERMINATE("solver is not implemented");
            return -1;
        }
        
        /// Complex standard Hermitian-definite eigenproblem.
        virtual int solve(int32_t matrix_size, double_complex* A, int32_t lda, double* eval, double_complex* Z, int32_t ldz)
        {
            TERMINATE("standard eigen-value solver is not configured");
            return -1;
        }

        /// Real standard symmetric-definite eigenproblem.
        virtual int solve(int32_t matrix_size, double* A, int32_t lda, double* eval, double* Z, int32_t ldz)
        {
            TERMINATE("standard eigen-value solver is not configured");
            return -1;
        }

        virtual bool parallel() = 0;

        virtual ev_solver_t type() = 0;
};

/// Interface for LAPACK eigen-value solvers.
class Eigenproblem_lapack: public Eigenproblem
{
    private:

        double abstol_;
    
        std::vector<int32_t> get_work_sizes(int32_t matrix_size)
        {
            std::vector<int32_t> work_sizes(3);
            
            work_sizes[0] = 2 * matrix_size + matrix_size * matrix_size;
            work_sizes[1] = 1 + 5 * matrix_size + 2 * matrix_size * matrix_size;
            work_sizes[2] = 3 + 5 * matrix_size;
            return work_sizes;
        }

    public:

        Eigenproblem_lapack(double abstol__ = -1) : abstol_(abstol__)
        {
        }

        int solve(int32_t matrix_size, int32_t nevec, 
                  double_complex* A, int32_t lda,
                  double_complex* B, int32_t ldb,
                  double* eval, 
                  double_complex* Z, int32_t ldz,
                  int32_t num_rows_loc = 0, int32_t num_cols_loc = 0)
        {
            assert(nevec <= matrix_size);

            int nb = linalg_base::ilaenv(1, "ZHETRD", "U", matrix_size, 0, 0, 0);
            int lwork = (nb + 1) * matrix_size;
            int lrwork = 7 * matrix_size;
            int liwork = 5 * matrix_size;
            
            std::vector<double_complex> work(lwork);
            std::vector<double> rwork(lrwork);
            std::vector<int32_t> iwork(liwork);
            std::vector<int32_t> ifail(matrix_size);
            std::vector<double> w(matrix_size);
            double vl = 0.0;
            double vu = 0.0;
            int32_t m;
            int32_t info;
       
            int32_t ione = 1;
            FORTRAN(zhegvx)(&ione, "V", "I", "U", &matrix_size, A, &lda, B, &ldb, &vl, &vu, &ione, &nevec, &abstol_, &m, 
                            &w[0], Z, &ldz, &work[0], &lwork, &rwork[0], &iwork[0], &ifail[0], &info, (int32_t)1, 
                            (int32_t)1, (int32_t)1);

            if (m != nevec)
            {
                std::stringstream s;
                s << "not all eigen-values are found" << std::endl
                  << "target number of eign-values: " << nevec << std::endl
                  << "number of eign-values found: " << m;
                //TERMINATE(s);
                return 1;
            }

            if (info)
            {
                std::stringstream s;
                s << "zhegvx returned " << info; 
                TERMINATE(s);
            }

            std::memcpy(eval, &w[0], nevec * sizeof(double));

            return 0;
        }

        int solve(int32_t matrix_size, int32_t nevec, 
                  double* A, int32_t lda,
                  double* B, int32_t ldb,
                  double* eval, 
                  double* Z, int32_t ldz,
                  int32_t num_rows_loc = 0, int32_t num_cols_loc = 0)
        {
            assert(nevec <= matrix_size);

            int nb = linalg_base::ilaenv(1, "DSYTRD", "U", matrix_size, 0, 0, 0);
            int lwork = (nb + 3) * matrix_size;
            int liwork = 5 * matrix_size;
            
            std::vector<double> work(lwork);
            std::vector<int32_t> iwork(liwork);
            std::vector<int32_t> ifail(matrix_size);
            std::vector<double> w(matrix_size);
            double vl = 0.0;
            double vu = 0.0;
            int32_t m;
            int32_t info;
       
            int32_t ione = 1;
            FORTRAN(dsygvx)(&ione, "V", "I", "U", &matrix_size, A, &lda, B, &ldb, &vl, &vu, &ione, &nevec, &abstol_, &m, 
                            &w[0], Z, &ldz, &work[0], &lwork, &iwork[0], &ifail[0], &info, (int32_t)1, 
                            (int32_t)1, (int32_t)1);

            if (m != nevec)
            {
                std::stringstream s;
                s << "not all eigen-values are found" << std::endl
                  << "target number of eign-values: " << nevec << std::endl
                  << "number of eign-values found: " << m;
                WARNING(s);
                return 1;
            }

            if (info)
            {
                std::stringstream s;
                s << "dsygvx returned " << info; 
                TERMINATE(s);
            }

            std::memcpy(eval, &w[0], nevec * sizeof(double));

            return 0;
        }

        int solve(int32_t matrix_size, double_complex* A, int32_t lda, double* eval, double_complex* Z, int32_t ldz)
        {
            std::vector<int32_t> work_sizes = get_work_sizes(matrix_size);
            
            std::vector<double_complex> work(work_sizes[0]);
            std::vector<double> rwork(work_sizes[1]);
            std::vector<int32_t> iwork(work_sizes[2]);
            int32_t info;

            FORTRAN(zheevd)("V", "U", &matrix_size, A, &lda, eval, &work[0], &work_sizes[0], &rwork[0], &work_sizes[1], 
                            &iwork[0], &work_sizes[2], &info, (int32_t)1, (int32_t)1);
            
            for (int i = 0; i < matrix_size; i++)
                std::memcpy(&Z[ldz * i], &A[lda * i], matrix_size * sizeof(double_complex));
            
            if (info)
            {
                std::stringstream s;
                s << "zheevd returned " << info; 
                TERMINATE(s);
            }
            return 0;
        }

        bool parallel()
        {
            return false;
        }

        ev_solver_t type()
        {
            return ev_lapack;
        }
};

#ifdef __PLASMA
extern "C" void plasma_zheevd_wrapper(int32_t matrix_size, void* a, int32_t lda, void* z,
                                      int32_t ldz, double* eval);
#endif

/// Interface for PLASMA eigen-value solvers.
class Eigenproblem_plasma: public Eigenproblem
{
    public:

        Eigenproblem_plasma()
        {
        }

        #ifdef __PLASMA
        void solve(int32_t matrix_size, double_complex* A, int32_t lda, double* eval, double_complex* Z, int32_t ldz)
        {
            //plasma_set_num_threads(1);
            //omp_set_num_threads(1);
            //printf("before call to plasma_zheevd_wrapper\n");
            plasma_zheevd_wrapper(matrix_size, a, lda, z, lda, eval);
            //printf("after call to plasma_zheevd_wrapper\n");
            //plasma_set_num_threads(8);
            //omp_set_num_threads(8);
        }
        #endif
        
        bool parallel()
        {
            return false;
        }

        ev_solver_t type()
        {
            return ev_plasma;
        }
};

/// Interface for MAGMA eigen-value solvers.
class Eigenproblem_magma: public Eigenproblem
{
    public:

        Eigenproblem_magma()
        {
        }

        #ifdef __MAGMA
        int solve(int32_t matrix_size, int32_t nevec, 
                  double_complex* A, int32_t lda,
                  double_complex* B, int32_t ldb,
                  double* eval, 
                  double_complex* Z, int32_t ldz,
                  int32_t num_rows_loc = 0, int32_t num_cols_loc = 0)
        {
            assert(nevec <= matrix_size);

            int nt = omp_get_max_threads();
            
            magma_zhegvdx_2stage_wrapper(matrix_size, nevec, a, lda, b, ldb, eval);

            if (nt != omp_get_max_threads())
            {
                TERMINATE("magma has changed the number of threads");
            }
            
            for (int i = 0; i < nevec; i++) std::memcpy(&z[ldz * i], &a[lda * i], matrix_size * sizeof(double_complex));

            return 0;
        }
        #endif

        bool parallel()
        {
            return false;
        }

        ev_solver_t type()
        {
            return ev_magma;
        }
};

/// Interface for ScaLAPACK eigen-value solvers.
class Eigenproblem_scalapack: public Eigenproblem
{
    private:

        int32_t bs_row_;
        int32_t bs_col_;
        int num_ranks_row_;
        int num_ranks_col_;
        int blacs_context_;
        double abstol_;
        
        #ifdef __SCALAPACK
        std::vector<int32_t> get_work_sizes(int32_t matrix_size, int32_t nb, int32_t nprow, int32_t npcol, 
                                            int blacs_context)
        {
            std::vector<int32_t> work_sizes(3);
            
            int32_t nn = std::max(matrix_size, std::max(nb, 2));
            
            int32_t np0 = linalg_base::numroc(nn, nb, 0, 0, nprow);
            int32_t mq0 = linalg_base::numroc(nn, nb, 0, 0, npcol);
        
            work_sizes[0] = matrix_size + (np0 + mq0 + nb) * nb;
        
            work_sizes[1] = 1 + 9 * matrix_size + 3 * np0 * mq0;
        
            work_sizes[2] = 7 * matrix_size + 8 * npcol + 2;
            
            return work_sizes;
        }

        std::vector<int32_t> get_work_sizes_gevp(int32_t matrix_size, int32_t nb, int32_t nprow, int32_t npcol, 
                                                 int blacs_context)
        {
            std::vector<int32_t> work_sizes(3);
            
            int32_t nn = std::max(matrix_size, std::max(nb, 2));
            
            int32_t neig = std::max(1024, nb);

            int32_t nmax3 = std::max(neig, std::max(nb, 2));
            
            int32_t np = nprow * npcol;

            // due to the mess in the documentation, take the maximum of np0, nq0, mq0
            int32_t nmpq0 = std::max(linalg_base::numroc(nn, nb, 0, 0, nprow), 
                                  std::max(linalg_base::numroc(nn, nb, 0, 0, npcol),
                                           linalg_base::numroc(nmax3, nb, 0, 0, npcol))); 

            int32_t anb = linalg_base::pjlaenv(blacs_context, 3, "PZHETTRD", "L", 0, 0, 0, 0);
            int32_t sqnpc = (int32_t)pow(double(np), 0.5);
            int32_t nps = std::max(linalg_base::numroc(nn, 1, 0, 0, sqnpc), 2 * anb);

            work_sizes[0] = matrix_size + (2 * nmpq0 + nb) * nb;
            work_sizes[0] = std::max(work_sizes[0], matrix_size + 2 * (anb + 1) * (4 * nps + 2) + (nps + 1) * nps);
            work_sizes[0] = std::max(work_sizes[0], 3 * nmpq0 * nb + nb * nb);

            work_sizes[1] = 4 * matrix_size + std::max(5 * matrix_size, nmpq0 * nmpq0) + 
                            linalg_base::iceil(neig, np) * nn + neig * matrix_size;

            int32_t nnp = std::max(matrix_size, std::max(np + 1, 4));
            work_sizes[2] = 6 * nnp;

            return work_sizes;
        }
        #endif

    public:

        Eigenproblem_scalapack(BLACS_grid const& blacs_grid__, int32_t bs_row__, int32_t bs_col__, double abstol__ = -1)
            : bs_row_(bs_row__),
              bs_col_(bs_col__),
              num_ranks_row_(blacs_grid__.num_ranks_row()), 
              num_ranks_col_(blacs_grid__.num_ranks_col()), 
              blacs_context_(blacs_grid__.context()),
              abstol_(abstol__)
        {
        }

        #ifdef __SCALAPACK
        int solve(int32_t matrix_size, double_complex* A, int32_t lda, double* eval, double_complex* Z, int32_t ldz)
        {

            int desca[9];
            linalg_base::descinit(desca, matrix_size, matrix_size, bs_row_, bs_col_, 0, 0, blacs_context_, lda);
            
            int descz[9];
            linalg_base::descinit(descz, matrix_size, matrix_size, bs_row_, bs_col_, 0, 0, blacs_context_, ldz);
            
            std::vector<int32_t> work_sizes = get_work_sizes(matrix_size, std::max(bs_row_, bs_col_),
                                                             num_ranks_row_, num_ranks_col_, blacs_context_);
            
            std::vector<double_complex> work(work_sizes[0]);
            std::vector<double> rwork(work_sizes[1]);
            std::vector<int32_t> iwork(work_sizes[2]);
            int32_t info;

            int32_t ione = 1;
            FORTRAN(pzheevd)("V", "U", &matrix_size, A, &ione, &ione, desca, eval, Z, &ione, &ione, descz, &work[0], 
                             &work_sizes[0], &rwork[0], &work_sizes[1], &iwork[0], &work_sizes[2], &info, (int32_t)1, 
                             (int32_t)1);

            if (info)
            {
                std::stringstream s;
                s << "pzheevd returned " << info; 
                TERMINATE(s);
            }

            return 0;
        }

        int solve(int32_t matrix_size,  int32_t nevec, 
                  double_complex* A, int32_t lda,
                  double_complex* B, int32_t ldb,
                  double* eval, 
                  double_complex* Z, int32_t ldz,
                  int32_t num_rows_loc = 0, int32_t num_cols_loc = 0)
        {
            assert(nevec <= matrix_size);
            
            int32_t desca[9];
            linalg_base::descinit(desca, matrix_size, matrix_size, bs_row_, bs_col_, 0, 0, blacs_context_, lda);

            int32_t descb[9];
            linalg_base::descinit(descb, matrix_size, matrix_size, bs_row_, bs_col_, 0, 0, blacs_context_, ldb); 

            int32_t descz[9];
            linalg_base::descinit(descz, matrix_size, matrix_size, bs_row_, bs_col_, 0, 0, blacs_context_, ldz); 

            std::vector<int32_t> work_sizes = get_work_sizes_gevp(matrix_size, std::max(bs_row_, bs_col_), 
                                                                  num_ranks_row_, num_ranks_col_, blacs_context_);
            
            std::vector<double_complex> work(work_sizes[0]);
            std::vector<double> rwork(work_sizes[1]);
            std::vector<int32_t> iwork(work_sizes[2]);
            
            std::vector<int32_t> ifail(matrix_size);
            std::vector<int32_t> iclustr(2 * num_ranks_row_ * num_ranks_col_);
            std::vector<double> gap(num_ranks_row_ * num_ranks_col_);
            std::vector<double> w(matrix_size);
            
            double orfac = 1e-6;
            int32_t ione = 1;
            
            int32_t m;
            int32_t nz;
            double d1;
            int32_t info;

            FORTRAN(pzhegvx)(&ione, "V", "I", "U", &matrix_size, A, &ione, &ione, desca, B, &ione, &ione, descb, &d1, &d1, 
                             &ione, &nevec, &abstol_, &m, &nz, &w[0], &orfac, Z, &ione, &ione, descz, &work[0], &work_sizes[0], 
                             &rwork[0], &work_sizes[1], &iwork[0], &work_sizes[2], &ifail[0], &iclustr[0], &gap[0], &info, 
                             (int32_t)1, (int32_t)1, (int32_t)1); 

            if (info)
            {
                if ((info / 2) % 2)
                {
                    std::stringstream s;
                    s << "eigenvectors corresponding to one or more clusters of eigenvalues" << std::endl  
                      << "could not be reorthogonalized because of insufficient workspace" << std::endl;

                    int k = num_ranks_row_ * num_ranks_col_;
                    for (int i = 0; i < num_ranks_row_ * num_ranks_col_ - 1; i++)
                    {
                        if ((iclustr[2 * i + 1] != 0) && (iclustr[2 * (i + 1)] == 0))
                        {
                            k = i + 1;
                            break;
                        }
                    }
                   
                    s << "number of eigenvalue clusters : " << k << std::endl;
                    for (int i = 0; i < k; i++) s << iclustr[2 * i] << " : " << iclustr[2 * i + 1] << std::endl; 
                    TERMINATE(s);
                }

                std::stringstream s;
                s << "pzhegvx returned " << info; 
                TERMINATE(s);
            }

            if ((m != nevec) || (nz != nevec))
                TERMINATE("Not all eigen-vectors or eigen-values are found.");

            std::memcpy(eval, &w[0], nevec * sizeof(double));

            return 0;
        }
        #endif

        bool parallel()
        {
            return true;
        }

        ev_solver_t type()
        {
            return ev_scalapack;
        }
};

class Eigenproblem_elpa1: public Eigenproblem
{
    private:
        
        int32_t block_size_;
        int32_t num_ranks_row_;
        int32_t rank_row_;
        int32_t num_ranks_col_;
        int32_t rank_col_;
        int blacs_context_;
        Communicator const& comm_row_;
        Communicator const& comm_col_;

    public:
        
        Eigenproblem_elpa1(BLACS_grid const& blacs_grid__, int32_t block_size__)
            : block_size_(block_size__),
              num_ranks_row_(blacs_grid__.num_ranks_row()), 
              rank_row_(blacs_grid__.rank_row()),
              num_ranks_col_(blacs_grid__.num_ranks_col()), 
              rank_col_(blacs_grid__.rank_col()),
              blacs_context_(blacs_grid__.context()), 
              comm_row_(blacs_grid__.comm_row()), 
              comm_col_(blacs_grid__.comm_col())
        {
        }
        
        #ifdef __ELPA
        int solve(int32_t matrix_size, int32_t nevec, 
                  double_complex* A, int32_t lda,
                  double_complex* B, int32_t ldb,
                  double* eval, 
                  double_complex* Z, int32_t ldz,
                  int32_t num_rows_loc = 0, int32_t num_cols_loc = 0)
        {
            assert(nevec <= matrix_size);

            int32_t mpi_comm_rows = MPI_Comm_c2f(comm_row_.mpi_comm());
            int32_t mpi_comm_cols = MPI_Comm_c2f(comm_col_.mpi_comm());

            runtime::Timer *t;

            t = new runtime::Timer("elpa::ort");
            FORTRAN(elpa_cholesky_complex)(&matrix_size, B, &ldb, &block_size_, &mpi_comm_rows, &mpi_comm_cols);
            FORTRAN(elpa_invert_trm_complex)(&matrix_size, B, &ldb, &block_size_, &mpi_comm_rows, &mpi_comm_cols);
       
            mdarray<double_complex, 2> tmp1(num_rows_loc, num_cols_loc);
            mdarray<double_complex, 2> tmp2(num_rows_loc, num_cols_loc);

            FORTRAN(elpa_mult_ah_b_complex)("U", "L", &matrix_size, &matrix_size, B, &ldb, A, &lda, &block_size_, 
                                            &mpi_comm_rows, &mpi_comm_cols, tmp1.at<CPU>(), &num_rows_loc, (int32_t)1, 
                                            (int32_t)1);

            int32_t descc[9];
            linalg_base::descinit(descc, matrix_size, matrix_size, block_size_, block_size_, 0, 0, blacs_context_, lda);
            
            linalg_base::pztranc(matrix_size, matrix_size, complex_one, tmp1.at<CPU>(), 1, 1, descc, 
                                 complex_zero, tmp2.at<CPU>(), 1, 1, descc);

            FORTRAN(elpa_mult_ah_b_complex)("U", "U", &matrix_size, &matrix_size, B, &ldb, tmp2.at<CPU>(), &num_rows_loc, 
                                            &block_size_, &mpi_comm_rows, &mpi_comm_cols, A, &lda, (int32_t)1, 
                                            (int32_t)1);

            linalg_base::pztranc(matrix_size, matrix_size, complex_one, A, 1, 1, descc, complex_zero, 
                                 tmp1.at<CPU>(), 1, 1, descc);

            for (int i = 0; i < num_cols_loc; i++)
            {
                int32_t n_col = linalg_base::indxl2g(i + 1, block_size_, rank_col_, 0, num_ranks_col_);
                int32_t n_row = linalg_base::numroc(n_col, block_size_, rank_row_, 0, num_ranks_row_);
                for (int j = n_row; j < num_rows_loc; j++) 
                {
                    assert(j < num_rows_loc);
                    assert(i < num_cols_loc);
                    A[j + i * lda] = tmp1(j, i);
                }
            }
            delete t;
            
            t = new runtime::Timer("elpa::diag");
            std::vector<double> w(matrix_size);
            FORTRAN(elpa_solve_evp_complex)(&matrix_size, &nevec, A, &lda, &w[0], tmp1.at<CPU>(), &num_rows_loc, 
                                            &block_size_, &mpi_comm_rows, &mpi_comm_cols);
            delete t;

            t = new runtime::Timer("elpa::bt");
            linalg_base::pztranc(matrix_size, matrix_size, complex_one, B, 1, 1, descc, complex_zero, 
                                 tmp2.at<CPU>(), 1, 1, descc);

            FORTRAN(elpa_mult_ah_b_complex)("L", "N", &matrix_size, &nevec, tmp2.at<CPU>(), &num_rows_loc, tmp1.at<CPU>(), 
                                            &num_rows_loc, &block_size_, &mpi_comm_rows, &mpi_comm_cols, Z, &ldz, 
                                            (int32_t)1, (int32_t)1);
            delete t;

            std::memcpy(eval, &w[0], nevec * sizeof(double));

            return 0;
        }
        #endif

        bool parallel()
        {
            return true;
        }

        ev_solver_t type()
        {
            return ev_elpa1;
        }
};

class Eigenproblem_elpa2: public Eigenproblem
{
    private:
        
        int32_t block_size_;
        int32_t num_ranks_row_;
        int32_t rank_row_;
        int32_t num_ranks_col_;
        int32_t rank_col_;
        int blacs_context_;
        Communicator const& comm_row_;
        Communicator const& comm_col_;
        Communicator const& comm_all_;

    public:
        
        Eigenproblem_elpa2(BLACS_grid const& blacs_grid__, int32_t block_size__)
            : block_size_(block_size__),
              num_ranks_row_(blacs_grid__.num_ranks_row()), 
              rank_row_(blacs_grid__.rank_row()),
              num_ranks_col_(blacs_grid__.num_ranks_col()), 
              rank_col_(blacs_grid__.rank_col()),
              blacs_context_(blacs_grid__.context()), 
              comm_row_(blacs_grid__.comm_row()), 
              comm_col_(blacs_grid__.comm_col()),
              comm_all_(blacs_grid__.comm())
        {
        }
        
        #ifdef __ELPA
        int solve(int32_t matrix_size, int32_t nevec, 
                  double_complex* A, int32_t lda,
                  double_complex* B, int32_t ldb,
                  double* eval, 
                  double_complex* Z, int32_t ldz,
                  int32_t num_rows_loc, int32_t num_cols_loc)
        {
            assert(nevec <= matrix_size);

            int32_t mpi_comm_rows = MPI_Comm_c2f(comm_row_.mpi_comm());
            int32_t mpi_comm_cols = MPI_Comm_c2f(comm_col_.mpi_comm());
            int32_t mpi_comm_all = MPI_Comm_c2f(comm_all_.mpi_comm());

            runtime::Timer *t;

            t = new runtime::Timer("elpa::ort");
            FORTRAN(elpa_cholesky_complex)(&matrix_size, B, &ldb, &block_size_, &mpi_comm_rows, &mpi_comm_cols);
            FORTRAN(elpa_invert_trm_complex)(&matrix_size, B, &ldb, &block_size_, &mpi_comm_rows, &mpi_comm_cols);
       
            mdarray<double_complex, 2> tmp1(num_rows_loc, num_cols_loc);
            mdarray<double_complex, 2> tmp2(num_rows_loc, num_cols_loc);

            FORTRAN(elpa_mult_ah_b_complex)("U", "L", &matrix_size, &matrix_size, B, &ldb, A, &lda, &block_size_, 
                                            &mpi_comm_rows, &mpi_comm_cols, tmp1.at<CPU>(), &num_rows_loc, (int32_t)1, 
                                            (int32_t)1);

            int32_t descc[9];
            linalg_base::descinit(descc, matrix_size, matrix_size, block_size_, block_size_, 0, 0, blacs_context_, lda);

            linalg_base::pztranc(matrix_size, matrix_size, complex_one, tmp1.at<CPU>(), 1, 1, descc, complex_zero, 
                                 tmp2.at<CPU>(), 1, 1, descc);

            FORTRAN(elpa_mult_ah_b_complex)("U", "U", &matrix_size, &matrix_size, B, &ldb, tmp2.at<CPU>(), &num_rows_loc, 
                                            &block_size_, &mpi_comm_rows, &mpi_comm_cols, A, &lda, (int32_t)1, 
                                            (int32_t)1);

            linalg_base::pztranc(matrix_size, matrix_size, complex_one, A, 1, 1, descc, complex_zero, tmp1.at<CPU>(), 1, 1, descc);

            for (int i = 0; i < num_cols_loc; i++)
            {
                int32_t n_col = linalg_base::indxl2g(i + 1, block_size_, rank_col_, 0, num_ranks_col_);
                int32_t n_row = linalg_base::numroc(n_col, block_size_, rank_row_, 0, num_ranks_row_);
                for (int j = n_row; j < num_rows_loc; j++) 
                {
                    assert(j < num_rows_loc);
                    assert(i < num_cols_loc);
                    A[j + i * lda] = tmp1(j, i);
                }
            }
            delete t;
            
            t = new runtime::Timer("elpa::diag");
            std::vector<double> w(matrix_size);
            FORTRAN(elpa_solve_evp_complex_2stage)(&matrix_size, &nevec, A, &lda, &w[0], tmp1.at<CPU>(), &num_rows_loc, 
                                                   &block_size_, &mpi_comm_rows, &mpi_comm_cols, &mpi_comm_all);
            delete t;

            t = new runtime::Timer("elpa::bt");
            linalg_base::pztranc(matrix_size, matrix_size, complex_one, B, 1, 1, descc, complex_zero, 
                                 tmp2.at<CPU>(), 1, 1, descc);

            FORTRAN(elpa_mult_ah_b_complex)("L", "N", &matrix_size, &nevec, tmp2.at<CPU>(), &num_rows_loc, tmp1.at<CPU>(), 
                                            &num_rows_loc, &block_size_, &mpi_comm_rows, &mpi_comm_cols, Z, &ldz, 
                                            (int32_t)1, (int32_t)1);
            delete t;

            memcpy(eval, &w[0], nevec * sizeof(double));

            return 0;
        }
        #endif

        bool parallel()
        {
            return true;
        }

        ev_solver_t type()
        {
            return ev_elpa2;
        }
};

#ifdef __RS_GEN_EIG
void my_gen_eig(char uplo, int n, int nev, double_complex* a, int ia, int ja, int* desca,
                double_complex* b, int ib, int jb, int* descb, double* d,
                double_complex* q, int iq, int jq, int* descq, int* info);

void my_gen_eig_cpu(char uplo, int n, int nev, double_complex* a, int ia, int ja, int* desca,
                    double_complex* b, int ib, int jb, int* descb, double* d,
                    double_complex* q, int iq, int jq, int* descq, int* info);
#endif

class Eigenproblem_RS_CPU: public Eigenproblem
{
    private:

        int32_t bs_row_;
        int32_t bs_col_;
        int num_ranks_row_;
        int rank_row_;
        int num_ranks_col_;
        int rank_col_;
        int blacs_context_;
        
    public:

        Eigenproblem_RS_CPU(BLACS_grid const& blacs_grid__, int32_t bs_row__, int32_t bs_col__)
            : bs_row_(bs_row__),
              bs_col_(bs_col__),
              num_ranks_row_(blacs_grid__.num_ranks_row()),
              rank_row_(blacs_grid__.rank_row()),
              num_ranks_col_(blacs_grid__.num_ranks_col()), 
              rank_col_(blacs_grid__.rank_col()),
              blacs_context_(blacs_grid__.context())
        {
        }

        #ifdef __RS_GEN_EIG
        int solve(int32_t matrix_size, int32_t nevec, 
                  double_complex* A, int32_t lda,
                  double_complex* B, int32_t ldb,
                  double* eval, 
                  double_complex* Z, int32_t ldz,
                  int32_t num_rows_loc, int32_t num_cols_loc)
        {
        
            assert(nevec <= matrix_size);
            
            int32_t desca[9];
            lin_alg<scalapack>::descinit(desca, matrix_size, matrix_size, block_size_, block_size_, 0, 0, 
                                        blacs_context_, lda);

            int32_t descb[9];
            lin_alg<scalapack>::descinit(descb, matrix_size, matrix_size, block_size_, block_size_, 0, 0, 
                                        blacs_context_, ldb); 

            mdarray<double_complex, 2> ztmp(num_rows_loc, num_cols_loc);
            int32_t descz[9];
            lin_alg<scalapack>::descinit(descz, matrix_size, matrix_size, block_size_, block_size_, 0, 0, 
                                        blacs_context_, num_rows_loc); 
            
            std::vector<double> eval_tmp(matrix_size);

            int info;
            my_gen_eig_cpu('L', matrix_size, nevec, a, 1, 1, desca, b, 1, 1, descb, &eval_tmp[0], ztmp.ptr(), 1, 1, descz, &info);
            if (info)
            {
                std::stringstream s;
                s << "my_gen_eig " << info; 
                TERMINATE(s);
            }

            for (int i = 0; i < lin_alg<scalapack>::numroc(nevec, block_size_, rank_col_, 0, num_ranks_col_); i++)
                memcpy(&z[ldz * i], &ztmp(0, i), num_rows_loc * sizeof(double_complex));

            memcpy(eval, &eval_tmp[0], nevec * sizeof(double));

            return 0;
        }
        #endif

        bool parallel()
        {
            return true;
        }

        ev_solver_t type()
        {
            return ev_rs_cpu;
        }

};

class Eigenproblem_RS_GPU: public Eigenproblem
{
    private:

        int32_t bs_row_;
        int32_t bs_col_;
        int num_ranks_row_;
        int rank_row_;
        int num_ranks_col_;
        int rank_col_;
        int blacs_context_;
        
    public:

        Eigenproblem_RS_GPU(BLACS_grid const& blacs_grid__, int32_t bs_row__, int32_t bs_col__)
            : bs_row_(bs_row__),
              bs_col_(bs_col__),
              num_ranks_row_(blacs_grid__.num_ranks_row()),
              rank_row_(blacs_grid__.rank_row()),
              num_ranks_col_(blacs_grid__.num_ranks_col()), 
              rank_col_(blacs_grid__.rank_col()),
              blacs_context_(blacs_grid__.context())
        {
        }

        #ifdef __RS_GEN_EIG
        int solve(int32_t matrix_size, int32_t nevec,
                  double_complex* A, int32_t lda,
                  double_complex* B, int32_t ldb,
                  double* eval, 
                  double_complex* Z, int32_t ldz,
                  int32_t num_rows_loc, int32_t num_cols_loc)
        {
            assert(nevec <= matrix_size);
            
            int32_t desca[9];
            lin_alg<scalapack>::descinit(desca, matrix_size, matrix_size, block_size_, block_size_, 0, 0, 
                                        blacs_context_, lda);

            int32_t descb[9];
            lin_alg<scalapack>::descinit(descb, matrix_size, matrix_size, block_size_, block_size_, 0, 0, 
                                        blacs_context_, ldb); 

            mdarray<double_complex, 2> ztmp(nullptr, num_rows_loc, num_cols_loc);
            ztmp.allocate(1);
            int32_t descz[9];
            lin_alg<scalapack>::descinit(descz, matrix_size, matrix_size, block_size_, block_size_, 0, 0, 
                                        blacs_context_, num_rows_loc); 
            
            std::vector<double> eval_tmp(matrix_size);

            int info;
            my_gen_eig('L', matrix_size, nevec, a, 1, 1, desca, b, 1, 1, descb, &eval_tmp[0], ztmp.ptr(), 1, 1, descz, &info);
            if (info)
            {
                std::stringstream s;
                s << "my_gen_eig " << info; 
                TERMINATE(s);
            }

            for (int i = 0; i < lin_alg<scalapack>::numroc(nevec, block_size_, rank_col_, 0, num_ranks_col_); i++)
                std::memcpy(&Z[ldz * i], &ztmp(0, i), num_rows_loc * sizeof(double_complex));

            std::memcpy(eval, &eval_tmp[0], nevec * sizeof(double));

            return 0;
        }
        #endif

        bool parallel()
        {
            return true;
        }

        ev_solver_t type()
        {
            return ev_rs_gpu;
        }
};

#endif // __EIGENPROBLEM_H__

