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

/** \file utils.h
 *   
 *  \brief Contains definition and partial implementation of sirius::Utils class.
 */

#ifndef __UTILS_H__
#define __UTILS_H__

#include <gsl/gsl_sf_erf.h>
#include <fstream>
#include <string>
#include <complex>
#include "sirius_internal.h"
#include "typedefs.h"
#include "constants.h"
#include "mdarray.h"
#include "vector3d.h"
#include "matrix3d.h"

/// Utility class.
class Utils
{
    public:
        
        /// Maximum number of \f$ \ell, m \f$ combinations for a given \f$ \ell_{max} \f$
        static inline int lmmax(int lmax)
        {
            return (lmax + 1) * (lmax + 1);
        }

        static inline int lm_by_l_m(int l, int m)
        {
            return (l * l + l + m);
        }

        static inline int lmax_by_lmmax(int lmmax__)
        {
            int lmax = int(std::sqrt(double(lmmax__)) + 1e-8) - 1;
            if (lmmax(lmax) != lmmax__) TERMINATE("wrong lmmax");
            return lmax;
        }

        static inline bool file_exists(const std::string file_name)
        {
            std::ifstream ifs(file_name.c_str());
            if (ifs.is_open()) return true;
            return false;
        }

        static inline double fermi_dirac_distribution(double e)
        {
            double kT = 0.001;
            if (e > 100 * kT) return 0.0;
            if (e < -100 * kT) return 1.0;
            return (1.0 / (exp(e / kT) + 1.0));
        }
        
        static inline double gaussian_smearing(double e, double delta)
        {
            return 0.5 * (1 - gsl_sf_erf(e / delta));
        }
        
        static inline double cold_smearing(double e)
        {
            double a = -0.5634;

            if (e < -10.0) return 1.0;
            if (e > 10.0) return 0.0;

            return 0.5 * (1 - gsl_sf_erf(e)) - 1 - 0.25 * exp(-e * e) * (a + 2 * e - 2 * a * e * e) / sqrt(pi);
        }

        static std::string double_to_string(double val, int precision = -1)
        {
            char buf[100];

            double abs_val = std::abs(val);

            if (precision == -1)
            {
                if (abs_val > 1.0) 
                {
                    precision = 6;
                }
                else if (abs_val > 1e-14)
                {
                    precision = int(-std::log(abs_val) / std::log(10.0)) + 7;
                }
                else
                {
                    return std::string("0.0");
                }
            }

            std::stringstream fmt;
            fmt << "%." << precision << "f";
        
            int len = snprintf(buf, 100, fmt.str().c_str(), val);
            for (int i = len - 1; i >= 1; i--) 
            {
                if (buf[i] == '0' && buf[i - 1] == '0') 
                {
                    buf[i] = 0;
                }
                else
                {
                    break;
                }
            }
            return std::string(buf);
        }

        static inline double phi_by_sin_cos(double sinp, double cosp)
        {
            double phi = std::atan2(sinp, cosp);
            if (phi < 0) phi += twopi;
            return phi;
        }

        static inline long double factorial(int n)
        {
            assert(n >= 0);

            long double result = 1.0L;
            for (int i = 1; i <= n; i++) result *= i;
            return result;
        }
        
        /// Simple hash function.
        /** Example: printf("hash: %16llX\n", hash()); */
        static uint64_t hash(void const* buff, size_t size, uint64_t h = 5381)
        {
            unsigned char const* p = static_cast<unsigned char const*>(buff);
            for(size_t i = 0; i < size; i++) h = ((h << 5) + h) + p[i];
            return h;
        }

        static void write_matrix(const std::string& fname,
                                 mdarray<double_complex, 2>& matrix,
                                 int nrow,
                                 int ncol,
                                 bool write_upper_only = true,
                                 bool write_abs_only = false,
                                 std::string fmt = "%18.12f")
        {
            static int icount = 0;

            if (nrow < 0 || nrow > (int)matrix.size(0) || ncol < 0 || ncol > (int)matrix.size(1))
                TERMINATE("wrong number of rows or columns");

            icount++;
            std::stringstream s;
            s << icount;
            std::string full_name = s.str() + "_" + fname;

            FILE* fout = fopen(full_name.c_str(), "w");

            for (int icol = 0; icol < ncol; icol++)
            {
                fprintf(fout, "column : %4i\n", icol);
                for (int i = 0; i < 80; i++) fprintf(fout, "-");
                fprintf(fout, "\n");
                if (write_abs_only)
                {
                    fprintf(fout, " row, absolute value\n");
                }
                else
                {
                    fprintf(fout, " row, real part, imaginary part, absolute value\n");
                }
                for (int i = 0; i < 80; i++) fprintf(fout, "-");
                fprintf(fout, "\n");
                
                int max_row = (write_upper_only) ? std::min(icol, nrow - 1) : (nrow - 1);
                for (int j = 0; j <= max_row; j++)
                {
                    if (write_abs_only)
                    {
                        std::string s = "%4i  " + fmt + "\n";
                        fprintf(fout, s.c_str(), j, abs(matrix(j, icol)));
                    }
                    else
                    {
                        fprintf(fout, "%4i  %18.12f %18.12f %18.12f\n", j, real(matrix(j, icol)), imag(matrix(j, icol)), 
                                                                        abs(matrix(j, icol)));
                    }
                }
                fprintf(fout,"\n");
            }

            fclose(fout);
        }

        
        static void write_matrix(std::string const& fname,
                                 bool write_all,
                                 mdarray<double, 2>& matrix)
        {
            static int icount = 0;

            icount++;
            std::stringstream s;
            s << icount;
            std::string full_name = s.str() + "_" + fname;

            FILE* fout = fopen(full_name.c_str(), "w");

            for (int icol = 0; icol < (int)matrix.size(1); icol++)
            {
                fprintf(fout, "column : %4i\n", icol);
                for (int i = 0; i < 80; i++) fprintf(fout, "-");
                fprintf(fout, "\n");
                fprintf(fout, " row\n");
                for (int i = 0; i < 80; i++) fprintf(fout, "-");
                fprintf(fout, "\n");
                
                int max_row = (write_all) ? ((int)matrix.size(0) - 1) : std::min(icol, (int)matrix.size(0) - 1);
                for (int j = 0; j <= max_row; j++)
                {
                    fprintf(fout, "%4i  %18.12f\n", j, matrix(j, icol));
                }
                fprintf(fout,"\n");
            }

            fclose(fout);
        }

        static void write_matrix(std::string const& fname,
                                 bool write_all,
                                 matrix<double_complex> const& mtrx)
        {
            static int icount = 0;

            icount++;
            std::stringstream s;
            s << icount;
            std::string full_name = s.str() + "_" + fname;

            FILE* fout = fopen(full_name.c_str(), "w");

            for (int icol = 0; icol < (int)mtrx.size(1); icol++)
            {
                fprintf(fout, "column : %4i\n", icol);
                for (int i = 0; i < 80; i++) fprintf(fout, "-");
                fprintf(fout, "\n");
                fprintf(fout, " row\n");
                for (int i = 0; i < 80; i++) fprintf(fout, "-");
                fprintf(fout, "\n");
                
                int max_row = (write_all) ? ((int)mtrx.size(0) - 1) : std::min(icol, (int)mtrx.size(0) - 1);
                for (int j = 0; j <= max_row; j++)
                {
                    fprintf(fout, "%4i  %18.12f %18.12f\n", j, real(mtrx(j, icol)), imag(mtrx(j, icol)));
                }
                fprintf(fout,"\n");
            }

            fclose(fout);
        }

        template <typename T>
        static void check_hermitian(const std::string& name, matrix<T> const& mtrx, int n = -1)
        {
            assert(mtrx.size(0) == mtrx.size(1));

            double maxdiff = 0.0;
            int i0 = -1;
            int j0 = -1;

            if (n == -1) {
                n = static_cast<int>(mtrx.size(0));
            }

            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    double diff = std::abs(mtrx(i, j) - type_wrapper<T>::conjugate(mtrx(j, i)));
                    if (diff > maxdiff) {
                        maxdiff = diff;
                        i0 = i;
                        j0 = j;
                    }
                }
            }

            if (maxdiff > 1e-10) {
                std::stringstream s;
                s << name << " is not a symmetric or hermitian matrix" << std::endl
                  << "  maximum error: i, j : " << i0 << " " << j0 << " diff : " << maxdiff;

                WARNING(s);
            }
        }

        static double confined_polynomial(double r, double R, int p1, int p2, int dm)
        {
            double t = 1.0 - std::pow(r / R, 2);
            switch (dm)
            {
                case 0:
                {
                    return (std::pow(r, p1) * std::pow(t, p2));
                }
                case 2:
                {
                    return (-4 * p1 * p2 * std::pow(r, p1) * std::pow(t, p2 - 1) / std::pow(R, 2) +
                            p1 * (p1 - 1) * std::pow(r, p1 - 2) * std::pow(t, p2) + 
                            std::pow(r, p1) * (4 * (p2 - 1) * p2 * std::pow(r, 2) * std::pow(t, p2 - 2) / std::pow(R, 4) - 
                                          2 * p2 * std::pow(t, p2 - 1) / std::pow(R, 2)));
                }
                default:
                {
                    TERMINATE("wrong derivative order");
                    return 0.0;
                }
            }
        }

        static std::vector<int> l_by_lm(int lmax)
        {
            std::vector<int> l_by_lm__(lmmax(lmax));
            for (int l = 0; l <= lmax; l++) {
                for (int m = -l; m <= l; m++) {
                    l_by_lm__[lm_by_l_m(l, m)] = l;
                }
            }
            return l_by_lm__;
        }

        static std::pair<vector3d<double>, vector3d<int>> reduce_coordinates(vector3d<double> coord)
        {
            const double eps{1e-6};

            std::pair<vector3d<double>, vector3d<int>> v; 
            
            v.first = coord;
            for (int i = 0; i < 3; i++) {
                v.second[i] = (int)floor(v.first[i]);
                v.first[i] -= v.second[i];
                if (v.first[i] < -eps || v.first[i] > 1.0 + eps) {
                    std::stringstream s;
                    s << "wrong fractional coordinates" << std::endl
                      << v.first[0] << " " << v.first[1] << " " << v.first[2];
                    TERMINATE(s);
                }
                if (v.first[i] < 0) {
                    v.first[i] = 0;
                }
                if (v.first[i] >= (1 - eps)) {
                    v.first[i] = 0;
                    v.second[i] += 1;
                }
            }
            for (int x: {0, 1, 2}) {
                if (std::abs(coord[x] - (v.first[x] + v.second[x])) > eps) {
                    std::stringstream s;
                    s << "wrong coordinate reduction" << std::endl
                      << "  original coord: " << coord << std::endl
                      << "  reduced coord: " << v.first << std::endl
                      << "  T: " << v.second;
                    TERMINATE(s);
                }
            }
            return v;
        }

        static vector3d<int> find_translations(double radius__, matrix3d<double> const& lattice_vectors__)
        {
            /* Volume = |(a0 x a1) * a2| = N1 * N2 * N3 * determinant of a lattice vectors matrix 
               Volume = h * S = 2 * R * |a_i x a_j| * N_i * N_j */

            vector3d<double> a0(lattice_vectors__(0, 0), lattice_vectors__(1, 0), lattice_vectors__(2, 0));
            vector3d<double> a1(lattice_vectors__(0, 1), lattice_vectors__(1, 1), lattice_vectors__(2, 1));
            vector3d<double> a2(lattice_vectors__(0, 2), lattice_vectors__(1, 2), lattice_vectors__(2, 2));

            double det = std::abs(lattice_vectors__.det());

            vector3d<int> limits;

            limits[0] = static_cast<int>(2 * radius__ * cross(a1, a2).length() / det) + 1;
            limits[1] = static_cast<int>(2 * radius__ * cross(a0, a2).length() / det) + 1;
            limits[2] = static_cast<int>(2 * radius__ * cross(a0, a1).length() / det) + 1;

            return limits;
        }

        static std::vector< std::pair<int, int> > l_m_by_lm(int lmax)
        {
            std::vector< std::pair<int, int> > l_m(lmmax(lmax));
            for (int l = 0; l <= lmax; l++) {
                for (int m = -l; m <= l; m++) {
                    int lm = lm_by_l_m(l, m);
                    l_m[lm].first = l;
                    l_m[lm].second = m;
                }
            }
            return l_m;
        }

        inline static double round(double a__, int n__)
        {
            double a0 = std::floor(a__);
            double b = std::round((a__ - a0) * std::pow(10, n__)) / std::pow(10, n__);
            return a0 + b;
        }

        template <typename T>
        inline static int sign(T val)
        {
            return (T(0) < val) - (val < T(0));
        }
};

#endif

