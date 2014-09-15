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

/** \file mdarray.h
 *   
 *  \brief Contains implementation of mdarray class.
 */

#ifndef __MDARRAY_H__
#define __MDARRAY_H__

#include <cassert>
#include <memory>
#include <atomic>
#include <vector>
#include <cstring>
#include <initializer_list>
#ifdef _GPU_
#include "gpu_interface.h"
#endif
#include "typedefs.h"

/// Index descriptor of mdarray.
class mdarray_index_descriptor
{
    private:
        
        /// Beginning of index.
        int64_t begin_;

        /// End of index.
        int64_t end_;

        /// Size of index.
        size_t size_;

    public:
  
        /// Constructor of empty descriptor.
        mdarray_index_descriptor() : begin_(0), end_(-1), size_(0)
        {
        }
        
        /// Constructor for index range [0, size).
        mdarray_index_descriptor(size_t const size__) : begin_(0), end_(size__ - 1), size_(size__)
        {
        }
    
        /// Constructor for index range [begin, end]
        mdarray_index_descriptor(int64_t const begin__, int64_t const end__)
            : begin_(begin__), 
              end_(end__) , 
              size_(end_ - begin_ + 1)
        {
            assert(end_ >= begin_);
        };
        
        /// Return first index value.
        inline int64_t begin() const
        {
            return begin_;
        }
        
        /// Return last index value.
        inline int64_t end() const
        {
            return end_;
        }
        
        /// Return index size.
        inline size_t size() const
        {
            return size_;
        }
};

extern std::atomic<int64_t> mdarray_mem_count;
extern std::atomic<int64_t> mdarray_mem_count_max;

/// Simple delete handler which keeps track of allocated and deallocated memory ammount.
template<typename T>
struct mdarray_deleter
{
    /// Number of elements of the current allocation.
    size_t size_;

    int mode_;
    
    mdarray_deleter() : size_(0), mode_(0)
    {
    }

    mdarray_deleter(size_t const size__, int const mode__) : size_(size__), mode_(mode__)
    {
        mdarray_mem_count += size_ * sizeof(T);
        mdarray_mem_count_max = std::max(mdarray_mem_count.load(), mdarray_mem_count_max.load());
    }
    
    void operator()(T* p__) const
    {
        mdarray_mem_count -= size_ * sizeof(T);
        switch (mode_)
        {
            case 0:
            {
                delete[] p__;
                break;
            }
            #ifdef _GPU_
            case 1:
            {
                cuda_free_host(p__);
                break;
            }
            case 2:
            {
                cuda_free(p__);
                break;
            }
            #endif
            default:
            {
                printf("error at line %i of file %s: wrong delete mode\n", __LINE__, __FILE__);
                exit(-1);
            }
        }
    }
};

/// Base class of multidimensional array.
template <typename T, int N> 
class mdarray_base
{
    protected:
    
        /// Unique pointer to the allocated memory.
        std::unique_ptr<T[], mdarray_deleter<T> > unique_ptr_;
        
        /// Raw pointer.
        T* ptr_;
        
        #ifdef _GPU_
        /// Unique pointer to the allocated GPU memory.
        std::unique_ptr<T[], mdarray_deleter<T> > unique_ptr_device_;
        
        /// Raw pointer to GPU memory
        T* ptr_device_;  
        
        bool pinned_;
        #endif
        
        /// Array dimensions.
        mdarray_index_descriptor dims_[N];
        
        /// List of offsets to compute the element location by dimension indices. 
        int64_t offsets_[N];
        
    public:
        
        /// Constructor of an empty array.
        mdarray_base() 
            : unique_ptr_(nullptr),
              ptr_(nullptr)
              #ifdef _GPU_
              ,unique_ptr_device_(nullptr),
              ptr_device_(nullptr), 
              pinned_(false)
              #endif
        { 
        }
        
        /// Destructor.
        ~mdarray_base()
        {
            deallocate();
            #ifdef _GPU_
            deallocate_on_device();
            #endif
        }

        /// Copy constructor is forbidden
        mdarray_base(mdarray_base<T, N> const& src) = delete;
        
        /// Assignment operator is forbidden
        mdarray_base<T, N>& operator=(mdarray_base<T, N> const& src) = delete;
        
        /// Move constructor
        mdarray_base(mdarray_base<T, N>&& src) : unique_ptr_(std::move(src.unique_ptr_))
        {
            ptr_ = src.ptr_;
            for (int i = 0; i < N; i++)
            {
                dims_[i] = src.dims_[i];
                offsets_[i] = src.offsets_[i];
            }
        }

        /// Move assigment operator
        inline mdarray_base<T, N>& operator=(mdarray_base<T, N>&& src)
        {
            if (this != &src)
            {
                unique_ptr_ = std::move(src.unique_ptr_);
                ptr_ = src.ptr_;
                for (int i = 0; i < N; i++)
                {
                    dims_[i] = src.dims_[i];
                    offsets_[i] = src.offsets_[i];
                }
            }
            return *this;
        }

        //== void init_dimensions(std::vector<mdarray_index_descriptor> const& vd)
        //== {
        //==     assert(vd.size() == N);
        //==     
        //==     for (int i = 0; i < N; i++) dims_[i] = vd[i];
        //==     
        //==     offsets_[0] = -dims_[0].begin();
        //==     size_t n = 1;
        //==     for (int i = 1; i < N; i++) 
        //==     {
        //==         n *= dims_[i - 1].size();
        //==         offsets_[i] = n;
        //==         offsets_[0] -= offsets_[i] * dims_[i].begin();
        //==     }
        //== }

        void init_dimensions(std::initializer_list<mdarray_index_descriptor> args)
        {
            assert(args.size() == N);

            int i = 0;
            for (auto d: args) dims_[i++] = d;
            
            offsets_[0] = -dims_[0].begin();
            size_t n = 1;
            for (int i = 1; i < N; i++) 
            {
                n *= dims_[i - 1].size();
                offsets_[i] = n;
                offsets_[0] -= offsets_[i] * dims_[i].begin();
            }
        }
 
        /// Return total size (number of elements) of the array.
        inline size_t size() const
        {
            size_t size_ = 1;

            for (int i = 0; i < N; i++) size_ *= dims_[i].size();

            return size_;
        }
        
        /// Return size of particular dimension.
        inline size_t size(int i) const
        {
           assert(i < N);
           return dims_[i].size();
        }
        
        /// Return leading dimension size.
        inline uint32_t ld() const
        {
            assert(dims_[0].size() < size_t(1 << 31));

            return (int32_t)dims_[0].size();
        }

        /// Allocate memory for array.
        /** mode = 0: normal allocation of CPU memory \n
         *  mode = 1: page-locked allocation of CPU memory \n
         */
        void allocate(int mode__ = 0)
        {
            deallocate();

            size_t sz = size();

            if (mode__ == 0)
            {
                unique_ptr_ = std::unique_ptr< T[], mdarray_deleter<T> >(new T[sz], mdarray_deleter<T>(sz, 0));
                ptr_ = unique_ptr_.get();
            }

            if (mode__ == 1)
            {
                #ifdef _GPU_
                ptr_ = static_cast<T*>(cuda_malloc_host(sz * sizeof(T)));
                unique_ptr_ = std::unique_ptr< T[], mdarray_deleter<T> >(ptr_, mdarray_deleter<T>(sz, 1));
                #else
                printf("error at line %i of file %s: not compiled with GPU support\n", __LINE__, __FILE__);
                exit(0);
                #endif
            }
        }
        
        /// Deallocate memory and reset pointers.
        void deallocate()
        {
            unique_ptr_.reset(nullptr);
            ptr_ = nullptr;
        }
        
        inline void zero()
        {
            if (size() > 0)
            {
                assert(ptr_ != nullptr);
                memset(ptr_, 0, size() * sizeof(T));
            }
        }

        /// Set raw pointer.
        inline void set_ptr(T* ptr__)
        {
            ptr_ = ptr__;
        }
        
        /// Return raw pointer.
        inline T* ptr() const
        {
            return ptr_;
        }
        
        /// Compute hash of the array
        /** Example: printf("hash(h) : %16llX\n", h.hash()); */
        uint64_t hash() const
        {
            uint64_t h = 5381;

            for(size_t i = 0; i < size() * sizeof(T); i++) h = ((h << 5) + h) + ((unsigned char*)ptr_)[i];

            return h;
        }
        
        /// Copy the content of the array to dest
        void operator>>(mdarray_base<T, N>& dest__)
        {
            for (int i = 0; i < N; i++) 
            {
                if (dest__.dims_[i].begin() != dims_[i].begin() || dest__.dims_[i].end() != dims_[i].end())
                {
                    printf("error at line %i of file %s: array dimensions don't match\n", __LINE__, __FILE__);
                    exit(-1);
                }
            }
            memcpy(dest__.ptr(), ptr(), size() * sizeof(T));
        }

        #ifdef _GPU_
        void allocate_on_device()
        {
            size_t sz = size();

            ptr_device_ = (T*)cuda_malloc(sz * sizeof(T));

            unique_ptr_device_ = std::unique_ptr<T[], mdarray_deleter<T> >(ptr_device_, mdarray_deleter<T>(sz, 2));
        }

        void deallocate_on_device()
        {
            unique_ptr_device_.reset(nullptr);
            ptr_device_ = nullptr;
        }

        void allocate_page_locked()
        {
            allocate(1);
        }

        void copy_to_device() 
        {
            assert(ptr_ != nullptr);
            assert(ptr_device_ != nullptr);

            cuda_copy_to_device(ptr_device_, ptr_, size() * sizeof(T));
        }

        void copy_to_host() 
        {
            assert(ptr_ != nullptr);
            assert(ptr_device_ != nullptr);
            
            cuda_copy_to_host(ptr_, ptr_device_, size() * sizeof(T));
        }

        void async_copy_to_device(int stream_id__ = -1) 
        {
            cuda_async_copy_to_device(ptr_device_, ptr_, size() * sizeof(T), stream_id__);
        }
        
        void async_copy_to_host(int stream_id__ = -1) 
        {
            cuda_async_copy_to_host(ptr_, ptr_device_, size() * sizeof(T), stream_id__);
        }

        void zero_on_device()
        {
            cuda_memset(ptr_device_, 0, size() * sizeof(T));
        }

        void pin_memory()
        {
            if (pinned_)
            {
                printf("error at line %i of file %s: array is already pinned\n", __LINE__, __FILE__);
                exit(-1);
            }
            cuda_host_register(ptr_, size() * sizeof(T));
            pinned_ = true;
        }
        
        void unpin_memory()
        {
            if (pinned_)
            {
                cuda_host_unregister(ptr_);
                pinned_ = false;
            }
        }

        /// Set raw device pointer.
        inline void set_ptr_device(T* ptr_device__)
        {
            ptr_device_ = ptr_device__;
        }

        #endif
};

/// Muldidimensional array implementation.
template <typename T, int N> 
class mdarray : public mdarray_base<T, N>
{
};

// 1d specialization of multidimensional array.
template <typename T> 
class mdarray<T, 1>: public mdarray_base<T, 1> 
{
    public:
  
        mdarray() 
        {
        }

        mdarray(T* ptr__, mdarray_index_descriptor const& d0)
        {
            this->init_dimensions({d0});
            this->set_ptr(ptr__);
        }
        
        mdarray(mdarray_index_descriptor const& d0)
        {
            this->init_dimensions({d0});
            this->allocate();
        }

        inline T& operator()(int64_t const i0) 
        {
            assert(i0 >= this->dims_[0].begin() && i0 <= this->dims_[0].end());
            assert(this->ptr_ != nullptr);
            
            int64_t i = this->offsets_[0] + i0;
            return this->ptr_[i];
        }

        template <processing_unit_t pu>
        inline T* at()
        {
            switch (pu)
            {
                case cpu:
                {
                    assert(this->ptr_ != nullptr);
                    return this->ptr_;
                }
                case gpu:
                {
                    #ifdef _GPU_
                    assert(this->ptr_device_ != nullptr);
                    return this->ptr_device_;
                    #else
                    printf("error at line %i of file %s: not compiled with GPU support\n", __LINE__, __FILE__);
                    exit(0);
                    #endif
                }
            }
            return nullptr;
        }

        template <processing_unit_t pu>
        inline T* at(int64_t const i0)
        {
            assert(i0 >= this->dims_[0].begin() && i0 <= this->dims_[0].end());

            int64_t i = this->offsets_[0] + i0;

            switch (pu)
            {
                case cpu:
                {
                    assert(this->ptr_ != nullptr);
                    return &this->ptr_[i];
                }
                case gpu:
                {
                    #ifdef _GPU_
                    assert(this->ptr_device_ != nullptr);
                    return &this->ptr_device_[i];
                    #else
                    printf("error at line %i of file %s: not compiled with GPU support\n", __LINE__, __FILE__);
                    exit(0);
                    #endif
                }
            }
            return nullptr;
        }
};

// 2d specialization
template <typename T> 
class mdarray<T, 2>: public mdarray_base<T, 2> 
{
    public:
  
        mdarray() 
        {
        }

        mdarray(T* data_ptr, mdarray_index_descriptor const& d0, mdarray_index_descriptor const& d1)
        {
            set_dimensions(d0, d1);
            this->set_ptr(data_ptr);
        }

        #ifdef _GPU_
        mdarray(T* ptr__, T* ptr_device__, mdarray_index_descriptor const& d0, mdarray_index_descriptor const& d1)
        {
            set_dimensions(d0, d1);
            this->set_ptr(ptr__);
            this->set_ptr_device(ptr_device__);
        }
        #endif
        
        mdarray(mdarray_index_descriptor const& d0, mdarray_index_descriptor const& d1)
        {
            set_dimensions(d0, d1);
            this->allocate();
        }
        
        void set_dimensions(mdarray_index_descriptor const& d0, mdarray_index_descriptor const& d1)
        {
            //std::vector<mdarray_index_descriptor> vd;
            //vd.push_back(d0);
            //vd.push_back(d1);
            //this->init_dimensions(vd);
            this->init_dimensions({d0, d1});
        }
        template <processing_unit_t pu>
        inline T* at()
        {
            if (pu == cpu)
            {
                assert(this->ptr_ != nullptr);
                return this->ptr_;
            }
            #ifdef _GPU_
            else if (pu == gpu)
            {
                assert(this->ptr_device_ != nullptr);
                return this->ptr_device_;
            }
            #endif
            else
            {
                return nullptr;
            }
        }
        
        template <processing_unit_t pu>
        inline T* at(int64_t const i0, int64_t const i1)
        {
            assert(i0 >= this->dims_[0].begin() && i0 <= this->dims_[0].end());
            assert(i1 >= this->dims_[1].begin() && i1 <= this->dims_[1].end());

            int64_t i = this->offsets_[0] + i0 + i1 * this->offsets_[1];

            if (pu == cpu)
            {
                assert(this->ptr_ != nullptr);
                return &this->ptr_[i];
            }
            #ifdef _GPU_
            else if (pu == gpu)
            {
                assert(this->ptr_device_ != nullptr);
                return &this->ptr_device_[i];
            }
            #endif
            else
            {
                return nullptr;
            }
        }
    
        inline T& operator()(const int64_t i0, const int64_t i1) 
        {
            assert(i0 >= this->dims_[0].begin() && i0 <= this->dims_[0].end());
            assert(i1 >= this->dims_[1].begin() && i1 <= this->dims_[1].end());
            assert(this->ptr_ != nullptr);
            
            int64_t i = this->offsets_[0] + i0 + i1 * this->offsets_[1];
            return this->ptr_[i];
        }
    
        #ifdef _GPU_
        //==inline T* ptr_device()
        //=={
        //==    return this->ptr_device_;
        //==}

        //==inline T* ptr_device(const int64_t i0, const int64_t i1) 
        //=={
        //==    assert(i0 >= this->dims_[0].begin() && i0 <= this->dims_[0].end());
        //==    assert(i1 >= this->dims_[1].begin() && i1 <= this->dims_[1].end());
        //==    assert(this->ptr_device_ != nullptr);
        //==    
        //==    int64_t i = this->offsets_[0] + i0 + i1 * this->offsets_[1];
        //==    return &this->ptr_device_[i];
        //==}

        //== inline void copy_to_device() 
        //== {
        //==     assert(this->ptr_ != nullptr);
        //==     assert(this->ptr_device_ != nullptr);

        //==     cuda_copy_to_device(this->ptr_device_, this->ptr_, this->size() * sizeof(T));
        //== }

        //== inline void copy_to_device(int irow, int icol, int nrow, int ncol)
        //== {
        //==     cublas_set_matrix(nrow, ncol, sizeof(T), this->at<cpu>(irow, icol), this->ld(), 
        //==                       this->at<gpu>(irow, icol), this->ld());
        //== }
        #endif
};

// Alias for matrix
template <typename T> using matrix = mdarray<T, 2>;

// 3d specialization
template <typename T> 
class mdarray<T, 3>: public mdarray_base<T, 3> 
{
    private:

        mdarray<T, 2> submatrix_;

    public:
  
        mdarray() 
        {
        }

        mdarray(T* data_ptr, const mdarray_index_descriptor& d0, const mdarray_index_descriptor& d1, const mdarray_index_descriptor& d2)
        {
            set_dimensions(d0, d1, d2);
            this->set_ptr(data_ptr);
        }
        
        mdarray(const mdarray_index_descriptor& d0, const mdarray_index_descriptor& d1, const mdarray_index_descriptor& d2)
        {
            set_dimensions(d0, d1, d2);
            this->allocate();
        }
        
        void set_dimensions(const mdarray_index_descriptor& d0, const mdarray_index_descriptor& d1, const mdarray_index_descriptor& d2)
        {
            //std::vector<mdarray_index_descriptor> vd;
            //vd.push_back(d0);
            //vd.push_back(d1);
            //vd.push_back(d2);
            //this->init_dimensions(vd);
            this->init_dimensions({d0, d1, d2});
        }
    
        inline T& operator()(const int64_t i0, const int64_t i1, const int64_t i2) 
        {
            assert(i0 >= this->dims_[0].begin() && i0 <= this->dims_[0].end());
            assert(i1 >= this->dims_[1].begin() && i1 <= this->dims_[1].end());
            assert(i2 >= this->dims_[2].begin() && i2 <= this->dims_[2].end());
            assert(this->ptr_ != nullptr);
            
            int64_t i = this->offsets_[0] + i0 + i1 * this->offsets_[1] + i2 * this->offsets_[2];
            return this->ptr_[i];
        }

        #ifdef _GPU_
        inline T* ptr_device()
        {
            return this->ptr_device_;
        }

        inline T* ptr_device(const int64_t i0, const int64_t i1, const int64_t i2) 
        {
            assert(i0 >= this->dims_[0].begin() && i0 <= this->dims_[0].end());
            assert(i1 >= this->dims_[1].begin() && i1 <= this->dims_[1].end());
            assert(i2 >= this->dims_[2].begin() && i2 <= this->dims_[2].end());
            assert(this->ptr_device_ != nullptr);
            
            int64_t i = this->offsets_[0] + i0 + i1 * this->offsets_[1] + i2 * this->offsets_[2];
            return &this->ptr_device_[i];
        }
        #endif

        mdarray<T, 2>& submatrix(int idx)
        {
            submatrix_.set_dimensions(this->dims_[0], this->dims_[1]);
            submatrix_.set_ptr(&(*this)(0, 0, idx));
            return submatrix_;
        }
};

// 4d specialization
template <typename T> 
class mdarray<T, 4>: public mdarray_base<T, 4> 
{
    public:
  
        mdarray() 
        {
        }

        mdarray(T* data_ptr, const mdarray_index_descriptor& d0, const mdarray_index_descriptor& d1, const mdarray_index_descriptor& d2, const mdarray_index_descriptor& d3)
        {
            set_dimensions(d0, d1, d2, d3);
            this->set_ptr(data_ptr);
        }
        
        mdarray(const mdarray_index_descriptor& d0, const mdarray_index_descriptor& d1, const mdarray_index_descriptor& d2, const mdarray_index_descriptor& d3)
        {
            set_dimensions(d0, d1, d2, d3);
            this->allocate();
        }
         
        void set_dimensions(const mdarray_index_descriptor& d0, const mdarray_index_descriptor& d1, const mdarray_index_descriptor& d2, const mdarray_index_descriptor& d3)
        {
            //std::vector<mdarray_index_descriptor> vd;
            //vd.push_back(d0);
            //vd.push_back(d1);
            //vd.push_back(d2);
            //vd.push_back(d3);            
            //this->init_dimensions(vd);
            this->init_dimensions({d0, d1, d2, d3});
        }
    
        inline T& operator()(const int64_t i0, const int64_t i1, const int64_t i2, const int64_t i3) 
        {
            assert(i0 >= this->dims_[0].begin() && i0 <= this->dims_[0].end());
            assert(i1 >= this->dims_[1].begin() && i1 <= this->dims_[1].end());
            assert(i2 >= this->dims_[2].begin() && i2 <= this->dims_[2].end());
            assert(i3 >= this->dims_[3].begin() && i3 <= this->dims_[3].end());
            assert(this->ptr_ != nullptr);
            
            int64_t i = this->offsets_[0] + i0 + i1 * this->offsets_[1] + i2 * this->offsets_[2] + i3 * this->offsets_[3];
            return this->ptr_[i];
        }

        #ifdef _GPU_
        inline T* ptr_device()
        {
            return this->ptr_device_;
        }

        inline T* ptr_device(const int64_t i0, const int64_t i1, const int64_t i2, const int64_t i3) 
        {
            assert(i0 >= this->dims_[0].begin() && i0 <= this->dims_[0].end());
            assert(i1 >= this->dims_[1].begin() && i1 <= this->dims_[1].end());
            assert(i2 >= this->dims_[2].begin() && i2 <= this->dims_[2].end());
            assert(i3 >= this->dims_[3].begin() && i3 <= this->dims_[3].end());
            assert(this->ptr_device_ != nullptr);
            
            int64_t i = this->offsets_[0] + i0 + i1 * this->offsets_[1] + i2 * this->offsets_[2] + i3 * this->offsets_[3];
            return &this->ptr_device_[i];
        }
        #endif
};

#endif // __MDARRAY_H__

