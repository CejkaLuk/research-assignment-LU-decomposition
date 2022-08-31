/***************************************************************************
                          CroutMethodTest.h  -  description
                             -------------------
    begin                : Jul 18, 2021
    copyright            : (C) 2021 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <iostream>
#include <Decomposition/LU/CroutMethodIterative.h>
#include <math.h>

namespace Decomposition {
namespace LU {

CroutMethodIterative::CroutMethodIterative()
{
}

template< typename Matrix1, typename Matrix2 >
inline bool matrixDifferenceTolerable( Matrix1& L,
                                       Matrix2& U,
                                       Matrix1& Lnew,
                                       Matrix2& Unew,
                                       bool debug = false,
                                       typename Matrix1::RealType tolerance = 0.001 )
{
    if( debug )
    {
        auto diffL = Lnew.getValues() - L.getValues();
        auto diffU = Unew.getValues() - U.getValues();
        std::cout << "diffL = " << diffL << "\n diffU = " << diffU << std::endl;
        std::cout << "Max difference in elements of L matrix: " << max( abs( diffL ) ) << std::endl;
        std::cout << "Max difference in elements of U matrix: " << max( abs( diffU ) ) << std::endl;
    }

    return max( abs( Lnew.getValues() - L.getValues() ) ) <= tolerance &&
           max( abs( Unew.getValues() - U.getValues() ) ) <= tolerance;
}

template< typename Matrix1 >
inline bool matrixDifferenceTolerable( Matrix1& Z,
                                       Matrix1& Znew,
                                       bool debug = false,
                                       typename Matrix1::RealType tolerance = 0.001 )
{
    if( debug )
    {
        auto diffZ = Znew.getValues() - Z.getValues();
        std::cout << "diffZ = " << diffZ << std::endl;
        std::cout << "Max difference in elements of Z matrices: " << max( abs( diffZ ) ) << std::endl;
    }

    return max( abs( Znew.getValues() - Z.getValues() ) ) <= tolerance;
}

#ifdef HAVE_CUDA
template< typename RealType,
          typename IndexType,
          typename Matrix1,
          typename Matrix2,
          typename Matrix3 >
__global__ void DecomposeKernel( const Matrix1* _A,
                                 Matrix2* _L,
                                 Matrix3* _U,
                                 Matrix2* _Lnew,
                                 Matrix3* _Unew,
                                 const IndexType col_sStart,
                                 const IndexType num_cols,
                                 const IndexType row_sStart,
                                 const IndexType num_rows,
                                 const RealType* tolerance,
                                 bool* converged )
{
    auto& A = *_A;
    auto& L = *_L; auto& Lnew = *_Lnew;
    auto& U = *_U; auto& Unew = *_Unew;

    IndexType row = blockIdx.y * blockDim.y + threadIdx.y + row_sStart;
    IndexType col = blockIdx.x * blockDim.x + threadIdx.x + col_sStart;

    IndexType tx = threadIdx.x;
    IndexType ty = threadIdx.y;

    RealType sum = 0;

    IndexType max_col = num_cols - 1;
    IndexType max_row = num_rows - 1;
    IndexType new_row = min( row, max_row );
    IndexType new_col = min( col, max_col );

    IndexType min_row_col = min( new_row, new_col ) - BLOCK_SIZE;

    __shared__ RealType Ls[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ RealType Us[BLOCK_SIZE][BLOCK_SIZE];

    IndexType i, k;

    for ( i = 0; i <= min_row_col; i += BLOCK_SIZE ) {
        Ls[ty][tx] = L( new_row, i + tx );
        Us[ty][tx] = U( i + ty, new_col );

        __syncthreads();

#pragma unroll
        for( k = 0; k < BLOCK_SIZE; ++k ) {
            sum += Ls[ty][k] * Us[k][tx];
        }
        __syncthreads();
    }

    Ls[ty][tx] = L( new_row, min( i + tx, max_col ) );
    Us[ty][tx] = U( min( i + ty, max_row ), new_col );

    __syncthreads();

    IndexType t_to_use = row >= col ? tx : ty;

    for( k = 0; k < t_to_use; ++k ) {
        sum += Ls[ty][k] * Us[k][tx];
    }

    if( row >= num_rows || col >= num_rows ) {
        return;
    }

    if( row >= col ) {
        Lnew( row, col ) = A( row, col ) - sum;
    } else {
        if( L( row, row ) == 0 ) {
            printf( "Key element: %f\t(row:%d, col:%d)\n", L( row, row ), (int)row, (int)col );
            assert( L( row, row ) != 0 );
        }
        Unew( row, col ) = ( A( row, col ) - sum ) / L( row, row );
    }

    // Check convergence
    if( abs( Lnew( row, col ) - L( row, col ) ) > *tolerance ||
        abs( Unew( row, col ) - U( row, col ) ) > *tolerance )
    {
        *converged = false;
    }

    // Assign matrix for this iteration
    L( row, col ) = Lnew( row, col );
    U( row, col ) = Unew( row, col );
}
#endif

template< typename Matrix1,
          typename Matrix2,
          typename Matrix3 >
void CroutMethodIterative::decompose( Matrix1& A,
                                      Matrix2& L,
                                      Matrix3& U,
                                      const int threads = BLOCK_SIZE )
{
    if( BLOCK_SIZE != threads ) {
        std::cout << "Number of threads per block must be set manually in CroutMethodIterative.h for use in pragma unroll!" << std::endl;
        abort();
    }

    using RealType   = typename Matrix1::RealType;
    using DeviceType = typename Matrix1::DeviceType;
    using IndexType  = typename Matrix1::IndexType;

    IndexType num_rows = A.getRows();
    IndexType num_cols = A.getColumns();

    if( num_rows != num_cols ) {
        std::cout << "Matrix A must be a square matrix!" << std::endl;
        abort();
    }

    IndexType i, j;
    Matrix2 Lnew;
    Matrix3 Unew;

    Lnew.setLike( L );
    Unew.setLike( U );
    L = A;
    U = A;

    for( i = 0; i < num_rows; ++i )
    {
        U.setElement( i, i, 1 );
        Unew.setElement( i, i, 1 );
        for( j = i + 1; j < num_cols; ++j )
        {
            L.setElement( i, j, 0 );
            U.setElement( j, i, 0 );
        }
    }

    bool converged = false;
    RealType tolerance = 0;

    if( std::is_same< DeviceType, TNL::Devices::Host >::value )
    {
        IndexType k;
        RealType sum = 0;

        do
        {
            for( i = 0; i < num_rows; ++i )
            {
                for( j = 0; j < num_cols; ++j )
                {
                    if( i >= j )
                    {
                        sum = 0;
                        for( k = 0; k < j; ++k )
                            sum = sum + L( i, k ) * U( k, j );

                        Lnew( i, j ) = A( i, j ) - sum;
                    }
                    else
                    {
                        TNL_ASSERT( L( i, i ) != 0,
                                    std::cerr << "L( " << i << ", " << i <<
                                    " ) = 0. Cannot divide by 0!" << std::endl );
                        sum = 0;
                        for( k = 0; k < i; ++k )
                            sum = sum + L( i, k ) * U( k, j );

                        Unew( i, j ) = ( A( i, j ) - sum ) / L( i, i );
                    }
                }
            }
            converged = matrixDifferenceTolerable( L, U, Lnew, Unew, false, tolerance );

            L = Lnew;
            U = Unew;
        } while( !converged );
    }

    if( std::is_same< DeviceType, TNL::Devices::Cuda >::value )
    {
        IndexType sectionSize = min( max( num_cols / 10, (IndexType)256 ), (IndexType)1024 );
        sectionSize = ( sectionSize + BLOCK_SIZE - 1 ) / BLOCK_SIZE * BLOCK_SIZE;

        IndexType blocks = sectionSize / BLOCK_SIZE;
        dim3 threadsPerBlock( BLOCK_SIZE, BLOCK_SIZE );
        dim3 blocksPerGrid( blocks, blocks );

        Matrix1* matrixA_kernel     = TNL::Cuda::passToDevice( A );
        Matrix2* matrixL_kernel     = TNL::Cuda::passToDevice( L );
        Matrix3* matrixU_kernel     = TNL::Cuda::passToDevice( U );
        Matrix2* matrixLnew_kernel  = TNL::Cuda::passToDevice( Lnew );
        Matrix3* matrixUnew_kernel  = TNL::Cuda::passToDevice( Unew );
        bool*    converged_kernel   = TNL::Cuda::passToDevice( converged );
        RealType* tolerance_kernel  = TNL::Cuda::passToDevice( tolerance );

        bool* converged_host = NULL;
        cudaMallocHost( (void **) &converged_host, sizeof(bool) );

        // Allocate and initialize an array of stream handles
        int nstreams = 2;
        cudaStream_t *streams = (cudaStream_t *)malloc( nstreams * sizeof(cudaStream_t) );
        for( int i = 0; i < nstreams; i++ ) {
            cudaStreamCreate( &(streams[i]) );
        }

        IndexType diag_start, diag_end, sStart, sEnd;

        for( diag_start = 0, diag_end = min( num_cols, sectionSize );
             diag_start < diag_end;
             diag_start += sectionSize, diag_end = min( num_cols, diag_end + sectionSize) ) {
            // Converge the diagonal section first - Default stream
            do
            {
                *converged_host = true;
                cudaMemcpy( converged_kernel, converged_host, sizeof(bool), cudaMemcpyHostToDevice );

                DecomposeKernel< RealType, IndexType >
                             <<< blocksPerGrid,
                                 threadsPerBlock >>>
                               ( matrixA_kernel,
                                 matrixL_kernel,
                                 matrixU_kernel,
                                 matrixLnew_kernel,
                                 matrixUnew_kernel,
                                 diag_start,
                                 diag_end,
                                 diag_start,
                                 diag_end,
                                 tolerance_kernel,
                                 converged_kernel );

                // Synchronize after execution of kernel before copying value of converged
                cudaDeviceSynchronize();
                cudaMemcpy( converged_host, converged_kernel, sizeof(bool), cudaMemcpyDeviceToHost );
            } while( !*converged_host );

            for( sStart = diag_end, sEnd = min( num_cols, diag_end + sectionSize );
                 sStart < sEnd;
                 sStart += sectionSize, sEnd = min( num_cols, sEnd + sectionSize ) ) {
                do
                {
                    *converged_host = true;
                    cudaMemcpy( converged_kernel, converged_host, sizeof(bool), cudaMemcpyHostToDevice );

                    // Run Stream 1: iterate columns - rows should stay the same
                    DecomposeKernel< RealType, IndexType >
                                 <<< blocksPerGrid,
                                     threadsPerBlock,
                                     0,
                                     streams[0] >>>
                                   ( matrixA_kernel,
                                     matrixL_kernel,
                                     matrixU_kernel,
                                     matrixLnew_kernel,
                                     matrixUnew_kernel,
                                     sStart,
                                     sEnd,
                                     diag_start,
                                     diag_end,
                                     tolerance_kernel,
                                     converged_kernel );

                    // Run Stream 2: iterate rows - columns should stay the same
                    DecomposeKernel< RealType, IndexType >
                                 <<< blocksPerGrid,
                                     threadsPerBlock,
                                     0,
                                     streams[1] >>>
                                   ( matrixA_kernel,
                                     matrixL_kernel,
                                     matrixU_kernel,
                                     matrixLnew_kernel,
                                     matrixUnew_kernel,
                                     diag_start,
                                     diag_end,
                                     sStart,
                                     sEnd,
                                     tolerance_kernel,
                                     converged_kernel );

                    // Synchronize after execution of kernels before copying value of converged
                    cudaDeviceSynchronize();
                    cudaMemcpy( converged_host, converged_kernel, sizeof(bool), cudaMemcpyDeviceToHost );
                } while( !*converged_host );
            }
        }

        // Release resources
        for (int i = 0; i < nstreams; i++) {
            cudaStreamDestroy(streams[i]);
        }
        free(streams);

        TNL::Cuda::freeFromDevice( matrixA_kernel );
        TNL::Cuda::freeFromDevice( matrixL_kernel );
        TNL::Cuda::freeFromDevice( matrixU_kernel );
        TNL::Cuda::freeFromDevice( matrixLnew_kernel );
        TNL::Cuda::freeFromDevice( matrixUnew_kernel );
        TNL::Cuda::freeFromDevice( tolerance_kernel );
    }
}

#ifdef HAVE_CUDA
template< typename RealType,
          typename IndexType,
          typename Matrix1,
          typename Matrix2 >
__global__ void DecomposeKernel( const Matrix1* _A,
                                 Matrix2* _Z,
                                 Matrix2* _Znew,
                                 const IndexType col_sStart,
                                 const IndexType num_cols,
                                 const IndexType row_sStart,
                                 const IndexType num_rows,
                                 const RealType* tolerance,
                                 bool* converged )
{
    auto& A = *_A; auto& Z = *_Z; auto& Znew = *_Znew;

    IndexType row = blockIdx.y * blockDim.y + threadIdx.y + row_sStart;
    IndexType col = blockIdx.x * blockDim.x + threadIdx.x + col_sStart;

    IndexType tx = threadIdx.x;
    IndexType ty = threadIdx.y;

    RealType sum = 0;

    IndexType max_col = num_cols - 1;
    IndexType max_row = num_rows - 1;
    IndexType new_row = min( row, max_row );
    IndexType new_col = min( col, max_col );

    IndexType min_row_col = min( new_row, new_col ) - BLOCK_SIZE;

    __shared__ RealType ZsLower[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ RealType ZsUpper[BLOCK_SIZE][BLOCK_SIZE];

    IndexType i, k;

    for ( i = 0; i <= min_row_col; i += BLOCK_SIZE ) {
        ZsLower[ty][tx] = Z( new_row, i + tx );
        ZsUpper[ty][tx] = Z( i + ty, new_col );

        __syncthreads();

#pragma unroll
        for( k = 0; k < BLOCK_SIZE; ++k ) {
            sum += ZsLower[ty][k] * ZsUpper[k][tx];
        }
        __syncthreads();
    }

    ZsLower[ty][tx] = Z( new_row, min( i + tx, max_col ) );
    ZsUpper[ty][tx] = Z( min( i + ty, max_row ), new_col );

    __syncthreads();

    IndexType t_to_use = row >= col ? tx : ty;

    for( k = 0; k < t_to_use; ++k ) {
        sum += ZsLower[ty][k] * ZsUpper[k][tx];
    }

    if( row >= num_rows || col >= num_cols ) {
        return;
    }

    if( row >= col ) {
        Znew( row, col ) = A( row, col ) - sum;
    } else {
        if( Z( row, row ) == 0 ) {
            printf( "Key element: %f\t(row:%d, col:%d)\n", Z( row, row ), (int)row, (int)col );
            assert( Z( row, row ) != 0 );
        }
        Znew( row, col ) = ( A( row, col ) - sum ) / Z( row, row );
    }

    // Check convergence
    if( abs( Znew( row, col ) - Z( row, col ) ) > *tolerance )
    {
        *converged = false;
    }

    // Assign matrix for this iteration
    Z( row, col ) = Znew( row, col );
}
#endif

// WARNING: 1s found on the main diagonal of U (Crout decomposition) are not stored in matrix Z.
// Matrix Z has elements of L on its main diagonal (where L comes from Crout LU decomposition of matrix A).
template< typename Matrix1,
          typename Matrix2 >
void CroutMethodIterative::decompose( Matrix1& A,
                                      Matrix2& Z,
                                      const int threads = BLOCK_SIZE )
{
    if( BLOCK_SIZE != threads ) {
        std::cout << "Number of threads per block must be set manually in CroutMethodIterative.h for use in pragma unroll!" << std::endl;
        abort();
    }

    using RealType   = typename Matrix1::RealType;
    using DeviceType = typename Matrix1::DeviceType;
    using IndexType  = typename Matrix1::IndexType;

    IndexType num_rows = A.getRows();
    IndexType num_cols = A.getColumns();

    if( num_rows != num_cols ) {
        std::cout << "Matrix A must be a square matrix!" << std::endl;
        abort();
    }

    Matrix2 Znew;
    Znew.setLike( Z );

    Z = A;

    bool converged = true;
    RealType tolerance = 0;

    if( std::is_same< DeviceType, TNL::Devices::Host >::value )
    {
        IndexType i, j, k;
        RealType sum = 0;

        do
        {
            for( i = 0; i < num_rows; ++i )
            {
                for( j = 0; j < num_cols; ++j )
                {
                    if( i >= j )
                    {
                        sum = 0;
                        for( k = 0; k < j; ++k )
                            sum = sum + Z( i, k ) * Z( k, j );

                        Znew( i, j ) = A( i, j ) - sum;
                    }
                    else
                    {
                        TNL_ASSERT( Z( i, i ) != 0,
                                    std::cerr << "L( " << i << ", " << i <<
                                    " ) = 0. Cannot divide by 0!" << std::endl );
                        sum = 0;
                        for( k = 0; k < i; ++k )
                            sum = sum + Z( i, k ) * Z( k, j );

                        Znew( i, j ) = ( A( i, j ) - sum ) / Z( i, i );
                    }
                }
            }
            converged = matrixDifferenceTolerable( Z, Znew, false, tolerance );

            Z = Znew;
        } while( !converged );
    }

    if( std::is_same< DeviceType, TNL::Devices::Cuda >::value )
    {
        IndexType sectionSize = min( max( num_cols / 10, (IndexType)256 ), (IndexType)1024 );
        sectionSize = ( sectionSize + BLOCK_SIZE - 1 ) / BLOCK_SIZE * BLOCK_SIZE;

        IndexType blocks = sectionSize / BLOCK_SIZE;
        dim3 threadsPerBlock( BLOCK_SIZE, BLOCK_SIZE );
        dim3 blocksPerGrid( blocks, blocks );

        Matrix1* matrixA_kernel     = TNL::Cuda::passToDevice( A );
        Matrix2* matrixZ_kernel     = TNL::Cuda::passToDevice( Z );
        Matrix2* matrixZnew_kernel  = TNL::Cuda::passToDevice( Znew );
        bool*    converged_kernel   = TNL::Cuda::passToDevice( converged );
        RealType* tolerance_kernel  = TNL::Cuda::passToDevice( tolerance );

        bool* converged_host = NULL;
        cudaMallocHost( (void **) &converged_host, sizeof(bool) );

        // Allocate and initialize an array of stream handles
        int nstreams = 2;
        cudaStream_t *streams = (cudaStream_t *)malloc(nstreams * sizeof(cudaStream_t));
        for (int i = 0; i < nstreams; i++) {
            cudaStreamCreate(&(streams[i]));
        }

        IndexType diag_start, diag_end, sStart, sEnd;

        for( diag_start = 0, diag_end = min( num_cols, sectionSize );
             diag_start < diag_end;
             diag_start += sectionSize, diag_end = min( num_cols, diag_end + sectionSize) ) {
            // Converge the diagonal section first - Default stream
            do
            {
                *converged_host = true;
                cudaMemcpy( converged_kernel, converged_host, sizeof(bool), cudaMemcpyHostToDevice );

                DecomposeKernel< RealType, IndexType >
                             <<< blocksPerGrid,
                                 threadsPerBlock >>>
                               ( matrixA_kernel,
                                 matrixZ_kernel,
                                 matrixZnew_kernel,
                                 diag_start,
                                 diag_end,
                                 diag_start,
                                 diag_end,
                                 tolerance_kernel,
                                 converged_kernel );

                // Synchronize after execution of kernel before copying value of converged
                cudaDeviceSynchronize();
                cudaMemcpy( converged_host, converged_kernel, sizeof(bool), cudaMemcpyDeviceToHost );
            } while( !*converged_host );

            for( sStart = diag_end, sEnd = min( num_cols, diag_end + sectionSize );
                 sStart < sEnd;
                 sStart += sectionSize, sEnd = min( num_cols, sEnd + sectionSize ) ) {
                do
                {
                    *converged_host = true;
                    cudaMemcpy( converged_kernel, converged_host, sizeof(bool), cudaMemcpyHostToDevice );

                    // Run Stream 1: iterate columns - rows should stay the same
                    DecomposeKernel< RealType, IndexType >
                                 <<< blocksPerGrid,
                                     threadsPerBlock,
                                     0,
                                     streams[0] >>>
                                   ( matrixA_kernel,
                                     matrixZ_kernel,
                                     matrixZnew_kernel,
                                     sStart,
                                     sEnd,
                                     diag_start,
                                     diag_end,
                                     tolerance_kernel,
                                     converged_kernel );

                    // Run Stream 2: iterate rows - columns should stay the same
                    DecomposeKernel< RealType, IndexType >
                                 <<< blocksPerGrid,
                                     threadsPerBlock,
                                     0,
                                     streams[1] >>>
                                   ( matrixA_kernel,
                                     matrixZ_kernel,
                                     matrixZnew_kernel,
                                     diag_start,
                                     diag_end,
                                     sStart,
                                     sEnd,
                                     tolerance_kernel,
                                     converged_kernel );

                    // Synchronize after execution of kernels before copying value of converged
                    cudaDeviceSynchronize();
                    cudaMemcpy( converged_host, converged_kernel, sizeof(bool), cudaMemcpyDeviceToHost );
                } while( !*converged_host );
            }
        }

        // Release resources
        for (int i = 0; i < nstreams; i++) {
            cudaStreamDestroy(streams[i]);
        }
        free(streams);

        TNL::Cuda::freeFromDevice( matrixA_kernel );
        TNL::Cuda::freeFromDevice( matrixZ_kernel );
        TNL::Cuda::freeFromDevice( matrixZnew_kernel );
        TNL::Cuda::freeFromDevice( converged_kernel );
        TNL::Cuda::freeFromDevice( tolerance_kernel );
    }
}

void CroutMethodIterative::print()
{
    std::cout << "Printing LU Iterative Crout method" << std::endl;
}

} // namespace LU
} // namespace Decomposition
