#pragma once

#include <TNL/Benchmarks/Benchmarks.h>
#include <TNL/Matrices/DenseMatrix.h>

namespace Decomposition {
namespace Benchmark {

template< typename Real,
          typename Device,
          typename Index,
          typename ResultReal = Real,
          typename Logger = TNL::Benchmarks::JsonLogging >
struct DecompositionBenchmarkResult
: public TNL::Benchmarks::BenchmarkResult
{
   using RealType = Real;
   using DeviceType = Device;
   using IndexType = Index;

   using HostMatrix = TNL::Matrices::DenseMatrix< Real, TNL::Devices::Host, Index, TNL::Algorithms::Segments::RowMajorOrder >;
   using BenchmarkMatrix = TNL::Matrices::DenseMatrix< ResultReal, Device, Index, TNL::Algorithms::Segments::RowMajorOrder >;

   using typename BenchmarkResult::HeaderElements;
   using typename BenchmarkResult::RowElements;
   using BenchmarkResult::stddev;
   using BenchmarkResult::bandwidth;
   using BenchmarkResult::speedup;
   using BenchmarkResult::time;


   DecompositionBenchmarkResult( const BenchmarkMatrix& inputMatrixA,
                                 const HostMatrix& hostResult,
                                 const BenchmarkMatrix& benchmarkMtxResult,
                                 const bool multiplyLU = false )
   : inputMatrixA( inputMatrixA ), hostResult( hostResult ), benchmarkMtxResult( benchmarkMtxResult ), multiplyLU( multiplyLU )
   {}

   virtual HeaderElements getTableHeader() const override
   {
      return HeaderElements({ "time", "stddev", "stddev/time", "loops", "bandwidth", "speedup", "Dense Diff.Max", "Dense A_new Diff.Max" });
   }

   virtual std::vector< int > getColumnWidthHints() const override
   {
      return std::vector< int >({ 14, 14, 14, 6, 14, 14, 20, 24 });
   }

   virtual RowElements getRowElements() const override
   {
      HostMatrix benchmarkResultCopy, diff;
      benchmarkResultCopy = benchmarkMtxResult;

      diff.setLike( hostResult );
      diff.addMatrix( hostResult );
      diff.addMatrix( benchmarkResultCopy, -1.0, 1.0 );

      RowElements elements;
      elements << std::scientific << time << stddev << stddev/time << loops << bandwidth;
      if( speedup != 0.0 )
         elements << speedup;
      else
         elements << "N/A";

      RealType maxAbs = 0.0;
      for( IndexType i = 0; i < diff.getRows(); ++i )
      {
         for( IndexType j = 0; j < diff.getRows(); ++j )
         {
               if( abs( diff.getElement( i, j ) ) > maxAbs )
               {
                  maxAbs = abs( diff.getElement( i, j ) );
               }
         }
      }

      elements << maxAbs;

      Real sum = 0;
      Real maxAbs_A = 0.0;
      Real absDiff_A = 0.0;

      if( multiplyLU ) {
         // Compute result between A and A_new = L*U
         HostMatrix L, U;

         // Construct L and U matrices from Z matrix
         L = benchmarkMtxResult;
         U = benchmarkMtxResult;
         // Set opposit triangle of respective matrices to 0
         for( Index i = 0; i < L.getRows(); ++i )
         {
            U( i, i ) = 1;
            for( Index j = i + 1; j < U.getColumns(); ++j )
            {
               L( i, j ) = 0;
               U( j, i ) = 0;
            }
         }

         // Multiply L*U and compare result with original A matrix
         HostMatrix inputMatrixA_copy;
         inputMatrixA_copy = inputMatrixA;

         for( int i = 0; i < U.getRows(); ++i )
         {
            for( int j = 0; j < L.getColumns(); ++j )
            {
               sum = 0;

               // A_new(i, j) = L(i, k) * U(k, j)
               for( int k = 0; k <= i; ++k) {
                  sum += L( i, k ) * U( k, j );
               }

               absDiff_A = abs( inputMatrixA_copy( i, j ) - sum );
               maxAbs_A = absDiff_A > maxAbs_A ? absDiff_A : maxAbs_A;
            }
         }
      }

      elements << maxAbs_A;
      return elements;
   }

   const HostMatrix& hostResult;
   const BenchmarkMatrix& inputMatrixA;
   const BenchmarkMatrix& benchmarkMtxResult;
   const bool multiplyLU;
};

} //namespace Benchmarks
} //namespace Decomposition
