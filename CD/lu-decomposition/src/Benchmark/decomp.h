// Implemented by: Lukas Cejka
//      Original implemented by J. Klinkovsky in TNL/Benchmarks/SpMV
//      This is an edited copy of TNL/Benchmarks/SpMV/spmv.h.

#pragma once

#include <TNL/Benchmarks/Benchmarks.h>
#include <TNL/Benchmarks/JsonLogging.h>
#include <TNL/Matrices/MatrixReader.h>

#include <Decomposition/LU/CroutMethod.h>
#include <Decomposition/LU/CroutMethodIterative.h>

#include "utils/DecompositionBenchmarkResult.h"
#include "utils/BenchmarkCPUCroutResults.h"
#include "utils/MatrixInfo.h"

namespace Decomposition {
   namespace Benchmark {

const double oneGB = 1024.0 * 1024.0 * 1024.0;

using BenchmarkType = TNL::Benchmarks::Benchmark< TNL::Benchmarks::JsonLogging >;

template< typename Real,
          typename Index,
          typename InputMatrix >
void
benchmarkDecomp( BenchmarkType& benchmark,
               const InputMatrix& inputMatrix,
               const TNL::Matrices::DenseMatrix< Real, TNL::Devices::Host, Index >& denseHostOutMatrix,
               const std::string& inputFileName,
               bool iterativeCpuBenchmark,
               bool verboseMR,
               const int threads,
               bool multiplyLU )
{
   using HostMatrix = TNL::Matrices::DenseMatrix< Real, TNL::Devices::Host, Index, TNL::Algorithms::Segments::RowMajorOrder >;
   using CudaMatrix = TNL::Matrices::DenseMatrix< Real, TNL::Devices::Cuda, Index, TNL::Algorithms::Segments::RowMajorOrder >;

   /////
   // Benchmark Decomposition on host
   //
   HostMatrix hostMatrixA, hostMatrixZ;
   try
   {
      hostMatrixA = inputMatrix;
   }
   catch(const std::exception& e)
   {
      benchmark.addErrorMessage( "Unable to convert the matrix to the target format:" + TNL::String(e.what()) );
      return;
   }

   hostMatrixZ.setLike( hostMatrixA );

   const int nonzeros = hostMatrixA.getNonzeroElementsCount();
   const double datasetSize = (double) nonzeros * ( 2 * sizeof( Real ) + sizeof( Index ) ) / oneGB;
   benchmark.setDatasetSize( datasetSize );

   if( iterativeCpuBenchmark )
   {
      auto resetHostMatrix = [&]() {
         hostMatrixZ.setValue( 0.0 );
      };

      auto decomposeHost = [&]() {
         Decomposition::LU::CroutMethodIterative::decompose( hostMatrixA, hostMatrixZ, threads );
      };

      DecompositionBenchmarkResult< Real, TNL::Devices::Host, Index > hostBenchmarkResults( hostMatrixA, denseHostOutMatrix, hostMatrixZ, multiplyLU );
      benchmark.setMetadataElement({ "format", TNL::Matrices::MatrixInfo< HostMatrix >::getFormat() + " Iterative Crout [CPU]" });
      benchmark.time< TNL::Devices::Host >( resetHostMatrix, "CPU", decomposeHost, hostBenchmarkResults );
   }

   /////
   // Benchmark Decomposition on CUDA
   //
#ifdef HAVE_CUDA
   CudaMatrix cudaMatrixA, cudaMatrixZ;
   try
   {
      cudaMatrixA = inputMatrix;
   }
   catch(const std::exception& e)
   {
      benchmark.addErrorMessage( "Unable to copy the matrix on GPU: " + TNL::String(e.what()) );
      return;
   }

   cudaMatrixZ.setLike( cudaMatrixA );

   auto resetCudaMatrix = [&]() {
      cudaMatrixZ.setValue( 0.0 );
   };

   auto decomposeCuda = [&]() {
      Decomposition::LU::CroutMethodIterative::decompose( cudaMatrixA, cudaMatrixZ, threads );
   };

   DecompositionBenchmarkResult< Real, TNL::Devices::Cuda, Index > cudaBenchmarkResults( cudaMatrixA, denseHostOutMatrix, cudaMatrixZ, multiplyLU );
   benchmark.setMetadataElement({ "format", TNL::Matrices::MatrixInfo< CudaMatrix >::getFormat() + " Iterative Crout [GPU] (" + TNL::String( std::to_string(threads) ) + " threads)"});
   benchmark.time< TNL::Devices::Cuda >( resetCudaMatrix, "GPU", decomposeCuda, cudaBenchmarkResults );
 #endif
}

template< typename Real,
          typename Index >
void
dispatchDenseDecomp( BenchmarkType& benchmark,
              const TNL::Matrices::DenseMatrix< Real, TNL::Devices::Host, Index >& denseHostOutMatrix,
              const TNL::String& inputFileName,
              bool iterativeCpuBenchmark,
              bool verboseMR,
              const int threads,
              bool multiplyLU )
{
   using HostMatrixType = TNL::Matrices::DenseMatrix< Real, TNL::Devices::Host, Index >;
   HostMatrixType hostMatrix;
   TNL::Matrices::MatrixReader< HostMatrixType >::readMtx( inputFileName, hostMatrix, verboseMR );
   benchmarkDecomp< Real, Index, HostMatrixType >( benchmark, hostMatrix, denseHostOutMatrix, inputFileName, iterativeCpuBenchmark, verboseMR, threads, multiplyLU );
}

template< typename Real = double,
          typename Index = int >
void
benchmarkDecomp( BenchmarkType& benchmark,
               const TNL::String& inputFileName,
               const TNL::Config::ParameterContainer& parameters,
               bool verboseMR,
               std::ostream& output,
               int verbose )
{
   using DenseHostMatrix = TNL::Matrices::DenseMatrix< Real, TNL::Devices::Host, Index, TNL::Algorithms::Segments::RowMajorOrder >;
   using DenseCudaMatrix = TNL::Matrices::DenseMatrix< Real, TNL::Devices::Cuda, Index, TNL::Algorithms::Segments::RowMajorOrder >;
   using BenchmarkResult = DecompositionBenchmarkResult< Real, TNL::Devices::Host, Index >;

   bool multiplyLU = parameters.getParameter< bool >( "multiply-LU" );

   DenseHostMatrix denseHostMatrixA, denseHostMatrixZ;

   ////
   // Set-up benchmark datasize
   //
   TNL::Matrices::MatrixReader< DenseHostMatrix >::readMtx( inputFileName, denseHostMatrixA, verboseMR );
   denseHostMatrixZ.setLike( denseHostMatrixA );

   const int nonzeros = denseHostMatrixA.getNonzeroElementsCount();
   const double datasetSize = (double) nonzeros * ( 2 * sizeof( Real ) + sizeof( Index ) ) / oneGB;
   benchmark.setDatasetSize( datasetSize );

   // Sanitize input matrix filename from the path
   std::string base_filename = inputFileName.substr(inputFileName.find_last_of("/\\") + 1);

   benchmark.setMetadataColumns({
      { "matrix name", base_filename },
      { "precision", TNL::getType< Real >() },
      { "rows", TNL::convertToString( denseHostMatrixA.getRows() ) },
      { "columns", TNL::convertToString( denseHostMatrixA.getColumns() ) },
      { "nonzeros", TNL::convertToString( nonzeros ) },
      });
   benchmark.setMetadataWidths({
      { "matrix name", 32 },
      { "format", 46 },
   });

   ////
   // Perform benchmark on host with regular Crout Method as a reference
   //
   auto resetHostMatrix = [&]() {
      denseHostMatrixZ.setValue( 0.0 );
   };

   auto decomposeDenseHost = [&]() {
      Decomposition::LU::CroutMethod::decompose( denseHostMatrixA, denseHostMatrixZ );
   };

   BenchmarkResult hostBenchmarkResults( denseHostMatrixA, denseHostMatrixZ, denseHostMatrixZ, multiplyLU );

   TNL::Benchmarks::JsonLogging logger( output, verbose );
   logger.setMetadataColumns({
      { "matrix name", base_filename },
      { "precision", TNL::getType< Real >() },
      { "rows", TNL::convertToString( denseHostMatrixA.getRows() ) },
      { "columns", TNL::convertToString( denseHostMatrixA.getColumns() ) },
      { "nonzeros", TNL::convertToString( nonzeros ) },
      });
   logger.setMetadataWidths({
      { "matrix name", 32 },
      { "format", 46 },
   });
   logger.setMetadataElement({ "format", "Dense Crout [CPU]" });

   TNL::String machine = parameters.getParameter< std::string >( "machine" );
   const int loops = parameters.getParameter< int >( "loops" );

   // Get the saved CPU results for Crout Method
   bool matrixResultsFound = Decomposition::Benchmark::BenchmarkCPUCroutResults< BenchmarkResult >::loadMatrixResults( hostBenchmarkResults, base_filename, machine, loops, TNL::getType< Real >() );

   // If saved results not found then perform the benchmark normally
   if( !matrixResultsFound ) {
      benchmark.setMetadataElement({ "format", "Dense Crout [CPU]" });
      benchmark.time< TNL::Devices::Host >( resetHostMatrix, "CPU", decomposeDenseHost, hostBenchmarkResults );

      // Save the resulting Z matrix
      TNL::String denseHostMatrixZ_filename = inputFileName + "_result_" + TNL::String(TNL::getType< Real >());
      denseHostMatrixZ.save( denseHostMatrixZ_filename );
   } else {
      denseHostMatrixZ.load( inputFileName + "_result_" + TNL::String(TNL::getType< Real >()) );
      logger.logResult( "CPU", hostBenchmarkResults.getTableHeader(), hostBenchmarkResults.getRowElements(), hostBenchmarkResults.getColumnWidthHints() );
   }

   denseHostMatrixA.reset();

   bool iterativeCpuBenchmark = parameters.getParameter< bool >( "with-iterative-cpu-benchmark" );
   const int threads = parameters.getParameter< int >( "threads-squared-per-block" );

   /////
   // Dispatch Decomposition benchmarks for iterative crout using DenseMatrix
   //
   dispatchDenseDecomp< Real, Index >( benchmark, denseHostMatrixZ, inputFileName, iterativeCpuBenchmark, verboseMR, threads, multiplyLU );
}

} // namespace Benchmark
} // namespace Decomposition
