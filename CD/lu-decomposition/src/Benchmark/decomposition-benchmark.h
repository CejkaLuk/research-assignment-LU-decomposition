// Implemented by: Lukas Cejka
//      Original implemented by J. Klinkovsky in TNL/Benchmarks/BLAS
//      This is an edited copy of Benchmarks/BLAS/spmv.h by: Lukas Cejka

#pragma once

#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Config/parseCommandLine.h>

#include "decomp.h"

#include <TNL/Matrices/MatrixReader.h>

using namespace TNL::Matrices;

#include <exception>
#include <ctime>
#include <experimental/filesystem>

using namespace TNL;
using namespace TNL::Benchmarks;

template< typename Real >
void
runDecompositionBenchmarks( Decomposition::Benchmark::BenchmarkType & benchmark,
                            const String & inputFileName,
                            const Config::ParameterContainer& parameters,
                            std::ostream& output,
                            int verbose,
                            bool verboseMR = false )
{
   try {
      Decomposition::Benchmark::benchmarkDecomp< Real >( benchmark, inputFileName, parameters, verboseMR, output, verbose );
   }
   catch( const std::exception& ex ) {
      std::cerr << ex.what() << std::endl;
   }
}

std::string getCurrDateTime()
{
   time_t rawtime;
   struct tm * timeinfo;
   char buffer[ 80 ];
   time( &rawtime );
   timeinfo = localtime( &rawtime );
   strftime( buffer, sizeof( buffer ), "%Y-%m-%d--%H:%M:%S", timeinfo );
   std::string curr_date_time( buffer );
   return curr_date_time;
}

void
setupConfig( Config::ConfigDescription & config )
{
   config.addDelimiter( "Benchmark settings:" );
   config.addRequiredEntry< String >( "input-file", "Input file name." );
   config.addEntry< bool >( "with-iterative-cpu-benchmark", "All matrices are tested on both CPU and GPU using the Iterative Crout method.", false );
   config.addEntry< String >( "log-file", "Log file name.", "decomposition-benchmark::" + getCurrDateTime() + ".log");
   config.addEntry< String >( "output-mode", "Mode for opening the log file.", "append" );
   config.addEntryEnum( "append" );
   config.addEntryEnum( "overwrite" );
   config.addEntry< String >( "precision", "Precision of the arithmetics.", "double" );
   config.addEntryEnum( "float" );
   config.addEntryEnum( "double" );
   config.addEntryEnum( "all" );
   config.addEntry< int >( "loops", "Number of iterations for every computation.", 10 );
   config.addEntry< int >( "verbose", "Verbose mode.", 1 );
   config.addEntry< int >( "verbose-MReader", "Verbose mode for Matrix Reader.", 0 );
   config.addEntry< int >( "threads-squared-per-block", "Number of threads squared per block.", 32 );
   config.addEntry< String >( "machine", "What machine the benchmark is being run on.", "lukas-pc" );
   config.addEntry< bool >( "multiply-LU", "Multiply decomposed LU matrices and compare to original A matrix", false );

   config.addDelimiter( "Device settings:" );
   Devices::Host::configSetup( config );
   Devices::Cuda::configSetup( config );
}

int
main( int argc, char* argv[] )
{
   Config::ParameterContainer parameters;
   Config::ConfigDescription conf_desc;

   setupConfig( conf_desc );

   if( ! parseCommandLine( argc, argv, conf_desc, parameters ) )
      return EXIT_FAILURE;

   if( ! Devices::Host::setup( parameters ) ||
       ! Devices::Cuda::setup( parameters ) )
      return EXIT_FAILURE;

   const String & inputFileName = parameters.getParameter< String >( "input-file" );
   const String & logFileName = parameters.getParameter< String >( "log-file" );
   String outputMode = parameters.getParameter< String >( "output-mode" );
   const String & precision = parameters.getParameter< String >( "precision" );
   const int loops = parameters.getParameter< int >( "loops" );
   const int verbose = parameters.getParameter< int >( "verbose" );
   const int verboseMR = parameters.getParameter< int >( "verbose-MReader" );

   if( inputFileName == "" )
   {
      std::cerr << "ERROR: Input file name is required." << std::endl;
      return EXIT_FAILURE;
   }
   if( std::experimental::filesystem::exists(logFileName.getString()) )
   {
      std::cout << "Log file " << logFileName << " exists and ";
      if( outputMode == "append" )
         std::cout << "new logs will be appended." << std::endl;
      else
         std::cout << "will be overwritten." << std::endl;
   }

   auto mode = std::ios::out;
   if( outputMode == "append" )
       mode |= std::ios::app;
   std::ofstream logFile( logFileName, mode );

   Decomposition::Benchmark::BenchmarkType benchmark( logFile, loops, verbose );

   std::map< std::string, std::string > metadata = getHardwareMetadata();
   writeMapAsJson( metadata, logFileName, ".metadata.json" );

   if( precision == "all" || precision == "float" )
      runDecompositionBenchmarks< float >( benchmark, inputFileName, parameters, logFile, verbose, verboseMR );
   if( precision == "all" || precision == "double" )
      runDecompositionBenchmarks< double >( benchmark, inputFileName, parameters, logFile, verbose, verboseMR );

   std::cout << "\n==> BENCHMARK FINISHED" << std::endl;
   return EXIT_SUCCESS;
}
