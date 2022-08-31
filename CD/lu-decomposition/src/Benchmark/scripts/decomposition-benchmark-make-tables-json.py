#!/usr/bin/python3

from cmath import log
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from TNL.BenchmarkLogs import *
from python_libs.constants import * # Import constant variables
import re # Regex



####
# Create multiindex for columns
def get_multiindex( input_df, formats ):
   level1 = [ 'Matrix name', 'rows', 'columns', 'nonzeros per row' ]
   level2 = [ ''           , ''    , ''       , ''                 ]
   level3 = [ ''           , ''    , ''       , ''                 ]
   df_data = [[ ' ', ' ', ' ', ' ' ]]
   for format in formats:
      data_cols = ['bandwidth', 'time' ]
      # No need to compare the BASELINE format to itself -> max. difference only between other formats and baseline
      if not format == FORMATS.BASELINE:
         data_cols.append( 'diff.max' )
         data_cols.append( 'A_new diff.max' )
         data_cols.append( 'speed-up compared to' )
      else:
         data_cols.append( 'A_new diff.max' )
      for data in data_cols:
         level1.append( format )
         level2.append( data )
         # Write baseline format name only under speedup column, not under anything else
         if data != 'speed-up compared to':
            level3.append( '' )
         elif "GPU" in format and "- pert" in format:
            level3.append( format.replace( " - pert", "") )
         else:
            level3.append( FORMATS.BASELINE )
         df_data[ 0 ].append( ' ' )
      if format == 'Best':
         level1.append( format )
         level2.append( 'format' )
         level3.append( '' )
         df_data[ 0 ].append( ' ' )

         level1.append( format )
         level2.append( 'performer' )
         level3.append( '' )
         df_data[ 0 ].append( ' ' )

   multiColumns = pd.MultiIndex.from_arrays( [ level1, level2, level3 ] )
   return multiColumns, df_data

####
# Convert input table to better structured one
def convert_data_frame( input_df, multicolumns, df_data, begin_idx = 0, end_idx = -1 ):
   frames = []
   in_idx = 0
   out_idx = 0

   if end_idx == -1:
      end_idx = len(input_df.index)

   best_count = 0

   while in_idx < len(input_df.index) and out_idx < end_idx:
      # Get matrix name from line number in_idx
      matrixName = input_df.iloc[in_idx]['matrix name']

      # Get all records from the dataframe for that matrix
      df_matrix = input_df.loc[input_df['matrix name'] == matrixName]

      # Print out information id of row where this matrix name first appears out of how many lines
      if out_idx >= begin_idx:
         print( f'{out_idx} : {in_idx} / {len(input_df.index)} : {matrixName}' )
      else:
         print( f'{out_idx} : {in_idx} / {len(input_df.index)} : {matrixName} - SKIP' )

      # Create Frame for Data - columns for output.html
      aux_df = pd.DataFrame( df_data, columns = multicolumns, index = [out_idx] )

      fastest_format = {
         "format": input_df.iloc[in_idx]['format'],
         "bandwidth": input_df.iloc[in_idx]['bandwidth'],
         "time": input_df.iloc[in_idx]['time'],
         "diff.max": input_df.iloc[in_idx]['Dense Diff.Max'],
         "A_new diff.max": input_df.iloc[in_idx]['Dense A_new Diff.Max'],
         "performer": input_df.iloc[in_idx]['performer'],
      }

      for index,row in df_matrix.iterrows():
         # Save the data from that run
         aux_df.iloc[0]['Matrix name']      = row['matrix name']
         aux_df.iloc[0]['rows']             = row['rows']
         aux_df.iloc[0]['columns']          = row['columns']
         aux_df.iloc[0]['nonzeros per row'] = float(row['nonzeros'])/float(row['rows'])

         current_format = row['format']
         current_device = row['performer']

         # Measured data
         bw       = pd.to_numeric( row['bandwidth']     , errors='coerce' )
         time     = pd.to_numeric( row['time']          , errors='coerce' )
         diff_max = pd.to_numeric( row['Dense Diff.Max'], errors='coerce' )
         A_new_diff_max = pd.to_numeric( row['Dense A_new Diff.Max'], errors='coerce' )

         aux_df.iloc[0][( current_format, 'bandwidth', '' )] = bw
         aux_df.iloc[0][( current_format, 'time'     , '' )] = time
         aux_df.iloc[0][( current_format, 'diff.max' , '' )] = diff_max
         aux_df.iloc[0][( current_format, 'A_new diff.max' , '' )] = A_new_diff_max

         if time < fastest_format['time']:
            fastest_format['format'] = current_format
            fastest_format['bandwidth'] = bw
            fastest_format['time'] = time
            fastest_format['diff.max'] = diff_max
            fastest_format['A_new diff.max'] = A_new_diff_max
            fastest_format['performer'] = current_device

      # Fastest format fill data
      aux_df.iloc[0][(FORMATS.BEST, 'bandwidth', '')] = fastest_format['bandwidth']
      aux_df.iloc[0][(FORMATS.BEST, 'time', '')] = fastest_format['time']
      aux_df.iloc[0][(FORMATS.BEST, 'diff.max', '')] = fastest_format['diff.max']
      aux_df.iloc[0][(FORMATS.BEST, 'A_new diff.max', '')] = fastest_format['A_new diff.max']
      aux_df.iloc[0][(FORMATS.BEST, 'format', '')] = fastest_format['format']
      aux_df.iloc[0][(FORMATS.BEST, 'performer', '')] = fastest_format['performer']
      best_count += 1

      # Add the final line with data to the output
      if out_idx >= begin_idx:
         frames.append( aux_df )

      # Incremenet line id
      out_idx = out_idx + 1

      # Increment unique matrix line id (if a matrix takes up 3 lines - bcs 3 formats -> increment this id by 3)
      in_idx = in_idx + len(df_matrix.index)
   result = pd.concat( frames )
   return result

####
# Compute speed-up of particular formats compared to Cusparse on GPU and CSR on CPU
def compute_baseline_speedup( df, formats ):
   # Best format does not have a Device type, so here we add an empty string to DEVICES.ALL
   for format in formats:
      if not format in FORMATS.BASELINE:
         try:
            format_times_list = df[(format, 'time')]
         except:
            print("EXCEPTION!!!")
            continue
         print( 'Adding speed-up for ', format )
         if "- pert" in format:
            baseline_times_list = df[(format.replace(" - pert", ""), 'time')]
         else:
            baseline_times_list = df[(FORMATS.BASELINE, 'time')]

         baseline_speedup_list = []

         for( format_time, baseline_time ) in zip( format_times_list, baseline_times_list ):
            try:
               baseline_speedup_list.append( baseline_time / format_time  )
            except:
               baseline_speedup_list.append(float('nan'))
         if "- pert" in format:
            df[(format, 'speed-up compared to', format.replace(" - pert", ""))] = baseline_speedup_list
         else:
            df[(format, 'speed-up compared to', FORMATS.BASELINE)] = baseline_speedup_list

def compute_speedup( df, formats ):
   compute_baseline_speedup( df, formats )

###
# Draw several profiles into one figure
def draw_profiles( formats, profiles, matrices_list, xlabel, ylabel, filename, legend_loc='upper right', bar='none' ):
   fig, axs = plt.subplots( 1, 1, figsize=(6,4) )
   latexNames = []
   size = 1
   for format in formats:
      t = np.arange(profiles[format].size )

      axs.plot( t, profiles[format], '-o', ms=1, lw=1 )
      size = len( profiles[format] )
      latexNames.append( format )
   if bar != 'none':
      bar_data = np.full( size, 1 )
      axs.plot( t, bar_data, '-', ms=1, lw=1.5 )
      if bar != '':
         latexNames.append( bar )

   axs.set_xticks( t, matrices_list, rotation=45, ha='right' )

   axs.legend( latexNames, loc=legend_loc )
   axs.set_xlabel( xlabel )
   axs.set_ylabel( ylabel )
   plt.rcParams.update({
      "text.usetex": True,
      "font.family": "sans-serif",
      "font.sans-serif": ["Helvetica"]})

   plt.tight_layout()
   plt.savefig( filename + '.pdf')

   plt.gca().grid(which='major', axis='x', linestyle='--', alpha=0.3)

   axs.set_yscale( 'log' )
   plt.savefig( filename + '-log.pdf' )
   plt.savefig( filename + '-log.png', dpi=200)
   plt.close(fig)


####
# Effective BW profile
def effective_bw_profile( df, formats, head_size=10 ):
   if not os.path.exists( "BW-profile" ):
      os.mkdir( "BW-profile" )

   matrices_list = list(df[ 'Matrix name' ].values.tolist())

   profiles = {}
   for format in formats:
      # BandWidth of the best format is already created in previous formats BandWidth
      if format == FORMATS.BEST:
         continue

      print( f"Writing BW profile of {format}" )
      fig, axs = plt.subplots( 1, 1, figsize=(6,4) )
      t = np.arange( df[( format, 'bandwidth' )].size )

      if format == FORMATS.BASELINE:
         profiles[ format ] = df[( format, 'bandwidth' )].copy()
         axs.plot( t, df[( format, 'bandwidth')], '-o', ms=1, lw=1, label=format )
      else:
         profiles[ format ] = df[( format, 'bandwidth' )].copy()
         axs.plot( t, df[( format, 'bandwidth' )], '-o', ms=1, lw=1, label=format )

      axs.set_xticks( t, matrices_list, rotation=45, ha='right' )

      axs.legend( loc='upper right' )
      axs.set_xlabel( "Matrices - ascending matrix dimensions" )
      axs.set_ylabel( 'Effective bandwidth in GB/sec' )
      plt.rcParams.update({
         "text.usetex": True,
         "font.family": "sans-serif",
         "font.sans-serif": ["Helvetica"]})

      fig.tight_layout()
      plt.savefig( f"BW-profile/{format}.pdf" )
      plt.close( fig )

      fig, axs = plt.subplots( 1, 1, figsize=(6,4) )
      axs.set_yscale( 'log' )

      if format == FORMATS.BASELINE:
         axs.plot( t, result[( format, 'bandwidth' )], '-o', ms=1, lw=1, label=format )
      else:
         axs.plot( t, result[( format, 'bandwidth' )], '-o', ms=1, lw=1, label=format )

      axs.set_xticks( t, list(result[ 'Matrix name' ].values.tolist()), rotation=45, ha='right' )

      axs.legend( loc='lower left' )
      axs.set_xlabel( "Matrices - ascending matrix dimensions" )
      axs.set_ylabel( 'Effective bandwidth in GB/sec' )

      plt.rcParams.update({
         "text.usetex": True,
         "font.family": "sans-serif",
         "font.sans-serif": ["Helvetica"]})

      fig.tight_layout()
      plt.savefig( f"BW-profile/{format}-log.pdf")
      plt.close(fig)
      copy_df = df.copy()
      for f in formats:
         if not f in [FORMATS.BASELINE, format]:
            copy_df.drop( labels=f, axis='columns', level=0, inplace=True )
      copy_df.to_html( f"BW-profile/{format}.html" )

   # Draw All formats profiles
   current_formats = []
   xlabel = "Matrices - ascending matrix dimensions"
   ylabel = "Effective bandwidth in GB/sec"
   for format in formats:
      if format != FORMATS.BEST:
         current_formats.append( format )
   draw_profiles( current_formats, profiles, matrices_list, xlabel, ylabel, "all-profiles-bw", 'upper right', "none" )

####
# Comparison of speed-up w.r.t. Baseline format
def baseline_speedup_comparison( df, formats, head_size=10 ):
   speedup_folder_name = "Speed-up-profile"
   if not os.path.exists( speedup_folder_name ):
      os.mkdir( speedup_folder_name )

   matrices_list = list(df[ 'Matrix name' ].values.tolist())
   profiles = {}
   for format in formats:
      if not format in [ FORMATS.BASELINE, FORMATS.BEST ]:
         if "- pert" in format:
            base_format = format.replace(" - pert", "")
         else:
            base_format = FORMATS.BASELINE
         print( f"Writing comparison of speed-up of {format} compared to {base_format}" )

         df[ 'tmp' ] = df[( format, 'bandwidth' )]
         filtered_df = df.dropna( subset=[( 'tmp', '', '', '' )] )

         profiles[ format ] = filtered_df[( format, 'speed-up compared to', base_format )].copy()

         fig, axs = plt.subplots( 1, 1, figsize=(6,4) )
         size = len( filtered_df[( format, 'speed-up compared to', base_format )].index )
         t = np.arange( size )
         bar = np.full( size, 1 )

         axs.set_xticks( t, matrices_list, rotation=45, ha='right' )

         axs.plot( t, filtered_df[( format, 'speed-up compared to', base_format)], '-o', ms=1, lw=1 )
         axs.plot( t, bar, '-', ms=1, lw=1 )
         axs.legend( [ format, base_format ], loc='upper right' )
         axs.set_ylabel( 'Speedup' )
         axs.set_xlabel( "Matrices - ascending matrix dimensions" )

         plt.rcParams.update({
            "text.usetex": True,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica"]})

         plt.tight_layout()
         plt.savefig( f"{speedup_folder_name}/{base_format}-vs-{format}-speed-up.pdf")
         plt.close(fig)

         fig, axs = plt.subplots( 1, 1, figsize=(6,4) )
         axs.plot( t, filtered_df[( format, 'speed-up compared to', base_format)], '-o', ms=1, lw=1 )
         axs.plot( t, bar, '-', ms=1, lw=1 )
         # This is the only line that is different from the code block above that does the same thing
         # TODO: Can this block be removed and only the set_yscale( 'log' ) line along with the following one would stay -> save the figure with different name
         axs.legend( [ format, base_format ], loc='lower left' )
         axs.set_ylabel( 'Speedup' )
         axs.set_xlabel( "Matrices - ascending matrix dimensions" )

         axs.set_xticks( t, matrices_list, rotation=45, ha='right' )

         axs.set_yscale( 'log' )

         plt.tight_layout()
         plt.savefig( f"{speedup_folder_name}/{base_format}-vs-{format}-speed-up-log.pdf" )
         plt.close(fig)

         copy_df = df.copy()
         for f in formats:
            if not f in [ base_format, FORMATS.BEST, format ]:
               copy_df.drop( labels=f, axis='columns', level=0, inplace=True )
         copy_df.to_html( f"{speedup_folder_name}/{base_format}-vs-{format}-speed-up.html" )

####
# Make data analysis
def processDf( df, formats, input_file_dir, input_file_name, optimization_file_path, head_size = 10 ):
   print( "Writting to HTML file..." )
   df.to_html( f'output.html' )

   # Generate tables and figures
   effective_bw_profile( df, formats, head_size )
   baseline_speedup_comparison( df, formats, head_size )

   best = df[( 'Best', 'format' )].tolist()
   best_formats = list(set(best))
   sum = 0
   for format in formats:
      if( not 'Best' in format ):
         cases = best.count(format)
         print( f'{format} is best in {cases} cases.')
         sum += cases
   print( f'Total number of matrices: {sum}.' )
   print( f'Best formats {best_formats}.')


   # Add to optimization excel
   if optimization_file_path != None:

      # Get benchmark name
      # Get out of '<precision>/<machine>/Decomposition*/general' folder -> benchmark_dir
      os.chdir( "../.." )
      log_machine = os.path.basename( os.getcwd() )
      os.chdir( ".." )
      log_precision = os.path.basename( os.getcwd() )
      os.chdir( ".." )
      benchmark_dir = os.getcwd()
      benchmark_name = os.path.basename( benchmark_dir )
      # Quit if the file doesn't exist, we require the CPU times to be the same -> create a template file
      if not os.path.exists( optimization_file_path ):
         print( f"Optimization excel file '{optimization_file_path}' not found! Quitting..." )
         exit(1)

      print( f"Writting to optimization xlsx file: {optimization_file_path}" )

      # Load the file
      ws_name = input_file_name.replace("decomposition-benchmark-", "")

      optimization_dir = os.path.dirname( optimization_file_path )
      optimization_precision = os.path.basename( optimization_dir )
      optimization_machine = os.path.basename( optimization_file_path ).split("-optimization")[0]

      print( f"Optimization dir: {optimization_dir}" )

      if optimization_precision != log_precision or optimization_machine != log_machine:
         print( f"-> ERROR: Log file is from '{log_precision}/{log_machine}', but optimization file is '{optimization_precision}/{optimization_machine}'" )
         exit(1)

      ws_name = ws_name.replace(f"TpB_{optimization_precision}", "_threadBlocks")
      print( f"Worksheet name: {ws_name}" )

      # Check that the sheet for this threadxthread exists -> if no -> QUIT
      # Load all the columns into a df
      try:
         df_sheet_all = pd.read_excel( optimization_file_path, sheet_name=None )
         if df_sheet_all is None:
            raise Exception( "df_sheet was None! Quitting..." )
      except:
         print( f"Failed to load the sheet `{ws_name}`! Quitting..." )
         exit(1)

      print( f"Benchmark name: {benchmark_name}\n\n" )

      r = re.compile(".*threads.*")
      gpu_format = list(filter(r.match, formats))[0]

      df_sheet = df_sheet_all[ ws_name ]
      sheet_exist = False
      try:
         df_sheet[ benchmark_name ]
         sheet_exist = True
      except:
         print( f"Column '{benchmark_name}' not yet found in sheet '{ws_name}'. Adding data..." )

      if sheet_exist == True:
         print( f"Column '{benchmark_name}' already found! Quitting..." )
         exit(1)

      df_sheet[ benchmark_name ] = df[( gpu_format, 'time', '')]
      df_sheet_all[ ws_name ] = df_sheet

      #print( df_sheet_all )
      writer = pd.ExcelWriter( optimization_file_path, engine='xlsxwriter' )

      for sheet_name, df_iter in df_sheet_all.items():
         print( f"\nWriting sheet '{sheet_name}'" )
         df_iter.to_excel(writer, sheet_name=sheet_name, index=False)
      writer.save()

##################################################
# Main
##################################################

## Parse arguments
parser = argparse.ArgumentParser(description='Parse input file to load data from. Parse optimization csv to load data into')
# Optional argument
parser.add_argument('--input_file', type=str,
                    help='Optional input file path')
parser.add_argument('--optimization_file', type=str,
                    help='Optional file path of optimization csv', required=False)
args = parser.parse_args()

## Decide which input file to use
# - Default
# - From input argument
input_file_path = ""

# Don't have to supply entire path if the log file is in benchmark_results dir.
benchmark_results_dir = "/home/lukas/School/CVUT/FJFI/Research_assignment/benchmark_results/"

if args.input_file == None:
   input_file_path = "log-files/decomposition-benchmark.log"
elif os.path.exists( args.input_file ):
   input_file_path = args.input_file
elif os.path.exists( f"{benchmark_results_dir}{args.input_file}" ):
   input_file_path = f"{benchmark_results_dir}{args.input_file}"
else:
   print(f"File {args.input_file} not found! Exiting...")
   exit(1)

optimization_file_path = ""
if os.path.exists( args.optimization_file ):
   optimization_file_path = os.path.abspath( args.optimization_file )
else:
   print(f"File {args.optimization_file} not found! Cannot add optimization to file!")
   optimization_file_path = None

# Create different variables for input file
input_file_name = os.path.basename( input_file_path )
input_file_dir = input_file_path.replace(input_file_name, "")
input_file_name = input_file_name.split(".")[0]

# Create a separate log directory for this specific log file
log_dir = f"{input_file_dir}{input_file_name}"
if not os.path.exists( log_dir ):
   print( f"Creating log directory: {log_dir}" )
   os.mkdir( log_dir )

input_df = get_benchmark_dataframe( input_file_path )

print( f"Changing directory to: {log_dir}" )
os.chdir( log_dir )

# Sort by matrix columns (ascending) -> Easier to see trend
input_df.sort_values( by=['rows'], inplace=True, ascending=True )

# TODO: Parse file name of the .log file and use it as filename for the html file.
input_df.to_html( f"{input_file_name}.html" )

## Create output.html - Matrices as rows, formats as columns -> table in HTML file
# - Will contain inividual rows as matrices
# - Groups of columns are going to be formats + Best
#  - Each group will have for each matrix: bandwidth, time, diff.max speed-up (vs Dense Crout on CPU)
formats = sorted(list(set( input_df['format'].values.tolist() ))) # list of all formats in the benchmark results
formats.append('Best')

# get_multiindex will create the multiple levels of columns for output.html
multicolumns, df_data = get_multiindex( input_df, formats )

print( "Converting data..." )
result = convert_data_frame( input_df, multicolumns, df_data, 0, 2000 )

compute_speedup( result, formats )

# Replace empty values in results with NaN
result.replace( to_replace=' ', value=np.nan, inplace=True )

head_size = 25
if not os.path.exists( 'general' ):
   os.mkdir( 'general' )
os.chdir( 'general' )

processDf( result, formats, input_file_dir, input_file_name, optimization_file_path, head_size )