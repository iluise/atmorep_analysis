#!/bin/bash

############################################################################
# Script to merge the PanguWeather forecast files into monthly data files. #
# Also converts the format from grib to netCDF for convenience.            #
############################################################################

# parameters
year="2020"
month="06"

datadir="/p/scratch/atmo-rep/scratch/"     # directory where datafiles are located
preproc_dir="${datadir}/preprocessed/"     # directory where preprocessed datafiles will be saved
to_netcdf="true"                           # flag if conversion to netcdf is desired

# get modules
echo "Load required modules..."
ml Stages/2023  GCC/11.3.0  OpenMPI/4.1.4 CDO/2.1.1

fpatt="${datadir}/output_panguweather_6h_forecast_${year}${month}*_*00.grib"
fout="${preproc_dir}/output_panguweather_6h_forecast_${year}${month}.grib"

if [ ! -d "${preproc_dir}" ]; then
  echo "Creating directory for preprocessed data: '${preproc_dir}'."
  mkdir ${preproc_dir}
fi

# extract information of interest from data files (reduce file size)
for grb_file in $fpatt; do
  echo "Processing grib-file: $grb_file"
  tmp_file=`basename ${grb_file}`
  tmp_file="${datadir}/preprocessed/${tmp_file}" # "${grb_file%.grib}.nc"
  cdo -selvar,u,v,t,q -sellevel,100000,92500,85000,70000,50000 $grb_file $tmp_file
done

# merge to monthly files
echo "Concatenating hourly files to monthly file..."
cat ${preproc_dir}/output_panguweather_6h_forecast_${year}${month}*_*00.grib > ${fout}

# convert to netCDF if desired
if [ "${to_netcdf}" = "true" ]; then
   echo "Convert to netCDF-file '${fout%.grib}.nc'..." 
   # ML: Compression slows down the processing drastically. Disabled for now...
   #cdo -z zip_2 -f nc4 copy ${fout} "${fout%.grib}.nc"
   cdo -f nc4 copy ${fout} "${fout%.grib}.nc"
fi 

echo "Preprocessing done! Cleaning up..."
rm ${preproc_dir}/output_panguweather_6h_forecast_${year}${month}*_*00.grib

echo "Done!"



