#!/bin/bash

# Check if a parameter is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: ./bandwidth_column_adder.sh /path_to_your_directory"
    exit 1
fi

directory_path="$1"

for filename in "$directory_path"/*.csv; do
    # Extract bandwidth from the filename by splitting on 'm' or 'M'
    bandwidth=$(basename "$filename" | awk -F'[mM]' '{print $1}')
    
    # Add the bandwidth column to the CSV file
    awk -v bandwidth="$bandwidth" -F',' 'NR==1{print $0 ",bandwidth"} NR>1{print $0 "," bandwidth}' "$filename" > tmpfile && mv tmpfile "$filename"
done

echo "Finished updating files."
