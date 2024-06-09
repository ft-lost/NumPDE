#!/bin/bash

# Check if directory is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 directory"
    exit 1
fi

# Define the directory to search
DIR=$1

# Use grep to search for 'std::system' in all source files in the directory
# -n option is used to print line number with output
# -H option is used to print the file name
OUTPUT=$(grep -r -n -H --include=\*.cpp --include=\*.h --include=\*.hpp --include=\*.cc "std::system" $DIR)

if [ -n "$OUTPUT" ]
then
    # If grep found 'std::system', output a message and exit with an error code
    while IFS=: read -r filename line match
    do
        echo "Error: 'std::system' call found at $filename:$line:"
        echo ""
        awk -v line=$line 'NR >= line-3 && NR <= line+3 { if (NR == line) printf "\033[0;31m%-5d | %s\033[0m\n", NR, $0; else printf "%-5d | %s\n", NR, $0 }' $filename
        echo ""
        echo "This is forbidden because even if the system call fails, the cpp binary will continue to run."
        echo "Please use the local wrapper 'systemcall::execute()' (from include/systemcall.h) instead."
        echo ""
    done <<< "$OUTPUT"
    exit 1
fi