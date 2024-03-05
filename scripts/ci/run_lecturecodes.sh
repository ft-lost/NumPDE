#!/usr/bin/env bash
# This is a modified version of test_developers.sh

task() {
  
    if [[ $d =~ /CMakeFiles/ ]]; then #should not be checked
      return
    fi

    cd "$d"
    for file in lecturecodes*; do
        if [[ -f "$file" && -x "$file" ]]; then
            echo "Executing $file"
            if ! output=$(./$file 2>&1); then
                echo "ERROR: "
                printf "$output"
                exit 1
            fi
        fi
    done
    cd -
}

set -e
for d in ./lecturecodes/*/ ; do
    task $d
done
