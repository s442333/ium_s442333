# Fetch dataset
wget https://s442333.students.wmi.amu.edu.pl/weather.csv

# Count lines & print some lines
wc -l weather.csv
head -n 5 weather.csv

head -n 1 weather.csv > header.csv

# Remove header and shuffle data
tail -n +2 weather.csv | shuf > weather.csv.shuf

# Define percentages for train,dev and test sets
ten=$((${CUTOFF}/10))
twenty=$((2*${CUTOFF}/10))
eighty=$((8*${CUTOFF}/10))

# Divide set to subsets
cat header.csv > weather.csv.test
head -n ${ten} weather.csv.shuf >> weather.csv.test
cat header.csv > weather.csv.dev
head -n ${twenty} weather.csv.shuf | tail -n ${ten} >> weather.csv.dev
cat header.csv > weather.csv.train
head -n ${CUTOFF} weather.csv.shuf | tail -n ${eighty} >> weather.csv.train

# Remove tmp files
rm weather.csv.shuf
rm weather.csv
rm header.csv

# Check lines count on target files
wc -l weather.csv.*
