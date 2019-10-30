# compression
A Fast Reference Free Genome Compression using Deep Neural Networks
****First of all you must install Anaconda to install python, next you must import keras as a high level library, and finally you must use Spyder to run the program. This program is developed on Spyder 3.2.3 (the Scientific PYthon Development EnviRonment). So it is recommended that use this envoronment to run the program correctly. a line of code has been added to check the python version and if your python version differs from '3.6.2', the program throws an exception and exits. the main library is Keras. and the version of python that has been used is 3.6.2.
This program also imports math, os, time, numpy, keras and platform, which has been written above the program code.
After those steps follow these instructions:
1. Execute this command:
   wget ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR494/SRR494099/SRR494099.fastq.gz
2. Prepare the dataset with this command:
   zcat SRR494099.fastq.gz | awk 'NR%4==2{print}' | grep -v ">" | tr -d -c "ACGT" > SEQ.txt
3. Put the "SEQ.txt" file in your directory.
****!NOTE! I have trained the model and saved it (the "auto-encoder" file), so to reduce the execution time you can just run the test.py file and test the algorithm without running the train.py file.
4. Execute the train.py file(optional)
5. Execute the test.py file
