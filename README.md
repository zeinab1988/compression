# compression
A Fast Reference Free Genome Compression using Deep Neural Networks
1. Execute this command:
   wget ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR494/SRR494099/SRR494099.fastq.gz
2. Prepare the dataset with this command:
   zcat SRR494099.fastq.gz | awk 'NR%4==2{print}' | grep -v ">" | tr -d -c "ACGT" > SEQ.txt
3. Put the "SEQ.txt" file in your directory.
****!NOTE! I have trained the model and saved it (the "auto-encoder" file), so to reduce the execution time you can just run the test.py file and test the algorithm without running the train.py file.
4. Execute the train.py file(optional)
5. Execute the test.py file
