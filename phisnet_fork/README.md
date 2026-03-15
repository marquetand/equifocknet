# phisnet_fork
Fork of original PhiSNet code used for the work in CasNet. For further instructions see: https://github.com/rhjvanworkum/CasNet



### old instructions

Here we provide the code for training our PhiSNet model.
Package requirements (lower versions might work, but were not tested):
- python >= 3.7
- numpy >= 1.20.2
- torch >= 1.8.1
- cudatoolkit >= 10.2
- ase >= 3.21.1 
- apsw >= 3.34.0.r1
- tensorboardX >= 2.2

After installing the necessary packages, follow the steps below to train it on the PBE/def2-SVP dataset for water.

1) Download the tarball for the PBE/def2-SVP dataset for water from 
http://quantum-machine.org/data/schnorb_hamiltonian/schnorb_hamiltonian_water.tgz 

2) Unpack the tarball and put the file "schnorb_hamiltonian_water.db" in this folder

3) Convert the dataset to our format by running
> python3 convert_db.py
You should now have a file called "h2o_pbe-def2svp_4999.db"

4) Train PhiSNet by running
> python3 train.py @args.txt
