### pre-install instructions 
- Download [Thiers13 dataset](http://www.sociopatterns.org/datasets/high-school-contact-and-friendship-networks/) and place the file `HighSchool2013_proximity_net.csv` in the folder `code/python_temp_criticality/python_real_world_networks/real_world_data/`

- Download [Workplace15 dataset](http://www.sociopatterns.org/datasets/contacts-in-a-workplace/) and place the file `tij_InVS.dat` in the folder `code/python_temp_criticality/python_real_world_networks/real_world_data/`

Make sure submodules are loaded with 
```
git pull --recurse-submodules
```
 
#### for MacOS
```bash
brew install hdf5
```

### Install instructions
```bash
python3.11 -m venv .css
source .css/bin/activate
pip install -r code/python_temp_criticality/requirements.txt
```
