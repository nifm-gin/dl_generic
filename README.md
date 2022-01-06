# DL-Generic : Internal project at Grenoble Institut of Neuroscience
* Barbier Team
* Developed by Stenzel Cackowski

## Installation:
* pip3 install virtualenv
* virtualenv env
* source env/bin/activate
* sudo pip3 install -r requirements.txt

## Usage
* First dispatch your data in a container folder "$data" in 3 folder "$train" "$val" "$test", using the BIDS nomenclature $data/$train/$subject_id/${subject_id}_${sequences_name}.nii.gz
* Then generate patches to feed your neural network using the patches_generation.py script. --help will list required or optional additional parameters
* Then you can generate and train you model using the "use_model.py" script. --help will list required or optional additional parameters
* In case of generation / segmentation usage, you might need to reconstruct infered output using the reconstruction.py script


