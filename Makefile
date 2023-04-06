data/processed/qt-coyotes-merged.json: data_management/merge_coco.py preprocessing/night_mode_detection.py preprocessing/tabular_features.py data/processed/CHIL/CHIL_uwin_mange_Marit_07242020.json data/processed/CHIL-earlier/CHIL_earlier.json data/processed/mange_images/mange_images.json data/processed/mange_Toronto/mange_Toronto.json data/processed/coyote-dens/CumberlandA.json data/processed/coyote-dens/CumberlandB.json data/processed/coyote-dens/FalconerA.json data/processed/coyote-dens/FalconerB.json data/processed/coyote-dens/KinnardA.json data/processed/coyote-dens/KinnardB.json data/processed/coyote-dens/KinnardC.json data/processed/coyote-dens/RowlandC.json data/processed/coyote-dens/RowlandE.json data/processed/coyote-dens/RowlandF.json data/processed/coyote-dens/RowlandH.json data/processed/coyote-dens/RowlandJ.json data/processed/coyote-dens/RowlandK.json data/processed/coyote-dens/RowlandL.json data/processed/coyote-dens/RowlandN.json data/processed/coyote-dens/StrathearnA.json data/processed/coyote-dens/StrathearnB.json data/processed/coyote-dens/WagnerB.json data/processed/coyote-dens/WagnerC.json
	python data_management/merge_coco.py
	python preprocessing/night_mode_detection.py
	python preprocessing/tabular_features.py

data/processed/CHIL/CHIL_uwin_mange_Marit_07242020.json: data_management/importers/CHIL_to_json.py
	python data_management/importers/CHIL_to_json.py

data/processed/CHIL-earlier/CHIL_earlier.json: data_management/importers/CHIL_earlier_to_json.py
	python data_management/importers/CHIL_earlier_to_json.py

data/processed/mange_images/mange_images.json: data_management/importers/mange_images_to_json.py
	python data_management/importers/mange_images_to_json.py

data/processed/mange_Toronto/mange_Toronto.json: data_management/importers/mange_Toronto_to_json.py
	python data_management/importers/mange_Toronto_to_json.py
