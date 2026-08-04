# Copyright 2022 Cristóbal Alcázar
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""NIH Chest X-ray Dataset"""


import os
import datasets

from requests import get
from pandas import read_csv

logger = datasets.logging.get_logger(__name__)

_CITATION = """\
@inproceedings{Wang_2017,
	doi = {10.1109/cvpr.2017.369},
	url = {https://doi.org/10.1109%2Fcvpr.2017.369},
	year = 2017,
	month = {jul},
	publisher = {{IEEE}
},
	author = {Xiaosong Wang and Yifan Peng and Le Lu and Zhiyong Lu and Mohammadhadi Bagheri and Ronald M. Summers},
	title = {{ChestX}-Ray8: Hospital-Scale Chest X-Ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases},
	booktitle = {2017 {IEEE} Conference on Computer Vision and Pattern Recognition ({CVPR})}
}
"""


_DESCRIPTION = """\
The NIH Chest X-ray dataset consists of 100,000 de-identified images of chest x-rays. The images are in PNG format.

The data is provided by the NIH Clinical Center and is available through the NIH download site: https://nihcc.app.box.com/v/ChestXray-NIHCC
"""


_HOMEPAGE = "https://nihcc.app.box.com/v/chestxray-nihcc"


_REPO = "https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset/resolve/main/data"


_IMAGE_URLS = [
	f"{_REPO}/images/images_001.zip",
	f"{_REPO}/images/images_002.zip",
	f"{_REPO}/images/images_003.zip",
	f"{_REPO}/images/images_004.zip",
	f"{_REPO}/images/images_005.zip",
	f"{_REPO}/images/images_006.zip",
	f"{_REPO}/images/images_007.zip",
	f"{_REPO}/images/images_008.zip",
	f"{_REPO}/images/images_009.zip",
    f"{_REPO}/images/images_010.zip",
	f"{_REPO}/images/images_011.zip",
	f"{_REPO}/images/images_012.zip"
	#'https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset/resolve/main/dummy/0.0.0/images_001.tar.gz',
	#'https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset/resolve/main/dummy/0.0.0/images_002.tar.gz'
]


_URLS = {
	"train_val_list": f"{_REPO}/train_val_list.txt",
	"test_list": f"{_REPO}/test_list.txt",
	"labels": f"{_REPO}/Data_Entry_2017_v2020.csv",
	"BBox": f"{_REPO}/BBox_List_2017.csv",
	"image_urls": _IMAGE_URLS
}


_LABEL2IDX = {"No Finding": 0,
	     "Atelectasis": 1,
	     "Cardiomegaly": 2,
	     "Effusion": 3,
	     "Infiltration": 4,
	     "Mass": 5,
	     "Nodule": 6,
	     "Pneumonia": 7,
	     "Pneumothorax": 8,
  	     "Consolidation": 9,
	     "Edema": 10,
	     "Emphysema": 11,
	     "Fibrosis": 12,
	     "Pleural_Thickening": 13,
	     "Hernia": 14}


_NAMES = list(_LABEL2IDX.keys())


class ChestXray14Config(datasets.BuilderConfig):
	"""NIH Image Chest X-ray14 configuration."""
	
	def __init__(self, name, **kwargs):
	    super(ChestXray14Config, self).__init__(
		version=datasets.Version("1.0.0"),
		name=name,
		description="NIH ChestX-ray14",
		**kwargs,
	    )



class ChestXray14(datasets.GeneratorBasedBuilder):
	"""NIH Image Chest X-ray14 dataset."""


	BUILDER_CONFIGS = [
		ChestXray14Config("image-classification"),
		ChestXray14Config("object-detection"),
	]

	def _info(self):
		if self.config.name == "image-classification":
		    features = datasets.Features(
                       {
		        "image": datasets.Image(),
  		        "labels": datasets.features.Sequence(
                                     datasets.features.ClassLabel(
                                        num_classes=len(_NAMES),
                                        names=_NAMES
                                     )
                                 ),
                       }
                    )
		    keys = ("image", "labels")


		if self.config.name == "object-detection":
		    features = datasets.Features(
                       {
			"image_id": datasets.Value("string"),
			"patient_id": datasets.Value("int32"),
		        "image": datasets.Image(),
			"width": datasets.Value("int32"),
			"height": datasets.Value("int32"),
                       }
                    )
		    object_dict = {
			"image_id": datasets.Value("string"), 
			"area": datasets.Value("int64"),
			"bbox": datasets.Sequence(datasets.Value("float32"), length=4),
			}
		    features["objects"] = [object_dict]
		    keys = ("image", "objects")



		return datasets.DatasetInfo(
		    description=_DESCRIPTION,
		    features=features,
		    supervised_keys=keys,
		    homepage=_HOMEPAGE,
		    citation=_CITATION,
		)


	def _split_generators(self, dl_manager):
		# Get the image names that belong to the train-val dataset
		logger.info("Downloading the train_val_list image names")
		train_val_list = get(_URLS['train_val_list']).iter_lines()
		train_val_list = set([x.decode('UTF8') for x in train_val_list])
		logger.info(f"Check train_val_list: {train_val_list}")

		# Create list for store the name of the images for each dataset
		train_files = []
		test_files = []
		
		# Download batches
		data_files = dl_manager.download_and_extract(_URLS["image_urls"])

		# Iterate trought image folder and check if they belong to
		# the trainset or testset

		for batch in data_files:
		  logger.info(f"Batch for data_files: {batch}")
		  path_files = dl_manager.iter_files(batch)
		  for img in path_files:
		    if os.path.basename(img) in train_val_list:
		      train_files.append(img)
		    else:
		      test_files.append(img)
		
		return [
		    datasets.SplitGenerator(
			name=datasets.Split.TRAIN,
			gen_kwargs={
				"files": train_files
			}

		    ),
		    datasets.SplitGenerator(
			name=datasets.Split.TEST,
			gen_kwargs={
				"files": test_files
			}
		    )
		]

	def _generate_examples(self, files):

		if self.config.name == "image-classification":
		    # Read csv with image labels
		    label_csv = read_csv(_URLS["labels"])
		    for i, path in enumerate(files):
		        file_name = os.path.basename(path)
		        # Get image id to filter the respective row of the csv 	
		        image_id = file_name
		        image_labels = label_csv[label_csv["Image Index"] == image_id]["Finding Labels"].values[0].split("|")
		        if file_name.endswith(".png"):
		            yield i, {
			        "image": path,
			        "labels": image_labels,
			    }

