# Create and Train a New Dataset/Model

1. mkdir `src_<site_month_seq>`
2. cd `src_<site_month_seq>`
3. mkdir `images`, mkdir `json`
4. `aws s3 cp --recursive <aws update images> images/`
5. `aws s3 cp --recursive <aws update jsons> json/`
6. run `create_dataset_from_recorded_images.ipynb`.

	up-to 3 new directiries are created, as follows:
  ```
     classified_obstacle
     classified_no_obstacle
     classified_not_decided
  ```
7. run `filter_classified_per_prediction_threshold.ipynb`

   new directories:

```
   obs_between_<lower1>_<upper1>
   ...
   no_obs_between_<lower1>_<upper1>
   ...
   not_decided_between_<lower1>_<upper1>
   ...
```
8. mkdir `mask_train`

```
   /obstcle
   /no_obstacle
   (optionally)/opposite
```      
9. Manually move images which are worth training to the 2/3 directories above
10. (if opposite):

	a. mkdir `opposite_src`
  
	b. open `remove_mask_multi.ipynb`, in the Settings section set the src_dir and dst_dir as follows:
	
	src_dir = <.../mask_train_opposite>,
	dst_dir = <.../opposite_src> -
	
	in addition, comment/uncomment lines 3 to 10 as follows:
	
	<img width="350" alt="image" src="https://user-images.githubusercontent.com/29920854/191482265-24c17b1c-2133-4854-b16f-219d2135e5d6.png">
	
	and run it.
  
        opposite_src now contains opposite candidated images without the mask
  
	c. mkdir `opposite_dst`
  
	d. open `create_opposite.ipynb`, in the Settings section set dataset_dir according to your dataset path, and run it.
	
11. mkdir `no_mask_train`
```
   /obstacle
   /no_obstacle
 ```
12. If opposite:
		copy contents of `opposite_dst/` to corresponding category in no_mask_train/
13. open `remove_mask_multi.ipynb`.

    In the Settings section:
    
    set src_dir to .../mask_train
    
    set dst_dir to .../no_mask_train
    
    comment/uncomment lines 3 to 10 as follows:
    
    <img width="370" alt="image" src="https://user-images.githubusercontent.com/29920854/191488553-fb1aac08-ab9f-4915-9395-709e54e5cb18.png">

    now `no_mask_train` contains `obstacle/` and `no_obstacle/` ready for training, without mask.
		
14. mkdir `eval`
```
   /obstacle
   /no_obstacle
```
15. Run `create_test_dataset.ipynb`

	In the Settings section:
	
	set dataset_dir according to the path of your dataset

	The input datasets should be: train: `../no_mask_train`, eval: `../eval`.
  
	Splits the train dataset into 90%/10%, the 10% is moved from the train dataset to the eval dataset.
  
	The % is configurable inside the `create_test_dataset` notebook. 
16. Copy all images from `.../no_mask_train.obstacle` and `.../no_mask_train_no_obstacle` to `rgb_6_balanced/sites/<name of the site where images were sourced>`.
17. Copy the images from `.../no_mask_train/obstacle` and `.../no_mask_train/no_obstacle` to `rgb_6_balanced/train/obstacle` and `rgb_6_balanced/train/no_obstacle`, correspodingly.
18. Copy the images from `.../eval/obstacle` and `.../eval/no_obstacle` to `rgb_6_balanced/eval/ostacle` and `rgb_6_balanced/eval/no_obstacle`, correspondigly.
19. Run `sample_weights_multi.ipynb` to create the new dataset

	Name the new dataset in the Settings section, under "out folder = ...".
  
	Set the parameters under "# Parameters used in the diff_metric to diff_coef assignent function"
	        as required (see the [TBD] separate documentation on the sample_weights parameters and their effect on sample_weight values)
20. In s3: make a new folder under the `obstacles-classification` bucket.

	The new folder's name should be exactly as the new dataset's name assigned in # 16 under "out_folder = ...". 
21. Upload the dataset to s3 from the Ubuntu terminal's command_line interface, using:

	`cd ../<one directory above the newly created dataset>`
  
	`aws s3 cp --recursive <dataset_name>/ s3://obstacles-classification/<dataset_name>/`
22. From the browser: 

  Browse AWS. 
  
  Login to the 634ai account. 
  
	From the top dashboard:
  
  <img width="400" alt="image" src="https://user-images.githubusercontent.com/29920854/190910506-484c7195-2ad2-4c9f-ba33-fc4b1a4e0fdb.png">

  <img width="200" alt="image" src="https://user-images.githubusercontent.com/29920854/190910976-3d1b9d46-2ab0-4a16-9b73-bd1b8e6997d3.png">
  
  <img width="400" alt="image" src="https://user-images.githubusercontent.com/29920854/190911113-9b941caf-8a33-4987-975b-fd26ddd19321.png">

  <img width="400" alt="image" src="https://user-images.githubusercontent.com/29920854/190911241-3e0aed73-6e90-4e9b-8ae7-520f3e7d763d.png">
  
  <img width="340" alt="image" src="https://user-images.githubusercontent.com/29920854/190911435-5958ac11-fc3f-489c-b13d-e7c0b709ee10.png">
  
  <img width="318" alt="image" src="https://user-images.githubusercontent.com/29920854/190911746-18cf4230-5dc2-44fb-9920-00761aa8f68d.png">

  set the train_dataset_name/eval_dataset_name to be the same as the dataset's name set in # 16
  
  <img width="338" alt="image" src="https://user-images.githubusercontent.com/29920854/190912402-fc09515a-7bc7-4c31-be3b-e85eda2224b2.png">
  
	Check in the page's bottom line whether the kernel is in Idle state, like this:
  
  <img width="350" alt="image" src="https://user-images.githubusercontent.com/29920854/190912653-d9001f67-168d-402c-9998-033009aa30a8.png">
  
  if not, refresh the page in the browser.
  
  <img width="383" alt="image" src="https://user-images.githubusercontent.com/29920854/190913684-8618349f-1a48-48bd-9f3c-53b3d41d7546.png">
  
	if you are practicing, set the no. of epochs to be low (30 is for a full training):
  
  <img width="305" alt="image" src="https://user-images.githubusercontent.com/29920854/190914182-e27b6d78-d28b-45f0-a6cb-d0496f779b7d.png">

  launch the training:
  
  <img width="328" alt="image" src="https://user-images.githubusercontent.com/29920854/190914303-03339017-56dc-4555-8e60-7eab922db184.png">

23. Open the Training Dashboard to trace the training's progress:

  <img width="320" alt="image" src="https://user-images.githubusercontent.com/29920854/190916378-557a6787-e408-4a31-99c8-740f25338206.png">

  <img width="254" alt="image" src="https://user-images.githubusercontent.com/29920854/190916557-8f069251-43fb-4b4c-96ad-c94d1d6847c6.png">

  click on names corresponding to model's name, till a new dashboard is launched at the right side 
  
  <img width="478" alt="image" src="https://user-images.githubusercontent.com/29920854/190916678-d151bba4-44d7-4a72-9cd8-71e36ea47aaf.png">
  
  Uncheck Show Right Sidebar to view a wider dashboard:
  
  <img width="305" alt="image" src="https://user-images.githubusercontent.com/29920854/190916842-c10b3660-eacd-4c20-bc2f-6d0afa5a6def.png">
  
  The metrics view starts empty, till the training code and data have been loaded, then it starts fill-in (need patience !):

  <img width="533" alt="image" src="https://user-images.githubusercontent.com/29920854/190916941-82bac11b-da95-440b-bead-c68748217609.png">

	A full training of 30 epochs may take 5 hours.
24. When training is resumed, see the s3 model's path from the training dashboard:

  <img width="400" alt="image" src="https://user-images.githubusercontent.com/29920854/190917414-bff40097-7639-4d93-a8c4-845e52ae6a1c.png">


	Browse to the location in s3
	Checkmark the model & download it 
	Copy the model to .../cs_video_processor/models/
25. Run `model_evaluation_custom_pp.ipynb`

	In the Settings section (last section) set model_path and dataset according to the model and dataset's path/name:
  
  <img width="634" alt="image" src="https://user-images.githubusercontent.com/29920854/190971693-adae5757-c2c2-4197-b89f-b0bf9f34a058.png">
  
  Set the list of single-threshold values you want to examine metrics for: 
  
  <img width="629" alt="image" src="https://user-images.githubusercontent.com/29920854/190972371-2cc94ded-01cf-427a-b0a7-e89b48d43b96.png">

	See the results for each threshold in the list:
  
  <img width="280" alt="image" src="https://user-images.githubusercontent.com/29920854/190972792-28ecb310-a064-407d-9f0f-ca5729c6ef65.png">

## An Explanation for the model_evaluation_custom_pp.ipynb Results
  
### Definitions

#### TP (True Positives)                  
All predictions where the Ground Truth is obstacle, and prediction > single-threshold value 

#### TN (True Negatives)                  
All predictions where the Ground Truth is no_obstacle, and prediction <= single-threshold value 

#### FP (False Positives)                 
All predictions where the Ground Truth is no_obstacle, and prediction > single-threshold value 

#### FN (False Negatives)                 
All predictions where the Ground Truth is obstacle, and prediction <= single-threshold value 

#### Recall    
The fraction of true-positives from all ground-true positives:

$\frac{TP}{TP+FN}$

A lower recall indicates a higher percentage of false-negatives. 

#### Specifity
The fraction of true-negatives from all ground-true negatives:

$\frac{TN}{TN+FP}$

A lower specifity indicates a higher percentage of false-positives.

#### Confustion Matrix Displayed as the Output of model_evaluation_custom_pp.ipynb

<img width="456" alt="image" src="https://user-images.githubusercontent.com/29920854/190986558-93df7ab3-4333-4a81-b243-293d92da8155.png">

For a comparable metric with other models, we usually use the accuracy metric:

#### Accuracy

The fraction of true predictions from all predictions:

$\frac{TP+TN}{TP+TN+FP+FN}$

# Find Lower and Upper Thresholds

Run `find_2_thresholds.ipynb`

The algorithm is explained inside the notebok, before the core-algorithm function `find_thresholds`:

<img width="679" alt="image" src="https://user-images.githubusercontent.com/29920854/191006922-12889632-6222-489c-882e-b235d60629ab.png">

Before running set the actual model and dataset paths:

<img width="646" alt="image" src="https://user-images.githubusercontent.com/29920854/191009109-516c9bd1-4ab7-45b5-9c0a-3cbaff8d44a0.png">

The algorithm's recommended lower and upper thresholds are printed as output:

<img width="339" alt="image" src="https://user-images.githubusercontent.com/29920854/191009831-87133d24-0750-4e2b-9920-cdd7fc1b2811.png">

For use in the model deployment - take the value rounded into 2 digits after the decimal point (0.xx). 



	
	
	
			
