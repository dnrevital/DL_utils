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

# How the Sample Weights are Calculated

## Relevant Notebooks

The notebook where the sample weights for training are calculated is `sample_weights_multi.ipynb`.

Another utility notebook, enabling to examine the effect of some parameters graphically, is `diff_coef_curve.ipynb`.

## What are Sample Weights?

Sample Weights are floating-point numbers, which are possibly attached to every sample (training image) of the image process. During the training process, the training function calculates the loss function, providing weights to the samples according to each one's sample_weights, such that a higher sample weight has a higher effect on the loss function. This enables to pre-set samples which, according to some criteria, represent harder cases for the model, thus require the model to put more attention to them, comparing to other samples. 

## Our Model's Criteria for Sampe Weights

Examine the diff image between the reference image and the current image.

Mask Image: Generated by thresholding the diff image, according to a given threshold, so that pixles below threshold = 0 and pixels above threshold = 255. 

ideally, we'd like that:
- The mask generated from obstacle images will have a large white area, as obstacles are featured by a salient difference between reference and current images.
- The maks generated from no_obstacle images will have a small white area, as no obstacles are featured by near-identical reference and current images.

We want to compensate for less ideal cases, such that:
- Obstacle images with a lower white area will be assigned a higher sample weight
- No obstacle images with a higher white area will be assigned a higher sample weight

The parameters defined in the Settings section of `sample_weights.ipynb` constitute the criteria for what's considered a "high" area or a "low" area of white in the mask image. Another parameter, swc (sample weight coefficient), indicates the factor by which the calculated sample_weight is multiplied. The default sample_weight, for images not requiring compensation, is 1.0. The overall sample_weight after calculating the required compensation is `1.0 + calc_weight * swc`, where calc_weight is the result of all other calculations, for the required compensation. 

## Calculating the Required Sample Weight

For each image in the training set:
- Build the mask image for a given threshold, as described above. The thresholds are constant per site.
- Calculate `diff_metric`, which is the fraction (0.0-1.0) of white pixles from the whole mask image.
- Find the average and std (standard deviation) of the `diff_metric`s of all training-set images, per each of the classes (obstacle, no_obstacle).

After calculating diff_metric's mean and std, for each image:
- diff_threshold is
--  `obstacles mean diff_metric + std_diff_threshold * obstacles std` for obstacle images,
--  `no_obstacles mean diff_metric - std_diff_threshold * no_obstacles std` for no_obstacle images
- `std_diff_threshold` is one of the parameters defined in the Settings section. It defines the factor by which the std is multiplied, to mark the threshold point between 'ideal' image_samples and image_samples requiring compensation. 
- `sigma_dist' is the distance, in units of the std, between the diff_metric of the current image and diff_threshold. The farther the image's diff_metric (% area of white) from the diff_threshold - the more compensation sample_weight it's assigned. "farther" is according to the class - lower for obstacle, higher for no_obstacle. 
- The 'sigma_dist' is passed through a function which uses a curve function to determine the required copensation weight, according to `sigma_dist`. The input to the curve function is the `sigma_dist` value, the output is a value between 0.0 and 1.0, which is detemined by the curve. The curve is controlled by 3 parameters (alfa, beta, gamma) which provide capabilities to determine its shape. A typical curve looks like this:
- ![image](https://user-images.githubusercontent.com/29920854/192759498-9b971484-a92e-49a0-9d87-1acf10f949ae.png)
- The explanation for how to use alfa, beta, gamma to control the shape is documented at the beginning of the diff_coef_curve.ipynb notebook. 
- The weight resulting from the curve function, is used in the final weight calculaiton is `1.0 + calc_weight * swc`, as explained above. 

## Examples

The following examples are taken from a dataset, where the diff_metric's mean and std are as follows:

| class         | mean          | std           |
| ------------- | ------------- | ------------- |
| obstacle      | 0.1647        | 0.1088        |
| no obstacle   | 0.0819        | 0.1146        |

The general parameters were set as follows:

- alfa = -3.5
- beta = 2.0
- gamma = 8
- swc = 2.0 # sample weight coefficient
- diff_threshold = 50
- std_threshold_dist = 1.5 # Distance from std to apply sample_weight correction

### Example 1

This is an example of an obstacle image requiring moderate compensation, as its white area's percent (26%) is a bit below the diff_threshold, which is 32.8%.

class: obstacle

input (ref, current):

<img width="275" alt="image" src="https://user-images.githubusercontent.com/29920854/192804006-778ee597-c235-48ed-afbd-a291ab977d73.png">

diff:

<img width="179" alt="image" src="https://user-images.githubusercontent.com/29920854/192804293-f52b16a7-d234-4973-947d-0a5e55ae1f1a.png">

mask:

<img width="182" alt="image" src="https://user-images.githubusercontent.com/29920854/192777864-1c6ad2e0-d8f5-48a7-946a-060f84392d15.png">

diff_metric: 0.2656

diff_threshold: 0.3280

sigma_dist: -0.5732

diff_coef: 0.5127

<img width="422" alt="image" src="https://user-images.githubusercontent.com/29920854/192793395-41f3e7f6-92b6-4fe7-9ab3-24effb6522cc.png">

sample_weight: 2.0255

### Example 2

This is an example of an obstacle image not requiring any compensation, as its white area's percent (32.5%) is very aligned with the diff_threshold, which is 32.8%.

class: obstacle

input (ref, current):

<img width="278" alt="image" src="https://user-images.githubusercontent.com/29920854/192806035-03d813e9-2bbd-4a6b-8ed9-4edeb86f59b1.png">

diff:

<img width="222" alt="image" src="https://user-images.githubusercontent.com/29920854/192806265-f0bf6199-abae-450b-a719-de2105fe9f71.png">

mask:

<img width="222" alt="image" src="https://user-images.githubusercontent.com/29920854/192810079-90156e75-33b4-470b-82ba-4fc709393b42.png">

diff_metric: 0.3252

diff_threshold: 0.3280

sigma_dist: -0.0250

diff_coef: 2.2697e-07

<img width="433" alt="image" src="https://user-images.githubusercontent.com/29920854/193009808-b908c693-4b0b-4e06-b01c-8318db1d2f9d.png">

sample_weight: 1.0000

# Other Useful Tools

## Finding Similar Images

This tool is very helpful for analyzing issues you have with the dataset. It receives as input a single image, and outputs all images which are similar to that images, in a similarity degree bigger than a goven threshold. Similarity degree between 2 images im1, im2 is a value between 0 and 1, calculated as '1 - cosine_distance(fs(im1), fs(im2))', where fs(.) is the feature set of the image, as calculated by a trained CNN model. 

A typical scenario fo using this tool is as follows:
- The model has false predictions an image or for a set of similar images (this is identified manually)
- You suspect that the a one or more samples of a similar image is present in the wrong class for training
- You run the image similarity tool, providing the problematic image as an input, and specifying the tool to search for the image in the suspect wrong class's training directory
- If you found such similarities - clearly you need to move the located wrong images to the correct class's dorectory for the next training

## How to Run the Image Similarity Tool

1. Open image_similarity_feature_vec_multi.ipynb
2. In the Settings section:

- Specify in im1_dir, im1_name the directory and image name of the image you want to search similarities for

- Specify in search_dir the path of the directory where you want search for images similar to the given image

- Specify in sims_base_path the root path for the resulting images to be present. The given directory will be generated automatically, and under it there will be directories containing the similar images according to specific search directory and similary-threshold

- Specify in sim_thresh the similarity threshold. Only images whith a similarity degree above this threshold to the given image will be displayed. 


	
	
	
			
