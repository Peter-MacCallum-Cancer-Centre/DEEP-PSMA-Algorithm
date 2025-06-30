Example algorithm and Docker build script for baseline model of DEEP-PSMA Grand Challenge.


To test working algorithm ensure git lfs is installed and clone from command:
git lfs clone https://github.com/Peter-MacCallum-Cancer-Centre/DEEP-PSMA-Algorithm.git



Before running, edit script 00_copy_example_cases_to_input_format.py line to match the appropriate location of the downloaded training data:

top='../CHALLENGE_DATA' #update to location of training data folder top directory as distributed. Subdirectories per case of the form 'train_XXXX'

There should now be input image data (*.mha) and json files in the test/input/interf0 subdirectory. 

That will populate the testing input directory with one case of PSMA and FDG image, threshold, organ segmentation, and image registration files in the format and directory structure as they are available on the grand challenge platform. Some explanation of the heirarchy and sample script to read in the relevant input sockets is available in the main inference.py script.

To test building a docker container once a sample case has been copied to input, run 01_do_build.sh which will build a docker environment based on Pytorch with gpu drivers and nnUNet dependencies as well as copy the supplementary script/model files into the image. This will take a little while on first run to resolve the packages.

If that completes successfully, running 02_do_test_run.sh will deploy the algorithm on the test case and output the predicted PSMA and FDG disease segmentation images into /test/output/

The heirarchy of output folders will map to the required sockets to match evaluation scripts on Grand Challenge. Overall, users should be able to focus on modifying the interf0_handler() function in the inference.py script which collects all of the case image data as SITK image objects, relevant segmentation thresholds, and the Euler 3D rigid registration objects before running our example inference function and saving out the labels.

Once the algorithm build and test scripts are working, the 03_do_save.sh script will save the docker image to a tarball tar.gz for submission on Grand Challenge. Given the large file size of the tar container images, there is also a more advanced functionality to save elements of the required scripts into the model folder which will be archived separately and can be attached along with the larger image file. This allows for updating a submission on the grand challenge web page without updating the full container image (10+ GB). For setting up a first working algorithm, it may be best to simplify the process and store all model weights in the resources/ folder (saved into container by default) or another subdirectory and update the dockerfile copy statements accordingly.

Building the tar image will take some time to complete but ensure that it has finished successfully before uploading to the platform. Once that's complete, follow the [instructions](https://deep-psma.grand-challenge.org/how-to-submit-your-algorithm/) here or on the grand challenge documentation pages. It is strongly recommended to complete one working debug submission to verify output on training cases 25, 60 and 83 before proceeding to the validation or testing leaderboard submissions.
