[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11047200.svg)](https://doi.org/10.5281/zenodo.11047200)

<b>Circle colocalzation analysis</b><br>
 Scripts used for the colocalization analaysis of Pex3 and ATG30 <br>


<b>Script developed by:</b><br>
<a href="https://www.rug.nl/staff/frank.mol/">Frank N. Mol</a><br>
<a href="https://www.rug.nl/research/zernike/molecular-biophysics/">Molecular Biophysics</a> - 
<a href="https://www.rug.nl/research/zernike/molecular-biophysics/vlijm-group/">Vlijm Group</a><br>
<a href="https://www.rug.nl/research/zernike/">Zernike Institute of Advanced Materials</a><br>
<a href="https://www.rug.nl/">University of Groningen</a><br><br>
The operations on the data are separated to enable batching manual interactions and running time.<br>
Before running, make sure a dataset is present in `data\sted` containing pairs of images:
- `NXXX` for STED images.
- `NXXX_conf` for corresponding confocal image.

The order of scripts is given by the number in their respective names, which is:
1. `1_combine_sted_conf.py` Combines images in a stack with different shapes, i.e. confocal and STED.<br>
2. `2_click_circles.py` Initializes the centers of the rings.<br>
3. `3a_fit_circles.py` Fits circles on the images using the initialized centers. <br>
3. `3b_fit_small.py` Fits small circles, as alternative to 3a.<br>
4. `4_confirm_circles.py` Confirm if the automatic fit is correct.<br>
5. `5_manual_circles.ipynb` Update, adjust or remove images selected in step 4.
6. `6_get_data.py` is used with final line profiles to generate the colocalization statistics.<br>
7. `7_generate_statistics.py` is used with final line profiles to generate the colocalization statistics.<br>
