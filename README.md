# Yeast cell and budding detection

Detecting yeast cells and their budding (multiplication) behavior in brightfield microscropy images.

## Participants

 * dr. Andreas Milias Argeitis, principal investigator, University of Groningen, Faculty of Science and Engineering
 * MSc. Paolo Guerra, second principal investigator, University of Groningen, Faculty of Science and Engineering
 * MSc Herbert Teun Kruitbosch, data scientist, University of Groningen, Data science team

(The data science team is a group of 10 data scientists and alike that assist researchers from all faculties with data science and scientific programming, as part of the universities Center of Information Technology)

## Project description

**Goal** Finding out the budding moment in time and identify the mother and child yeast cells based on brightfield microscopy images. 

**Data** The brightfield images are grayscale 512x512 images that measure a black-to-white border for each yeast cell, which is cause because the lense is slightly out of focus w.r.t. the prepared surface, measuring the surface deformation of the transparant cells by there volume. These are measured every 5 minutes, such that a movie of 200 to 300 frames is captured. 

<table>
  <tr>	
    <td><img src="images/yeast-movie.gif"/>
  </tr>
  <tr>
    <td>Figure 1. Example of a mother cell with several buddings</td>
  </tr>
</table>

<table>
  <tr>	
    <td><img src="images/brightfield-concept.gif"/>
  </tr>
  <tr>
    <td>Figure 2. Concept of recording a brightfield image</td>
  </tr>
</table>
 
Although we had a lot of data, we didn't have many labels. Since the images look relatively easily, we've created synthetic data by rendering black and while elipses with elastic deformations and other types of noice, for training. Figure 3 shows predictions of both synthetic and actual data, made by a model trained on synthetic data.


<table>
  <tr>	
    <td><img src="images/synthetic-data.png"/>
  </tr>
  <tr>
    <td>Figure 3. Examples of synthetic data, the red areas are the labels.</td>
  </tr>
</table>



 
**Motivation** Budding behavior is important to understand cell multiplication in various biological research questions. The best tool so far is [yeast spotter](https://academic.oup.com/bioinformatics/article/35/21/4525/5490207), which allows very good detection of cells, but only has heuristics to find the budding moment and nothing to find the mother. Therefore researcher often do many manual annotations, and hence automation would speed up this type of research.

**Cell detection results** Using the synthetic data for training, we can get decent results on detecting cells. We're not interested in finding all cells, however the precision should be high to avoid conclusions based on false positives.

<table>
  <tr>	
    <td><img src="images/predictions.png"/>
  </tr>
  <tr>
    <td>Figure 4. Predictions by a model trained on synthetic data.</td>
  </tr>
</table>

**Boundary detection** Based on the cell location, we tracked the boundary via seam carving on a polar-coordinate transform. By transforming the polar coordinates, and finding the path along the angle-axis from 0 to 2 pi with highest or lowest cummulative pixel values, we get an estimate of where the boundary is. Highest will locate the white, and lowest the black boundary. We took the average of both as the boundary location.

<table>
  <tr>	
    <td>
        <video width="320" height="240" controls>
          <source src="images/boundary-example-01.mp4" type="video/mp4">
        </video>
    </td>
    <td>
        <video width="320" height="240" controls>
          <source src="images/boundary-example-02.mp4" type="video/mp4">
        </video>
    </td>
  </tr>
  <tr>
    <td colspan="2">Figure 5. Examples of boundary detection ausing seam carving.</td>
  </tr>
</table>


## Implementation

We created a notebook that implements the entire pipeline.

 * [Entire pipeline](notebooks/aaa.ipynb)


## Usage

We also created a tool to create either jsons or tiffs with the results.

