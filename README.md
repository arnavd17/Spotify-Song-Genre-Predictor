# Spotify-Song-Genre-Predictor

![Spotify](https://www.thedigitalmediazone.com/wp-content/uploads/2018/11/PSX_20181121_153657-1024x571.jpg)
## INTRODUCTION
Launched on October 2008, the Spotify platform provides access to over 50 million tracks. Users can browse by parameters such as artist, album, or genre, and can create, edit, and share playlists. 
Companies these days are trying extremely hard to attract customers by providing specific recommenadations to users. Genres are an important part of understanding the likes/dislikes of the users of a music streaming platform.

## PROBLEM STATEMENT
Given a song with attributes like - liveliness, valence, acousticness, danceability etc, classify the genre of the song.

## APPROACH AND CONCEPTS
* **Data cleaning** - The data consisted of 130k songs which belonged to different genres. For our analysis we decided to keep only 7 genres including - pop, rock, blues, country, classical, hiphop and jazz. An important step to cleaning the data for ease of use and understanding was the reomoval of cross-genres like - pop-rock, blues-rock etc. Also, cut songs less than 2 minutes to eliminate potential “dirty” data(ie songs that lasted one second)
 
* **Undersampling** - Our data was skewed towards pop songs with a total of 65k songs that belonged only to that category. While not the best method to use for removing class imbalance, we decided to go with undersampling as it gave us the best results.

## MODELS USED
* **k-NN** 
* **Classification Tree**
* **Bagging** 
* **Boosting**
* **Random Forest** 

## Results
* Random Forest performed the best after rebalancing the data.
* The overall classification accuracy was improved from a baseline accuracy of 14% (7 classes) to 61%.

## Future Recommendations
* Use of better methods for class rebalancing
* Use of more adaptive models that can take care of the problem with classifying cross-genres 


## REPOSITORY DETAILS
Access -
1. [Final Presentation - Summary of Approach and Results](https://github.com/arnavd17/Spotify-Song-Genre-Predictor/blob/master/SpotifySongGenrePredictor.pdf)
2. [Final Code](https://github.com/arnavd17/Spotify-Song-Genre-Predictor/blob/master/Final_Code.R)
3. [Genre List](https://github.com/arnavd17/Spotify-Song-Genre-Predictor/blob/master/data/genre_list.txt)
