# APIAIS


### Aplication Program Interface for Artifisial Inteligence System is a set of tools that aids in the task of monitoring cameras for security purposes. 

the main two tools that have been developed for this aplication are:

## API FaVe: application Program Interface for Face Verification

the APIFaVe works in this way:

first in the APIAIS/data/notebooks/FaVe/ folder we can find several files, these notebooks will be used to clean the data in order to train an algorithm in the later 

the starting notebook named "etapa 0" has the task of merge several csv files to create a single csv file to work whit, the new database is named DB.csv 
<p align="center">
  <img src = "https://i.imgur.com/LwL5lvy.png">
</p>

now the "etapa 1" file we are going to perform a data cleaning by delleting all the special chars as "/", ",", ".", "  "(double space) and so. then reduce all the names that have an 90% of similarity(if there are homonimus people it would be take in to account on the "etapa 3" file). last we will take the N most comon names in the DB.csv in order to be recognized, here is a compromise between the number of pictures that we have and the accuaricy that we want. in the end of this step we are going to rebuild our DB.csv to change all the names of the people that has quasi-duplicates and save the new data as DB_cleaned.csv
<p align="center">
  <img src = "https://i.imgur.com/inBLDke.png">
</p>

the "etapa 2" file is just to create folders and separate the pictures by name, in other words we are going to create a folder for each different name in the DB_cleaned.csv
<p align="center">
  <img src = "https://i.imgur.com/R805WMd.png">
</p>
the "etapa 3" file is for separate people that have the same name but not the same face, in order to do this, we use the face_recognition library to make a vector of every picture, then compair it whit the rest of the pics in the folders, and the picturess whit the lower  score are droped in to another folder to be revised by hand later.  
 <br />
 <br />
the "etapa 4" is the begining of our training, we are going to take the folders whit the pictures and then use every name as a class, we are going to use the vggface pretrained net, then we retrain whit our examples and save the weights to use them later in the APIFaVe
<p align="center">
  <img src = "https://i.imgur.com/pkj7W9v.png">
</p>
<p align="center">
  <img src = "https://i.imgur.com/ojPlQof.png">
</p>
once this is done we are goint to start the last "etapa 5" file, this step is just to present some results and try to see if the result are consitent, this process is slow and done by hand revewing some of the data.    
 <br />
 <br />
now that we have the weights we can use the website and the API to recognize faces, here we have 3 choises: enrollment, live detection, and the API way, this means that you can send a picture to the API and then wait for the JSON that contains its name and folio. the tree routes for this are "/enrollment" that use a POST method for the enrollment process, "face_rec"(the API way) that uses another POST method but return a JSON whit our desired name and folio, at the end we have the id_cap page, here we feed the api whit a camera stream and do a live face recognition plataform. more details on the utils.py file
<p align="center">
  <img src = "https://i.imgur.com/8ThYcMS.png">
</p>

## API ANPR: application Program Interface for Automatic Number Plate Recognition

now for the ANPR app we have first to generate the data, i have choised a path to solve this problem the main idea is to use the technique of **image segmentation** to do so, we need to create tre data bases, the first for the image segmentation of the whole plate number, the pictures should look like this
<p align="center">
  <img src = "https://i.imgur.com/6kPknyL.png">
</p>

here we can see that we are creating a mask from the original one, the idea is to generate several hundreds of this kind of pictures then train a neural network in order to predict how the mask should look from the original picture. The seccond data base we need is one for the image segmentation of the numbers and letters in the plate as we can see in the next picture
<p align="center">
  <img src = "https://i.imgur.com/ZTt00lK.png">
</p>
and the last one is just one data for the pictures of the letters, it is worth to keep in mind that some letter cant be found the plates, such as the Q, O(letter o) y la letra I, todas por su parecido a los numeros 1 y 0


<p align="center">
  <img src = "https://i.imgur.com/a3u1904.png">
</p>

once this is complete we proced to do the dada augmentation and train for every dataset, this proceses are done whit the anpr notebooks files, when the weigths are saved on their path, we can start to perform real time ANPR analisis in the "/anpr" route.

