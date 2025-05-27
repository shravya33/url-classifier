## URL CLassification Model
This is a machine learning-based model which classifies the URLs into different risk levels (Low, Medium, High)

 ## 
 **Model Used:** Random Forest

 **Python libraries used:** pandas, numpy, scikit-learn, tldextract, matplotlib, seaborn, colorama
  
 **Dataset used:** The dataset used to train the model is in form of **.csv** files. It contains only two columns - url, type

##
**To train the model:**  ``` python train_model.py ```

##
**To test the model:**  ```python test_model.py ```

use -u or --url to analyze a single url

use -f or --file to classify URLs contained in a TXT file (with one URL per line)

use -o or --output to store the output results into a TXT file

example:
``` bash
python test_model.py -u youtube.com
python test_model.py -f urls.txt
python test_model.py -f urls.txt -o analysis_result
```
