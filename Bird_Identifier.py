#!/usr/bin/env python
# coding: utf-8

# In[38]:


# !pip install -Uqq fastbook
import fastbook
fastbook.setup_book()

from fastbook import *
from fastai.vision.widgets import *

import warnings
warnings.filterwarnings("ignore")

import requests
import matplotlib.pyplot as plt
import PIL.Image
from io import BytesIO
import os

from IPython.display import Image
from IPython.core.display import HTML


# In[39]:


try:
  os.mkdir('images')
except:
  pass


# In[40]:


subscription_key = os.environ.get('AZURE_SEARCH_KEY', '226866064dd7492994106f599e63c3c5')
search_url = "https://api.bing.microsoft.com/v7.0/images/search"
headers = {"Ocp-Apim-Subscription-Key" : subscription_key}


# In[41]:


search_term = 'eagle'

params  = {"q": search_term, "license": "public", "imageType": "photo", "count":"150"}
response = requests.get(search_url, headers=headers, params=params)
response.raise_for_status()

# Return json file
search_results = response.json()

# Create a set of thumbnails for visualization
thumbnail_urls = [img["thumbnailUrl"] for img in search_results["value"][:16]]


# In[42]:


f, axes = plt.subplots(4, 4)
for i in range(4):
    for j in range(4):
        image_data = requests.get(thumbnail_urls[i+4*j])
        image_data.raise_for_status()
        image = PIL.Image.open(BytesIO(image_data.content))        
        axes[i][j].imshow(image)
        axes[i][j].axis("off")
plt.show()


# In[43]:


img_urls = [img['contentUrl'] for img in search_results["value"]]
len(img_urls)


# In[44]:


dest = 'images/eagle_test.jpg'
download_url(img_urls[0], dest)


# In[45]:


img = PIL.Image.open(dest)
img.to_thumb(224,224)


# In[46]:


vehicle_types = 'bald eagle','osprey','red tailed hawk'
path = Path('birds_of_prey')


# In[47]:


if not path.exists():
    path.mkdir()
    for i in vehicle_types:
        dest = (path/i)
        dest.mkdir(exist_ok=True)
        
        search_term = i
        params  = {"q":search_term, "license":"public",   
                   "imageType":"photo", "count":"150"}
        response = requests.get(search_url, headers=headers, 
                   params=params)
        response.raise_for_status()
        search_results = response.json()
        img_urls = [img['contentUrl'] for img in   
                    search_results["value"]]
        
        # Downloading images from the list of image URLs
        download_images(dest, urls=img_urls)


# In[48]:


fns = get_image_files(path)
fns


# In[49]:


failed = verify_images(fns)
failed


# In[50]:


failed.map(Path.unlink);


# In[51]:


birds = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=12),
    get_y=parent_label,
    item_tfms=Resize(128))


# In[52]:


dls = birds.dataloaders(path)


# In[53]:


dls.valid.show_batch(max_n=5, nrows=1)


# In[54]:


birds = birds.new(
    item_tfms=RandomResizedCrop(224, min_scale=0.5),
    batch_tfms=aug_transforms())
dls = birds.dataloaders(path)


# In[55]:


learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(4)


# In[56]:


interpretation = ClassificationInterpretation.from_learner(learn)
interpretation.plot_confusion_matrix()


# In[57]:


interpretation.plot_top_losses(5, nrows=1)


# In[58]:


cleaner = ImageClassifierCleaner(learn)
cleaner


# In[63]:


for idx in cleaner.delete(): cleaner.fns[idx].unlink()
for idx,cat in cleaner.change(): shutil.move(str(cleaner.fns[idx]), path/cat)


# In[65]:


dls = birds.dataloaders(path)
learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(4)


# In[66]:


interpretation = ClassificationInterpretation.from_learner(learn)
interpretation.plot_confusion_matrix()


# In[67]:


learn.export()


# In[68]:


path = Path()
path.ls(file_exts='.pkl')


# In[69]:


learn_inf = load_learner(path/'export.pkl')


# In[71]:


learn_inf.predict('images/eagle_test.jpg')


# In[72]:


learn_inf.dls.vocab


# In[73]:


btn_upload = widgets.FileUpload()
btn_upload


# In[75]:


#hide
# For the book, we can't actually click an upload button, so we fake it
btn_upload = SimpleNamespace(data = ['images/eagle_test.jpg'])


# In[76]:


img = PILImage.create(btn_upload.data[-1])


# In[77]:


out_pl = widgets.Output()
out_pl.clear_output()
with out_pl: display(img.to_thumb(128,128))
out_pl


# In[78]:


pred,pred_idx,probs = learn_inf.predict(img)


# In[79]:


lbl_pred = widgets.Label()
lbl_pred.value = f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'
lbl_pred


# In[80]:


btn_run = widgets.Button(description='Classify')
btn_run


# In[81]:


def on_click_classify(change):
    img = PILImage.create(btn_upload.data[-1])
    out_pl.clear_output()
    with out_pl: display(img.to_thumb(128,128))
    pred,pred_idx,probs = learn_inf.predict(img)
    lbl_pred.value = f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'

btn_run.on_click(on_click_classify)


# In[82]:


#hide
#Putting back btn_upload to a widget for next cell
btn_upload = widgets.FileUpload()


# In[83]:


VBox([widgets.Label('Select your bird type!'), 
      btn_upload, btn_run, out_pl, lbl_pred])


# In[ ]:




