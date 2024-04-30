# virtual_patient

## Model Setup Instructions

To set up the necessary model files for this project, follow these steps:

1. **Ensure Required Libraries Are Installed**:
   Make sure you have the `audeer` library installed, which will be used for downloading and extracting the model files. You can install it using pip:

   ```bash
   pip install audeer

1. Install the requirements : 

    pip install -r requirements.txt

2. Script to download Tone Emotion Model : 

    import os
    import audeer
  
    model_root = 'tone_emotion_model'
    cache_root = 'cache'
  
    audeer.mkdir(cache_root)
    def cache_path(file):
        return os.path.join(cache_root, file)
    
    url = 'https://zenodo.org/record/6221127/files/w2v2-L-robust-12.6bc4a7fd-1.1.0.zip'
    dst_path = cache_path('model.zip')
    
    if not os.path.exists(dst_path):
        audeer.download_url(
            url, 
            dst_path, 
            verbose=True,
        )
        
    if not os.path.exists(model_root):
        audeer.extract_archive(
            dst_path, 
            model_root, 
            verbose=True,
        )



